// AC-xx — core/exec/almgren_chriss.hpp の仕様テスト
// Almgren–Chriss (2000) の離散最適執行: 閉形式の軌道・取引量・期待コスト・分散と、
// kappa の定義、lambda = 0 の TWAP 極限、lambda 単調性（前倒し / フロンティア）、
// そして閉形式 E[C], V[C] とモンテカルロの一致（固定 seed, 4 SE）を検証する。
//
// 本ファイルは「実装とは独立に」論文の式を書き下す方針を取る（AC-01 / AC-07 は
// sinh 比と arccosh をテスト側で直接組み立て、ヘッダの中間関数を経由しない）。
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "quantviz/core/exec/almgren_chriss.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::core::ac_cost;
using quantviz::core::ac_cost_mc;
using quantviz::core::ac_eta_tilde;
using quantviz::core::ac_frontier;
using quantviz::core::ac_kappa;
using quantviz::core::ac_kappa_tilde;
using quantviz::core::ac_sanitize;
using quantviz::core::ac_trades;
using quantviz::core::ac_trajectory;
using quantviz::core::AcCost;
using quantviz::core::AcParams;
using quantviz::core::kAcMinEtaTilde;

namespace {

/// Almgren–Chriss (2000) §3 の数値例（Table 1 の lambda = 2e-6 行）。
/// X = 10^6 株, T = 5 日, N = 5（tau = 1 日）, sigma = 0.95 $/株/sqrt(日),
/// gamma = 2.5e-7 $/株^2, eta = 2.5e-6 $·日/株^2, epsilon = 1/16 $/株, lambda = 2e-6 /$。
/// このパラメータで eta~ = 2.375e-6, kappa = 0.846297。
constexpr AcParams kPaper{/*X*/ 1.0e6,       /*T*/ 5.0,       /*sigma*/ 0.95,
                          /*eta*/ 2.5e-6,    /*gamma*/ 2.5e-7, /*epsilon*/ 0.0625,
                          /*lambda*/ 2.0e-6, /*n_steps*/ 5};

AcParams with_lambda(AcParams p, double lambda) noexcept {
    p.lambda = lambda;
    return p;
}

AcParams with_steps(AcParams p, std::size_t n) noexcept {
    p.n_steps = n;
    return p;
}

/// 論文の式をテスト側で独立に組み立てる。
/// tau = T/N, eta~ = eta - gamma tau / 2, kappa~^2 = lambda sigma^2 / eta~,
/// cosh(kappa tau) = 1 + kappa~^2 tau^2 / 2  =>  kappa = arccosh(1 + kappa~^2 tau^2 / 2) / tau
double reference_kappa(const AcParams& p) {
    const double tau       = p.T / static_cast<double>(p.n_steps);
    const double eta_tilde = p.eta - 0.5 * p.gamma * tau;
    const double kt2       = p.lambda * p.sigma * p.sigma / eta_tilde;
    return std::acosh(1.0 + 0.5 * kt2 * tau * tau) / tau;
}

/// x_j = X sinh(kappa (T - t_j)) / sinh(kappa T),  t_j = j tau
double reference_x(const AcParams& p, std::size_t j) {
    const double kappa = reference_kappa(p);
    const double tau   = p.T / static_cast<double>(p.n_steps);
    const double t     = static_cast<double>(j) * tau;
    return p.X * std::sinh(kappa * (p.T - t)) / std::sinh(kappa * p.T);
}

std::vector<double> trajectory_of(const AcParams& p) {
    std::vector<double> x(p.n_steps + 1, -1.0);
    const std::size_t   n = ac_trajectory(p, std::span<double>(x));
    REQUIRE(n == p.n_steps + 1);
    return x;
}

std::vector<double> trades_of(const AcParams& p) {
    std::vector<double> n(p.n_steps, -1.0);
    const std::size_t   m = ac_trades(p, std::span<double>(n));
    REQUIRE(m == p.n_steps);
    return n;
}

}  // namespace

TEST_CASE("AC-01: the trajectory equals X sinh(kappa(T-t))/sinh(kappa T)", "[ac][numeric]") {
    // kappa T が O(1) 以上になる組み合わせだけを使う（sinh 比が小さい kappa T で
    // 線形近似に切り替わる領域は AC-03 の担当）。
    const AcParams cases[] = {
        kPaper,
        with_steps(with_lambda(kPaper, 1.0e-6), 20),
        with_steps(with_lambda(kPaper, 1.0e-7), 50),
        with_steps(with_lambda(kPaper, 1.0e-8), 10),
        with_steps(with_lambda(kPaper, 1.0e-5), 20),
    };
    for (const AcParams& p : cases) {
        const double kappa_t = reference_kappa(p) * p.T;
        INFO("kappa*T = " << kappa_t);
        REQUIRE(kappa_t > 0.2);

        const std::vector<double> x = trajectory_of(p);
        CHECK(x.front() == p.X);   // 端点はビット一致（丸めを入れない）
        CHECK(x.back() == 0.0);
        for (std::size_t j = 1; j < p.n_steps; ++j) {
            INFO("j = " << j);
            CHECK_THAT(x[j], WithinRel(reference_x(p, j), 1e-10));
        }
    }

    SECTION("published Table 1 row of Almgren-Chriss (2000)") {
        // 有名な数表値との照合（§2.4）。kPaper のコメントにパラメータを明記してある。
        // 株数は絶対 0.1、kappa は絶対 1e-3、E[C] と sqrt(V[C]) は絶対 0.01。
        CHECK_THAT(ac_kappa(kPaper), WithinAbs(0.8463, 1e-3));

        const double              table1[] = {1.0e6, 428598.8, 182932.8, 76295.7, 27643.4, 0.0};
        const std::vector<double> x        = trajectory_of(kPaper);
        for (std::size_t j = 0; j < std::size(table1); ++j) {
            INFO("j = " << j);
            CHECK_THAT(x[j], WithinAbs(table1[j], 0.1));
        }

        const AcCost c = ac_cost(kPaper);
        CHECK_THAT(c.expected, WithinAbs(1140715.17, 0.01));
        CHECK_THAT(std::sqrt(c.variance), WithinAbs(449367.65, 0.01));
    }
}

TEST_CASE("AC-02: the trades sum back to the full order size X", "[ac][property]") {
    // 許容 1e-9 は素朴な総和の丸め（<= N * X * eps）を吸収できる範囲で選ぶ。
    // 以下の格子は N * X <= 2e7 なので最悪でも 2.2e-9 * ... ではなく 2.2e-9 未満に収まる。
    for (const double x0 : {1.0, 100.0, 1.0e4, 1.0e6}) {
        for (const std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{5}, std::size_t{20}}) {
            for (const double lam : {0.0, 1.0e-8, 2.0e-6, 1.0e-4}) {
                AcParams p = with_steps(with_lambda(kPaper, lam), n);
                p.X        = x0;
                INFO("X = " << x0 << " N = " << n << " lambda = " << lam);

                const std::vector<double> tr  = trades_of(p);
                double                    sum = 0.0;
                for (const double v : tr) {
                    CHECK(v >= 0.0);  // 売り切りプログラムなので手仕舞い方向のみ
                    sum += v;
                }
                CHECK_THAT(sum, WithinAbs(p.X, 1e-9));

                // n_j = x_{j-1} - x_j
                const std::vector<double> x = trajectory_of(p);
                for (std::size_t j = 0; j < n; ++j) CHECK_THAT(tr[j], WithinAbs(x[j] - x[j + 1], 0.0));
            }
        }
    }
}

TEST_CASE("AC-03: lambda = 0 collapses to the linear TWAP schedule", "[ac][numeric]") {
    for (const std::size_t n : {std::size_t{1}, std::size_t{3}, std::size_t{16}, std::size_t{257}}) {
        const AcParams p = with_steps(with_lambda(kPaper, 0.0), n);
        INFO("N = " << n);
        CHECK(ac_kappa(p) == 0.0);

        const std::vector<double> x  = trajectory_of(p);
        const std::vector<double> tr = trades_of(p);
        const double              nn = static_cast<double>(n);
        for (std::size_t j = 0; j <= n; ++j) {
            const double want = p.X * (nn - static_cast<double>(j)) / nn;
            INFO("j = " << j);
            CHECK_THAT(x[j], WithinRel(want, 1e-12));
        }
        for (const double v : tr) CHECK_THAT(v, WithinRel(p.X / nn, 1e-12));
    }

    // sigma = 0（リスクが無い）も同じ極限になる。
    AcParams flat = kPaper;
    flat.sigma    = 0.0;
    CHECK(ac_kappa(flat) == 0.0);
    const std::vector<double> xf = trajectory_of(flat);
    CHECK_THAT(xf[1], WithinRel(flat.X * 4.0 / 5.0, 1e-12));
}

TEST_CASE("AC-04: a larger lambda front-loads the schedule at every interior point", "[ac][property]") {
    // lambda の上端は内点がアンダーフローしない範囲で止めてある（N = 20 で x_19 は
    // lambda = 1e-4 なら 2.9e-6、lambda = 1e-2 まで上げると 1.3e-39）。これが厳密不等号 < を
    // 意味のある検査に保っている（大きい kappa T の領域は AC-07 が有限性・単調非増加で見る）。
    const double lambdas[] = {0.0, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4};
    const std::size_t n    = 20;

    std::vector<double> prev = trajectory_of(with_steps(with_lambda(kPaper, lambdas[0]), n));
    for (std::size_t i = 1; i < std::size(lambdas); ++i) {
        const std::vector<double> cur = trajectory_of(with_steps(with_lambda(kPaper, lambdas[i]), n));
        CHECK(cur.front() == prev.front());
        CHECK(cur.back() == prev.back());
        for (std::size_t j = 1; j < n; ++j) {
            INFO("lambda " << lambdas[i - 1] << " -> " << lambdas[i] << ", j = " << j);
            CHECK(cur[j] < prev[j]);
        }
        prev = cur;
    }
}

TEST_CASE("AC-05: the closed-form cost and variance match Monte Carlo", "[ac][statistical]") {
    // C は独立な N(0,1) の線形結合 + 定数なので厳密に正規。したがって
    //   SE(mean) = sqrt(V / n_sim),  SE(sample variance) = V sqrt(2 / (n_sim - 1))
    // n_sim = 20000 で SE(mean)/sqrt(V) = 0.71 %, SE(var)/V = 1.00 %。許容はどちらも 4 SE。
    constexpr std::size_t   kNSim = 20000;
    constexpr std::uint64_t kSeed = 20260912;

    SECTION("paper example (lambda = 2e-6)") {
        const AcCost closed = ac_cost(kPaper);
        const AcCost mc     = ac_cost_mc(kPaper, kSeed, kNSim);

        const double se_mean = std::sqrt(closed.variance / static_cast<double>(kNSim));
        const double se_var  = closed.variance * std::sqrt(2.0 / static_cast<double>(kNSim - 1));
        INFO("E closed = " << closed.expected << " MC = " << mc.expected << " SE = " << se_mean);
        INFO("V closed = " << closed.variance << " MC = " << mc.variance << " SE = " << se_var);
        CHECK_THAT(mc.expected, WithinAbs(closed.expected, 4.0 * se_mean));
        CHECK_THAT(mc.variance, WithinAbs(closed.variance, 4.0 * se_var));
    }

    SECTION("TWAP (lambda = 0, N = 20)") {
        const AcParams p      = with_steps(with_lambda(kPaper, 0.0), 20);
        const AcCost   closed = ac_cost(p);
        const AcCost   mc     = ac_cost_mc(p, kSeed, kNSim);

        const double se_mean = std::sqrt(closed.variance / static_cast<double>(kNSim));
        const double se_var  = closed.variance * std::sqrt(2.0 / static_cast<double>(kNSim - 1));
        INFO("E closed = " << closed.expected << " MC = " << mc.expected << " SE = " << se_mean);
        INFO("V closed = " << closed.variance << " MC = " << mc.variance << " SE = " << se_var);
        CHECK_THAT(mc.expected, WithinAbs(closed.expected, 4.0 * se_mean));
        CHECK_THAT(mc.variance, WithinAbs(closed.variance, 4.0 * se_var));
    }

    SECTION("same seed gives bit-identical results") {
        const AcCost a = ac_cost_mc(kPaper, kSeed, 1000);
        const AcCost b = ac_cost_mc(kPaper, kSeed, 1000);
        CHECK(a.expected == b.expected);
        CHECK(a.variance == b.variance);
    }
}

TEST_CASE("AC-06: the efficient frontier is monotone in lambda", "[ac][property]") {
    constexpr std::size_t kL = 32;
    std::vector<double>   lambdas(kL);
    for (std::size_t i = 0; i < kL; ++i) {
        const double u = static_cast<double>(i) / static_cast<double>(kL - 1);
        lambdas[i]     = std::pow(10.0, -8.0 + 5.0 * u);  // 1e-8 .. 1e-3
    }
    std::vector<AcCost> out(kL);
    const AcParams      base = with_steps(kPaper, 20);
    REQUIRE(ac_frontier(base, std::span<const double>(lambdas), std::span<AcCost>(out)) == kL);

    for (std::size_t i = 1; i < kL; ++i) {
        INFO("lambda " << lambdas[i - 1] << " -> " << lambdas[i]);
        CHECK(out[i].expected > out[i - 1].expected);
        CHECK(out[i].variance < out[i - 1].variance);
    }

    // lambda = 0 が期待コストの最小・分散の最大。
    const AcCost twap = ac_cost(with_lambda(base, 0.0));
    CHECK(out[0].expected > twap.expected);
    CHECK(out[0].variance < twap.variance);

    // フロンティアの各点は ac_cost を lambda ごとに評価したものと厳密に一致する。
    for (std::size_t i = 0; i < kL; ++i) {
        const AcCost one = ac_cost(with_lambda(base, lambdas[i]));
        CHECK(out[i].expected == one.expected);
        CHECK(out[i].variance == one.variance);
    }
}

TEST_CASE("AC-07: kappa satisfies cosh(kappa tau) - 1 = kappa~^2 tau^2 / 2", "[ac][unit]") {
    const AcParams cases[] = {
        kPaper,
        with_steps(with_lambda(kPaper, 1.0e-8), 3),
        with_steps(with_lambda(kPaper, 1.0e-4), 64),
        with_steps(with_lambda(kPaper, 5.0e-6), 1),
    };
    for (const AcParams& p : cases) {
        const double tau = p.T / static_cast<double>(p.n_steps);
        INFO("lambda = " << p.lambda << " N = " << p.n_steps);

        // eta~ = eta - gamma tau / 2
        CHECK_THAT(ac_eta_tilde(p), WithinRel(p.eta - 0.5 * p.gamma * tau, 1e-14));
        // kappa~^2 = lambda sigma^2 / eta~
        const double kt = ac_kappa_tilde(p);
        CHECK_THAT(kt * kt, WithinRel(p.lambda * p.sigma * p.sigma / ac_eta_tilde(p), 1e-14));
        // 定義式そのもの
        const double kappa = ac_kappa(p);
        CHECK_THAT(std::cosh(kappa * tau) - 1.0, WithinRel(0.5 * kt * kt * tau * tau, 1e-12));
        CHECK_THAT(kappa, WithinRel(reference_kappa(p), 1e-12));
    }

    // 退化: lambda = 0 と sigma = 0 で厳密に 0。
    CHECK(ac_kappa(with_lambda(kPaper, 0.0)) == 0.0);
    AcParams zero_vol = kPaper;
    zero_vol.sigma    = 0.0;
    CHECK(ac_kappa(zero_vol) == 0.0);

    // 連続極限 tau -> 0 で kappa -> kappa~（誤差は kappa~^2 tau^2 / 24）。
    const AcParams fine = with_steps(kPaper, 4096);
    CHECK_THAT(ac_kappa(fine), WithinRel(ac_kappa_tilde(fine), 1e-6));

    // 不正入力は ac_sanitize で既定へ丸められ、kappa は有限のまま。
    AcParams bad = kPaper;
    bad.sigma    = std::nan("");
    bad.eta      = -1.0;
    bad.n_steps  = 0;
    const AcParams fixed = ac_sanitize(bad);
    CHECK(fixed.n_steps >= 1);
    CHECK(fixed.eta > 0.0);
    CHECK(std::isfinite(fixed.sigma));
    CHECK(std::isfinite(ac_kappa(bad)));
    CHECK(std::isfinite(ac_cost(bad).expected));
    CHECK(std::isfinite(ac_cost(bad).variance));
    CHECK(ac_sanitize(fixed).n_steps == fixed.n_steps);  // 冪等

    // 大きい kappa T（クランプ域の内側でも 1500 超に届く）でも NaN を出さない。
    // sinh(kappa(T-t))/sinh(kappa T) をそのまま計算すると kappa T > 710 で inf/inf = NaN になる。
    // kappa T > 745 では内点が 0 へアンダーフローするので単調「非増加」で見る。
    // 実測の kappa T はそれぞれ 1510 / 735 / 2048。
    AcParams big_a = with_steps(with_lambda(kPaper, 1.0e-3), 256);
    big_a.T        = 250.0;
    AcParams big_b = with_steps(with_lambda(kPaper, 6.0e-2), 4096);
    AcParams big_c = with_steps(with_lambda(kPaper, 1.0e-4), 1024);
    big_c.eta      = 1.0e-9;
    for (const AcParams& p : {big_a, big_b, big_c}) {
        INFO("lambda = " << p.lambda << " T = " << p.T << " N = " << p.n_steps
                         << " kappa*T = " << ac_kappa(p) * p.T);
        CHECK(ac_kappa(p) * p.T > 700.0);

        const std::vector<double> x = trajectory_of(p);
        CHECK(x.front() == p.X);
        CHECK(x.back() == 0.0);
        for (std::size_t j = 0; j <= p.n_steps; ++j) {
            INFO("j = " << j);
            REQUIRE(std::isfinite(x[j]));
            CHECK(x[j] >= 0.0);
            if (j > 0) CHECK(x[j] <= x[j - 1]);
        }

        const AcCost c = ac_cost(p);
        CHECK(std::isfinite(c.expected));
        CHECK(std::isfinite(c.variance));
        const AcCost m = ac_cost_mc(p, 7, 16);
        CHECK(std::isfinite(m.expected));
        CHECK(std::isfinite(m.variance));
    }

    // eta~ <= 0（gamma tau / 2 >= eta）の退化領域でも kAcMinEtaTilde の床で有限に留まる。
    AcParams degenerate = kPaper;
    degenerate.eta      = 1.0e-15;
    degenerate.gamma    = 1.0;  // gamma tau / 2 = 0.5 >> eta
    CHECK(ac_eta_tilde(degenerate) == kAcMinEtaTilde);
    const AcCost dc = ac_cost(degenerate);
    CHECK(std::isfinite(dc.expected));
    CHECK(std::isfinite(dc.variance));
    const AcCost dm = ac_cost_mc(degenerate, 7, 64);
    CHECK(std::isfinite(dm.expected));
    CHECK(std::isfinite(dm.variance));

    // n_sim = 0 は {0, 0}（ヘッダに明記）。
    const AcCost empty = ac_cost_mc(kPaper, 7, 0);
    CHECK(empty.expected == 0.0);
    CHECK(empty.variance == 0.0);
}
