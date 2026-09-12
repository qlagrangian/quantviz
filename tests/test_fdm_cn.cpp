// FDM-xx — core/pricing/fdm_cn.hpp の仕様テスト
// 格子と境界条件、CN European の BS 解析解との一致と収束次数、格子上のパリティ・Δ、
// American（PSOR）の性質（European 以上・本源的価値以上・行使境界の単調性・収束）、時間ステップの
// 終端動作を検証する。
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "quantviz/core/pricing/black_scholes.hpp"
#include "quantviz/core/pricing/fdm_cn.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::core::bs_greeks;
using quantviz::core::bs_price;
using quantviz::core::FdmCn;
using quantviz::core::FdmGrid;
using quantviz::core::FdmParams;
using quantviz::core::OptionType;

namespace {

// 標準ケース: K = 100, T = 1, r = 0.05, σ = 0.2（BS-03 の数表値 C = 10.4506, P = 5.5735 と同じ設定）。
constexpr double kK     = 100.0;
constexpr double kT     = 1.0;
constexpr double kR     = 0.05;
constexpr double kSigma = 0.2;
// S_max = 4K。log(4)/(σ√T) ≈ 6.9 標準偏差なので境界の打ち切り誤差は無視できる。
// N = 100 / 200 / 400 のどれでも K が格子点に乗る（K = N/4 番目）。
constexpr double kSmax = 4.0 * kK;

FdmParams params(OptionType type, bool american, double q = 0.0) {
    FdmParams p{};
    p.K        = kK;
    p.T        = kT;
    p.r        = kR;
    p.sigma    = kSigma;
    p.q        = q;
    p.type     = type;
    p.american = american;
    return p;
}

/// init して満期から t = 0 まで全ステップ回す。
FdmCn solve_all(const FdmGrid& g, const FdmParams& p) {
    FdmCn f;
    f.init(g, p);
    while (f.step_backward()) {
    }
    return f;
}

double intrinsic(const FdmParams& p, double s) {
    return p.type == OptionType::Call ? std::max(s - p.K, 0.0) : std::max(p.K - s, 0.0);
}

}  // namespace

TEST_CASE("FDM-01: the grid (S_max, N, M) and the boundary values at S=0 and S=S_max follow the configuration",
          "[fdm][unit]") {
    constexpr std::size_t n = 200, m = 50;
    const double          h = kSmax / static_cast<double>(n);
    const double          k = kT / static_cast<double>(m);

    SECTION("uniform grid, t = T and the payoff at maturity") {
        FdmCn f;
        f.init(FdmGrid{kSmax, n, m}, params(OptionType::Put, false));
        const std::span<const double> s = f.spots();
        const std::span<const double> v = f.values();
        REQUIRE(s.size() == n + 1);
        REQUIRE(v.size() == n + 1);
        CHECK(s[0] == 0.0);
        CHECK_THAT(s[n], WithinRel(kSmax, 1e-15));
        for (std::size_t i = 0; i <= n; ++i) {
            INFO("i=" << i);
            CHECK_THAT(s[i], WithinAbs(h * static_cast<double>(i), 1e-12));
            CHECK(v[i] == std::max(kK - s[i], 0.0));  // 満期ペイオフはそのまま置く
        }
        CHECK(f.remaining() == m);
        CHECK(f.time() == kT);
        CHECK(std::isnan(f.exercise_boundary()));  // European
    }

    SECTION("European put: V(0) = K e^{-r tau}, V(S_max) = 0") {
        FdmCn f;
        f.init(FdmGrid{kSmax, n, m}, params(OptionType::Put, false));
        REQUIRE(f.step_backward());
        CHECK(f.remaining() == m - 1);
        CHECK_THAT(f.time(), WithinAbs(kT - k, 1e-12));
        CHECK_THAT(f.values()[0], WithinRel(kK * std::exp(-kR * k), 1e-12));
        CHECK(f.values()[n] == 0.0);
        CHECK(f.last_psor_converged());  // European: 三重対角ソルバ成功
        CHECK(f.last_psor_iterations() == 0);
    }

    SECTION("American put: V(0) = K, V(S_max) = 0") {
        FdmCn f;
        f.init(FdmGrid{kSmax, n, m}, params(OptionType::Put, true));
        REQUIRE(f.step_backward());
        CHECK(f.values()[0] == kK);
        CHECK(f.values()[n] == 0.0);
    }

    SECTION("European call with q: V(0) = 0, V(S_max) = S_max e^{-q tau} - K e^{-r tau}") {
        constexpr double q = 0.03;
        FdmCn            f;
        f.init(FdmGrid{kSmax, n, m}, params(OptionType::Call, false, q));
        REQUIRE(f.step_backward());
        CHECK(f.values()[0] == 0.0);
        CHECK_THAT(f.values()[n], WithinRel(kSmax * std::exp(-q * k) - kK * std::exp(-kR * k), 1e-12));
        // 2 ステップ目は tau = 2k
        REQUIRE(f.step_backward());
        CHECK_THAT(f.values()[n], WithinRel(kSmax * std::exp(-2 * q * k) - kK * std::exp(-2 * kR * k), 1e-12));
    }

    SECTION("value_at / delta_at outside [0, S_max] or NaN give NaN") {
        FdmCn f;
        f.init(FdmGrid{kSmax, n, m}, params(OptionType::Put, false));
        CHECK(std::isnan(f.value_at(-1.0)));
        CHECK(std::isnan(f.value_at(kSmax * 1.0001)));
        CHECK(std::isnan(f.value_at(std::numeric_limits<double>::quiet_NaN())));
        CHECK(std::isnan(f.delta_at(-1.0)));
        CHECK(f.value_at(kSmax) == 0.0);  // 端点は含む
        CHECK(f.value_at(0.0) == kK);
    }
}

TEST_CASE("FDM-02: the CN European call matches Black-Scholes to a relative 1e-3 near the money (N=M=200, S_max=4K)",
          "[fdm][numeric]") {
    // 許容は 03_tdd_spec.md §2.4「FDM vs 解析解」の相対 1e-3。S_max = 4K, N = 200 → h = 2 で K は 50 番目の
    // 節点（on-node）、S = 90, 100, 110 もすべて格子点。実測（キンクのセル平均 + Rannacher 始動）:
    // 相対誤差 4.0e-5 / −5.0e-5 / 1.4e-5。精度を担うのはセル平均で、Rannacher を外しても ATM 4.8e-5。
    // セル平均なし（キンクが格子点に乗ったまま）だと ATM −9.5e-4、S = 90 で −1.7e-3 となり許容を割る。
    // 格子点でない S（95, 105）は線形補間の誤差 h²Γ/8 ≈ 1e-2 が支配するのでここでは見ない。
    const FdmCn f = solve_all(FdmGrid{kSmax, 200, 200}, params(OptionType::Call, false));
    for (const double s : {90.0, 100.0, 110.0}) {
        const double expect = bs_price(s, kK, kT, kR, kSigma, OptionType::Call);
        INFO("S=" << s << " bs=" << expect << " fdm=" << f.value_at(s));
        CHECK_THAT(f.value_at(s), WithinRel(expect, 1e-3));
    }
    CHECK_THAT(f.value_at(kK), WithinAbs(10.4506, 1e-2));  // BS-03 の数表値（目視用の粗い絶対値）

    SECTION("off-node strike (S_max = 404): the interpolation error uses up most of the 1e-3") {
        // h = 2.02、K/h = 49.505 でほぼ節点の中点。value_at(K) の線形補間誤差 h²Γ/8 ≈ 1e-2（絶対）が加わり、
        // 実測の相対誤差は 8.7e-4（S_max = 402: 6.3e-4、405.2: 7.9e-4）。1e-3 の余裕が最も薄い配置を固定する。
        // S_max = 4K を選ぶ理由もこれ（K が節点に乗る）。
        const FdmCn  g      = solve_all(FdmGrid{404.0, 200, 200}, params(OptionType::Call, false));
        const double expect = bs_price(kK, kK, kT, kR, kSigma, OptionType::Call);
        INFO("bs=" << expect << " fdm=" << g.value_at(kK) << " rel=" << (g.value_at(kK) - expect) / expect);
        CHECK_THAT(g.value_at(kK), WithinRel(expect, 1e-3));
    }
}

TEST_CASE("FDM-03: doubling N and M divides the ATM error by about 4 (second order in h and k)",
          "[fdm][numeric]") {
    // N = M ∈ {100, 200, 400}、誤差 e_N = |V_N(K) − BS(K)|。連続する誤差比が 3〜5（≈ 4 = 2²）に入ること。
    // 2 次収束を与えるのはキンクのセル平均（fdm_cn.hpp 冒頭 (1)）。実測: e100 = 2.17e-3, e200 = 5.25e-4,
    // e400 = 1.30e-4、比 4.14 / 4.03。Rannacher 始動を外しても e = 2.07e-3 / 4.99e-4 / 1.24e-4、比 4.15 / 4.04
    // で変わらない（Rannacher の役割は粗い M での Γ の滑らかさ → FDM-05 の SECTION）。
    // 時間誤差は M を 200 → 3200 にしても ATM で 2e-5 しか動かず、誤差はほぼ空間項。
    const double        expect = bs_price(kK, kK, kT, kR, kSigma, OptionType::Call);
    std::vector<double> err;
    for (const std::size_t n : {std::size_t{100}, std::size_t{200}, std::size_t{400}}) {
        const FdmCn f = solve_all(FdmGrid{kSmax, n, n}, params(OptionType::Call, false));
        err.push_back(std::abs(f.value_at(kK) - expect));
    }
    const double r1 = err[0] / err[1];
    const double r2 = err[1] / err[2];
    INFO("e100=" << err[0] << " e200=" << err[1] << " e400=" << err[2] << " ratios " << r1 << ", " << r2);
    CHECK(err[0] > err[1]);
    CHECK(err[1] > err[2]);
    CHECK(r1 >= 3.0);
    CHECK(r1 <= 5.0);
    CHECK(r2 >= 3.0);
    CHECK(r2 <= 5.0);
}

TEST_CASE("FDM-04: put-call parity C - P = S e^{-qT} - K e^{-rT} holds at every grid node to 1e-3",
          "[fdm][numeric]") {
    // 中心差分は 1 次式に対して厳密、CN の指数近似の誤差は O(k²) なので、格子上のパリティは
    // 1e-3 より数桁良く成り立つ（実測 max |C − P − (S e^{−qT} − K e^{−rT})| = 2.9e-6）。
    // 許容は絶対 1e-3（K = 100 に対する相対 1e-5 に相当し、§5.6 の「1e-3」の厳しい方の読み）。
    constexpr double q = 0.02;
    const FdmCn call = solve_all(FdmGrid{kSmax, 200, 200}, params(OptionType::Call, false, q));
    const FdmCn put  = solve_all(FdmGrid{kSmax, 200, 200}, params(OptionType::Put, false, q));
    const std::span<const double> s = call.spots();
    for (std::size_t i = 0; i < s.size(); ++i) {
        const double expect = s[i] * std::exp(-q * kT) - kK * std::exp(-kR * kT);
        INFO("i=" << i << " S=" << s[i]);
        CHECK_THAT(call.values()[i] - put.values()[i], WithinAbs(expect, 1e-3));
    }
}

TEST_CASE("FDM-05: the delta read off the grid matches the Black-Scholes delta within 1e-2", "[fdm][numeric]") {
    // 中心差分の誤差は h²/6 · V_SSS ≈ (2²/6)·1e-3 ≈ 1e-3 以下（S_max = 4K, N = 200 → h = 2）。
    // 実測 |Δ_fdm − Δ_bs| ≤ 6e-4（S = 101 の補間点を含む）。許容 1e-2 は §5.6 の値。
    for (const OptionType type : {OptionType::Call, OptionType::Put}) {
        const FdmCn f = solve_all(FdmGrid{kSmax, 200, 200}, params(type, false));
        for (const double s : {80.0, 90.0, 100.0, 110.0, 120.0}) {
            const double expect = bs_greeks(s, kK, kT, kR, kSigma, type).delta;
            INFO("call=" << (type == OptionType::Call) << " S=" << s << " bs=" << expect
                         << " fdm=" << f.delta_at(s));
            CHECK_THAT(f.delta_at(s), WithinAbs(expect, 1e-2));
        }
        // 格子点の間（S = 101）でも線形補間で連続に振る舞う
        CHECK_THAT(f.delta_at(101.0), WithinAbs(bs_greeks(101.0, kK, kT, kR, kSigma, type).delta, 1e-2));
    }

    SECTION("Rannacher start keeps Gamma free of node-to-node oscillation on a coarse time grid (N = 200, M = 10)") {
        // 離散 Γ_i = (V_{i+1} − 2V_i + V_{i−1}) / h² の 2 階差分 max|Γ_{i+1} − 2Γ_i + Γ_{i−1}|（内点 i = 2..N−2）。
        // 滑らかな解では M に依らず 2.9e-4（M = 200 の収束値 2.88e-4、call / put 同じ）。Rannacher なしの
        // CN + セル平均だと M = 10 で 4.3e-2、M = 5 で 2.8e-1 と 2 桁以上大きい（ペイオフのキンクの高周波成分を
        // CN は減衰できない）。FDM シーンを StepOnce で手送りする粗い M がこの領域なので、ここで固定する。
        // 許容 5e-4 は収束値の 1.7 倍、振動時の値の 1/90。min Γ は M = 10 だと Rannacher なしでも ≈ 0 で
        // 判別できないため使わない。
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            const FdmCn                   g = solve_all(FdmGrid{kSmax, 200, 10}, params(type, false));
            const std::span<const double> v = g.values();
            const double                  h = kSmax / 200.0;
            const auto gamma = [&](std::size_t i) { return (v[i + 1] - 2.0 * v[i] + v[i - 1]) / (h * h); };
            double     worst = 0.0;
            for (std::size_t i = 2; i + 2 < v.size(); ++i)
                worst = std::max(worst, std::abs(gamma(i + 1) - 2.0 * gamma(i) + gamma(i - 1)));
            INFO("call=" << (type == OptionType::Call) << " max|d2 Gamma|=" << worst);
            CHECK(worst < 5e-4);
        }
    }
}

TEST_CASE("FDM-06: the American put is worth at least the European put at every grid node", "[fdm][property]") {
    const FdmCn am = solve_all(FdmGrid{kSmax, 200, 200}, params(OptionType::Put, true));
    const FdmCn eu = solve_all(FdmGrid{kSmax, 200, 200}, params(OptionType::Put, false));
    for (std::size_t i = 0; i < am.spots().size(); ++i) {
        INFO("i=" << i << " S=" << am.spots()[i]);
        // 1e-9 は PSOR の停止許容（1e-10）に備えた保険。実測の min(Am − Eu) はちょうど 0 で、余裕は使われていない。
        CHECK(am.values()[i] >= eu.values()[i] - 1e-9);
    }
    // ATM の早期行使プレミアム: 参考値 American put ≈ 6.09（二項木・本実装 N = M = 800 で 6.0899）、
    // European 5.5735。N = M = 200 の実測は 6.0848（min(Am − Eu) は 0 ちょうど）。
    CHECK(am.value_at(kK) > eu.value_at(kK) + 0.4);
    CHECK_THAT(am.value_at(kK), WithinAbs(6.09, 0.02));
}

TEST_CASE("FDM-07: the American value is at least the intrinsic value at every node after every step",
          "[fdm][property]") {
    // put は q = 0、call は q = 0.03（q > 0 でないと call の早期行使は起きない）。PSOR の射影は
    // v_i = max(payoff_i, ·) なので内点は厳密に成り立ち、境界は本源的価値で下から抑えている。
    for (const FdmParams& p : {params(OptionType::Put, true), params(OptionType::Call, true, 0.03)}) {
        FdmCn f;
        f.init(FdmGrid{kSmax, 200, 200}, p);
        std::size_t step = 0;
        do {
            double worst = std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < f.spots().size(); ++i)
                worst = std::min(worst, f.values()[i] - intrinsic(p, f.spots()[i]));
            INFO("call=" << (p.type == OptionType::Call) << " step=" << step << " min(V - intrinsic)=" << worst);
            CHECK(worst >= 0.0);
            ++step;
        } while (f.step_backward());
        CHECK(step == 201);
    }
}

TEST_CASE("FDM-08: with q = 0 the American call equals the European call (no early exercise) to a relative 1e-6",
          "[fdm][numeric]") {
    // q = 0 なら C_eu ≥ S − K e^{−rτ} ≥ S − K で制約は一度も効かない。差は PSOR の停止許容（1e-10）由来。
    // 深い OTM で価格が 0 近傍の節点は相対誤差が定義しにくいので絶対 1e-8 を併用する。
    const FdmCn am = solve_all(FdmGrid{kSmax, 200, 200}, params(OptionType::Call, true));
    const FdmCn eu = solve_all(FdmGrid{kSmax, 200, 200}, params(OptionType::Call, false));
    for (std::size_t i = 0; i < am.spots().size(); ++i) {
        INFO("i=" << i << " S=" << am.spots()[i]);
        CHECK_THAT(am.values()[i], WithinRel(eu.values()[i], 1e-6) || WithinAbs(eu.values()[i], 1e-8));
    }
    CHECK(std::isnan(am.exercise_boundary()));  // 行使域なし
}

TEST_CASE("FDM-09: the American put exercise boundary S*(t) is non-decreasing towards maturity",
          "[fdm][property]") {
    // 後退反復で τ = T − t が増えるので、記録した列は非増加であること（= t → T で非減少）。
    // τ = 0 では S < K で V = ペイオフの最大節点 K − h = 98。実測の軌跡は 98, 96, 94（step 2〜6）, 92（7〜12）,
    // 90（25）, 86（50）, 84, 82, 80 と h 刻みで単調に下がり、τ = T = 1 で S* = 80（N = 800 では 81。
    // 永久プットの境界 2r/(2r+σ²) K = 71.4 より上）。
    FdmCn f;
    f.init(FdmGrid{kSmax, 200, 200}, params(OptionType::Put, true));
    const double h = kSmax / 200.0;
    CHECK_THAT(f.exercise_boundary(), WithinAbs(kK - h, 1e-12));

    double prev = f.exercise_boundary();
    std::size_t step = 0;
    while (f.step_backward()) {
        const double b = f.exercise_boundary();
        INFO("step=" << step << " S*=" << b << " prev=" << prev);
        REQUIRE(std::isfinite(b));
        CHECK(b <= prev);
        prev = b;
        ++step;
    }
    CHECK(prev > 75.0);
    CHECK(prev < 95.0);
}

TEST_CASE("FDM-10: PSOR converges within psor_max_iter for omega in {1.1, 1.5, 1.9} and reports its iterations",
          "[fdm][unit]") {
    // American put, N = M = 200, psor_tol = 1e-10（既定）。実測（1 ステップあたりの最大 / 平均反復数、
    // 最大は Rannacher の 1 ステップ目 = 半ステップ 2 回の合算）:
    //   ω = 1.1: max 22 / mean 8.4,  ω = 1.5: max 58 / mean 23.0,  ω = 1.9: max 367 / mean 157.3
    // 行列は 1 + θΔτ(σ²i² + r) が対角で強く対角優位なので最適 ω は 1 に近く、ω を上げると遅くなる。
    // 収束後の ATM 価格は ω に依らず一致する（実測 6.08483525 で全て同一、許容 1e-7）。
    SECTION("omega in {1.1, 1.5, 1.9}: every step converges and the ATM price agrees") {
        double reference = std::numeric_limits<double>::quiet_NaN();
        for (const double omega : {1.1, 1.5, 1.9}) {
            FdmParams p  = params(OptionType::Put, true);
            p.psor_omega = omega;
            FdmCn f;
            f.init(FdmGrid{kSmax, 200, 200}, p);
            std::size_t max_iter = 0, total = 0, steps = 0;
            while (f.step_backward()) {
                INFO("omega=" << omega << " step=" << steps << " iterations=" << f.last_psor_iterations());
                REQUIRE(f.last_psor_converged());
                REQUIRE(f.last_psor_iterations() >= 1);
                REQUIRE(f.last_psor_iterations() <= p.psor_max_iter);
                max_iter = std::max(max_iter, f.last_psor_iterations());
                total += f.last_psor_iterations();
                ++steps;
            }
            INFO("omega=" << omega << " max=" << max_iter
                          << " mean=" << static_cast<double>(total) / static_cast<double>(steps));
            CHECK(max_iter < 1000);  // 退行ガード（実測の数倍）
            if (std::isnan(reference)) {
                reference = f.value_at(kK);
            } else {
                CHECK_THAT(f.value_at(kK), WithinAbs(reference, 1e-7));
            }
        }
    }

    SECTION("hitting psor_max_iter reports non-convergence but still advances time") {
        FdmParams p     = params(OptionType::Put, true);
        p.psor_max_iter = 2;
        FdmCn f;
        f.init(FdmGrid{kSmax, 200, 200}, p);
        // 1 ステップ目は Rannacher（陰的 Euler 半ステップ ×2）なので反復数は 2 サブステップの合算 = 4
        REQUIRE(f.step_backward());
        CHECK_FALSE(f.last_psor_converged());
        CHECK(f.last_psor_iterations() == 2 * p.psor_max_iter);
        CHECK(f.remaining() == 199);
        // 3 ステップ目は CN 1 サブステップ: ちょうど psor_max_iter で打ち切り
        REQUIRE(f.step_backward());
        REQUIRE(f.step_backward());
        CHECK_FALSE(f.last_psor_converged());
        CHECK(f.last_psor_iterations() == p.psor_max_iter);
        CHECK(f.remaining() == 197);
    }
}

TEST_CASE("FDM-11: M calls to step_backward take t from T to 0, after which further calls are no-ops",
          "[fdm][unit]") {
    constexpr std::size_t n = 50, m = 20;
    const FdmParams       p = params(OptionType::Put, false);
    FdmCn                 f;
    f.init(FdmGrid{kSmax, n, m}, p);

    for (std::size_t i = 0; i < m; ++i) {
        INFO("step " << i);
        REQUIRE(f.remaining() == m - i);
        REQUIRE(f.step_backward());
    }
    CHECK(f.remaining() == 0);
    CHECK_THAT(f.time(), WithinAbs(0.0, 1e-12));

    const std::vector<double> frozen(f.values().begin(), f.values().end());
    CHECK_FALSE(f.step_backward());
    CHECK_FALSE(f.step_backward());
    CHECK(f.remaining() == 0);
    CHECK(f.time() == 0.0);
    for (std::size_t i = 0; i <= n; ++i) {
        INFO("i=" << i);
        CHECK(f.values()[i] == frozen[i]);  // ビット一致で不変
    }

    SECTION("init again rewinds to the payoff at t = T") {
        f.init(FdmGrid{kSmax, n, m}, p);
        CHECK(f.remaining() == m);
        CHECK(f.time() == kT);
        for (std::size_t i = 0; i <= n; ++i) CHECK(f.values()[i] == intrinsic(p, f.spots()[i]));
    }
}
