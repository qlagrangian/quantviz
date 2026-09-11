// GARCH-xx — core/stats/garch.hpp（01–08）と scenes/garch_model.hpp（09–11）の仕様テスト
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <vector>

#include "quantviz/core/stats/garch.hpp"
#include "quantviz/core/stats/optim.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::core::garch_filter;
using quantviz::core::garch_fit;
using quantviz::core::garch_fit_into;
using quantviz::core::garch_from_unconstrained;
using quantviz::core::garch_half_life;
using quantviz::core::garch_log_likelihood;
using quantviz::core::garch_simulate;
using quantviz::core::garch_stationary;
using quantviz::core::garch_to_unconstrained;
using quantviz::core::garch_unconditional_variance;
using quantviz::core::GarchFit;
using quantviz::core::GarchParams;
using quantviz::core::OptimOptions;

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
using Theta            = std::array<double, 3>;  // 無制約パラメータ

// 日次リターン相当の代表的な真値（持続性 α+β = 0.98、無条件分散 5e-5 ≈ 年率 11%）
constexpr GarchParams kTruth{1e-6, 0.08, 0.90};

std::vector<double> simulate(GarchParams p, std::uint64_t seed, std::size_t n) {
    std::vector<double> r(n), s2(n);
    garch_simulate(p, seed, r, s2);
    return r;
}

// テスト側の「直接計算」: フィルタ出力から定義どおりに足し上げる
double direct_log_likelihood(GarchParams p, std::span<const double> r) {
    std::vector<double> s2(r.size());
    garch_filter(p, r, s2);
    double sum = 0.0;
    for (std::size_t t = 0; t < r.size(); ++t) {
        sum += std::log(2.0 * std::numbers::pi) + std::log(s2[t]) + r[t] * r[t] / s2[t];
    }
    return -0.5 * sum;
}
}  // namespace

TEST_CASE("GARCH-01: the filter matches sigma2_t = omega + alpha r2_{t-1} + beta sigma2_{t-1} over three hand-computed steps",
          "[garch][numeric]") {
    // ω=0.1, α=0.2, β=0.5 → 無条件分散 V = ω/(1−α−β) = 0.1/0.3 = 1/3
    const GarchParams           p{0.1, 0.2, 0.5};
    const std::array<double, 4> r{1.0, -2.0, 0.5, 3.0};
    std::array<double, 4>       s2{};
    garch_filter(p, r, s2);

    // 手計算（σ²_0 = V、σ²_t は r_{t−1} までの情報で決まる）
    //   σ²_0 = 1/3
    //   σ²_1 = 0.1 + 0.2·1²    + 0.5·(1/3)   = 0.1 + 0.2  + 1/6    = 7/15  ≈ 0.4666667
    //   σ²_2 = 0.1 + 0.2·(−2)² + 0.5·(7/15)  = 0.1 + 0.8  + 7/30   = 17/15 ≈ 1.1333333
    //   σ²_3 = 0.1 + 0.2·0.5²  + 0.5·(17/15) = 0.1 + 0.05 + 17/30  = 43/60 ≈ 0.7166667
    CHECK_THAT(garch_unconditional_variance(p), WithinRel(1.0 / 3.0, 1e-12));
    CHECK_THAT(s2[0], WithinRel(1.0 / 3.0, 1e-12));
    CHECK_THAT(s2[1], WithinRel(7.0 / 15.0, 1e-12));
    CHECK_THAT(s2[2], WithinRel(17.0 / 15.0, 1e-12));
    CHECK_THAT(s2[3], WithinRel(43.0 / 60.0, 1e-12));

    SECTION("empty input is a no-op") {
        std::vector<double> none;
        garch_filter(p, none, none);
        CHECK(garch_log_likelihood(p, none) == 0.0);
    }
}

TEST_CASE("GARCH-02: with constant input r^2 = V the filter converges to the unconditional variance omega/(1-alpha-beta)",
          "[garch][numeric]") {
    // 前半は r=0 で σ² を ω/(1−β) まで下げ、後半は r² = V の定数入力で V へ戻す。
    // 後半の収束は幾何級数（率 β）: 3000 ステップで 0.9^3000 ≈ 1e-137、倍精度では厳密に V に一致する。
    const GarchParams p{1e-6, 0.05, 0.90};
    const double      V  = garch_unconditional_variance(p);  // 1e-6 / 0.05 = 2e-5
    const std::size_t n0 = 500, n1 = 3000;
    std::vector<double> r(n0 + n1, 0.0);
    for (std::size_t t = n0; t < n0 + n1; ++t) r[t] = std::sqrt(V);
    std::vector<double> s2(r.size());
    garch_filter(p, r, s2);

    CHECK_THAT(V, WithinRel(2e-5, 1e-12));
    CHECK_THAT(s2[0], WithinRel(V, 1e-12));                         // 初期値は無条件分散
    CHECK_THAT(s2[n0], WithinRel(p.omega / (1.0 - p.beta), 1e-10));  // ゼロ入力の不動点 ω/(1−β)
    for (std::size_t t = n0 + 1; t < r.size(); ++t) {
        REQUIRE(s2[t] >= s2[t - 1]);  // V へ単調に戻る
        REQUIRE(s2[t] <= V * (1.0 + 1e-12));
    }
    CHECK_THAT(s2.back(), WithinRel(V, 1e-12));
}

TEST_CASE("GARCH-03: the log-likelihood -1/2 sum(log 2pi + log sigma2_t + r_t^2/sigma2_t) matches a direct computation",
          "[garch][numeric]") {
    const auto r = simulate(kTruth, 31415, 1000);
    // 生成に使った真値と、それ以外の定常パラメータの両方で一致すること
    for (const GarchParams p : {kTruth, GarchParams{2e-6, 0.15, 0.70}, GarchParams{5e-5, 0.0, 0.0}}) {
        CHECK_THAT(garch_log_likelihood(p, r), WithinRel(direct_log_likelihood(p, r), 1e-12));
    }
    CHECK(std::isfinite(garch_log_likelihood(kTruth, r)));
}

TEST_CASE("GARCH-04: alpha + beta >= 1 is rejected as non-stationary and is unreachable through the transform",
          "[garch][property]") {
    SECTION("garch_stationary requires omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1") {
        CHECK(garch_stationary(kTruth));
        CHECK(garch_stationary({1e-6, 0.0, 0.0}));
        CHECK(garch_stationary({1e-6, 0.0, 0.999}));
        CHECK_FALSE(garch_stationary({1e-6, 0.10, 0.90}));  // α+β = 1（IGARCH）
        CHECK_FALSE(garch_stationary({1e-6, 0.50, 0.60}));
        CHECK_FALSE(garch_stationary({0.0, 0.08, 0.90}));   // ω = 0
        CHECK_FALSE(garch_stationary({-1e-6, 0.08, 0.90}));
        CHECK_FALSE(garch_stationary({1e-6, -0.01, 0.90}));
        CHECK_FALSE(garch_stationary({1e-6, 0.08, -0.01}));
        CHECK_FALSE(garch_stationary({kNaN, 0.08, 0.90}));
        CHECK_FALSE(garch_stationary({1e-6, kNaN, 0.90}));
        CHECK_FALSE(garch_stationary({1e-6, 0.08, kInf}));
    }
    SECTION("the log-likelihood of a non-stationary parameter set is -inf") {
        const auto r = simulate(kTruth, 2718, 500);
        CHECK(garch_log_likelihood({1e-6, 0.10, 0.90}, r) == -kInf);
        CHECK(garch_log_likelihood({1e-6, 0.30, 0.80}, r) == -kInf);
        CHECK(garch_log_likelihood({0.0, 0.08, 0.90}, r) == -kInf);
        CHECK(std::isfinite(garch_log_likelihood({1e-6, 0.09, 0.90}, r)));
    }
    SECTION("every theta, including extreme and non-finite values, maps into the stationary region") {
        const std::array<double, 12> grid{-1e300, -700.0, -30.0, -1.0, 0.0,  1.0,
                                          30.0,   700.0,  1e300, -kInf, kInf, kNaN};
        for (const double t0 : grid)
            for (const double t1 : grid)
                for (const double t2 : grid) {
                    const GarchParams p = garch_from_unconstrained({t0, t1, t2});
                    REQUIRE(garch_stationary(p));
                    REQUIRE(p.alpha + p.beta < 1.0);
                }
    }
}

TEST_CASE("GARCH-05: the unconstrained <-> constrained parameter transform round-trips to identity", "[garch][numeric]") {
    SECTION("params -> theta -> params (including alpha or beta deep in a tail)") {
        // α/β の割合は θ2 = log α − log β で往復するので、どちらかが 1e-15 でも相対 1e-12 で戻る
        for (const GarchParams p : {kTruth, GarchParams{0.1, 0.2, 0.5}, GarchParams{3.0, 0.01, 0.98},
                                    GarchParams{1e-8, 0.45, 0.45}, GarchParams{1e-6, 1e-15, 0.9},
                                    GarchParams{1e-6, 0.9, 1e-15}}) {
            const GarchParams q = garch_from_unconstrained(garch_to_unconstrained(p));
            CHECK_THAT(q.omega, WithinRel(p.omega, 1e-12));
            CHECK_THAT(q.alpha, WithinRel(p.alpha, 1e-12));
            CHECK_THAT(q.beta, WithinRel(p.beta, 1e-12));
        }
    }
    SECTION("theta -> params -> theta (theta2 up to +-36, theta1 up to +-10)") {
        // θ2 は両裾とも厳密（±36 で sigmoid が 1 − 2e-16 / 2e-16 になっても log α − log β で復元できる）。
        // θ1 は α+β ≈ kGarchMaxPersistence 付近の 1 − u の相殺で分解能が落ちるため（|θ1| = 10 で ≈ 3e-13）、
        // 相対 1e-12 の往復が成り立つのは |θ1| ≲ 10 まで。倍精度で α+β を保持する以上これは避けられない。
        for (const Theta th : {Theta{-13.8, 3.9, -2.4}, Theta{0.0, 0.0, 0.0}, Theta{5.0, -5.0, 5.0},
                               Theta{-3.0, 10.0, -10.0}, Theta{-13.8, 3.9, 36.0}, Theta{-13.8, -10.0, -36.0},
                               Theta{0.0, 0.0, 20.0}}) {
            const auto back = garch_to_unconstrained(garch_from_unconstrained(th));
            for (std::size_t i = 0; i < 3; ++i) {
                CHECK_THAT(back[i], WithinRel(th[i], 1e-12) || WithinAbs(th[i], 1e-12));
            }
        }
    }
    SECTION("boundary alpha = 0 survives the round trip without NaN") {
        const GarchParams p{1e-6, 0.0, 0.9};
        const GarchParams q = garch_from_unconstrained(garch_to_unconstrained(p));
        CHECK(garch_stationary(q));
        CHECK_THAT(q.omega, WithinRel(p.omega, 1e-12));
        CHECK_THAT(q.alpha, WithinAbs(0.0, 1e-100));
        CHECK_THAT(q.beta, WithinRel(p.beta, 1e-12));
    }
}

TEST_CASE("GARCH-06: MLE recovers alpha and beta within +-0.03 from a synthetic GARCH path (N = 20000, fixed seed)",
          "[garch][statistical]") {
    // 真値 ω=1e-6, α=0.08, β=0.90, N=20000, seed 固定。
    // 根拠: 漸近 SE ≈ 0.007（§4.5 の注記）なので許容 ±0.03 ≈ 4 SE（§2.4「MLE パラメータ復元」）。
    //       この seed の標本で対数尤度のヘッセ行列を数値評価すると SE(α̂) ≈ 0.004, SE(β̂) ≈ 0.005
    //       （相関 −0.86）で、0.007 は保守的な見積もり。±0.03 は 6 SE 以上に相当する。
    // 初期値は真値から離した固定値 (α, β) = (0.05, 0.80)。ω は無条件分散が同程度（6.7e-5 vs 5e-5）になる値。
    constexpr std::size_t n = 20000;
    std::vector<double>   r(n), s2(n);
    garch_simulate(kTruth, 20240912, r, s2);
    const double      ll_truth = garch_log_likelihood(kTruth, r);
    const GarchParams init{1e-5, 0.05, 0.80};

    const auto check = [&](const GarchFit& fit) {
        CHECK(fit.optim.converged);
        CHECK(garch_stationary(fit.params));
        CHECK_THAT(fit.params.alpha, WithinAbs(kTruth.alpha, 0.03));
        CHECK_THAT(fit.params.beta, WithinAbs(kTruth.beta, 0.03));
        // 最尤点の尤度は真値の尤度以上（局所解や早期停止で止まっていない）
        CHECK(fit.log_lik >= ll_truth - 1e-6);
        // 返却値の整合: log_lik = LL(params) = −optim.f、params = 変換(optim.x)、軌跡の終点 = 解
        CHECK_THAT(fit.log_lik, WithinRel(garch_log_likelihood(fit.params, r), 1e-12));
        CHECK_THAT(fit.log_lik, WithinRel(-fit.optim.f, 1e-12));
        const GarchParams from_x = garch_from_unconstrained(fit.optim.x);
        CHECK(from_x.omega == fit.params.omega);
        CHECK(from_x.alpha == fit.params.alpha);
        CHECK(from_x.beta == fit.params.beta);
        REQUIRE_FALSE(fit.optim.path.empty());
        CHECK(fit.optim.path.back() == fit.optim.x);
    };
    SECTION("Nelder-Mead") { check(garch_fit(r, init)); }
    SECTION("BFGS") { check(garch_fit(r, init, OptimOptions{}, true)); }

    // 退化した初期値: β = 0 や α+β = 0 は変換の像の端（sigmoid が飽和して勾配が厳密に 0 の平坦部）に
    // 写るので、そのままでは最適化器が動けない。ω が標本分散の桁違いに大きい初期値でも単体法は平坦な
    // 方向で偽収束する。garch_fit は初期値を内部へ射影し、境界に張り付いた結果は追加初期値で救う。
    // 計算スレッド用の入口: 同じ GarchFit を使い回すと 2 回目以降はヒープ確保が起きない（capacity 不変）。
    SECTION("garch_fit_into reuses the result's allocation and matches garch_fit") {
        // 同じ呼び出し口からの 2 回はビット一致。garch_fit（別にインライン展開されるコピー）との比較は
        // 相対 1e-12: -mfma では σ² の漸化式 ω + α r² + β σ² の FMA 縮約がコピーごとに変わりうる。
        GarchFit out;
        garch_fit_into(out, r, init);
        check(out);
        const GarchFit ref = garch_fit(r, init);
        CHECK(out.optim.iters == ref.optim.iters);
        CHECK_THAT(out.params.omega, WithinRel(ref.params.omega, 1e-12));
        CHECK_THAT(out.params.alpha, WithinRel(ref.params.alpha, 1e-12));
        CHECK_THAT(out.params.beta, WithinRel(ref.params.beta, 1e-12));
        CHECK_THAT(out.log_lik, WithinRel(ref.log_lik, 1e-12));
        const GarchFit first = out;
        const auto     cap   = out.optim.path.capacity();
        REQUIRE(cap >= out.optim.path.size());
        garch_fit_into(out, r, out.params);  // ウォームスタートで再当てはめ
        check(out);
        CHECK(out.optim.path.capacity() == cap);
        garch_fit_into(out, r, GarchParams{1e-6, 0.10, 0.0}, OptimOptions{}, true);  // BFGS・退化した初期値
        check(out);
        CHECK(out.optim.path.capacity() == cap);
        garch_fit_into(out, r, init);  // 最初と同じ入力: 再利用しても結果はビット一致
        CHECK(out.optim.path.capacity() == cap);
        CHECK(out.log_lik == first.log_lik);
        CHECK(out.optim.x == first.optim.x);
        CHECK(out.optim.path == first.optim.path);
    }
    // 当てはめが定義できないデータ（空・全て 0・NaN 混入）では最適化せず init をそのまま返す:
    // converged = false、log_lik は init での対数尤度（空なら 0、NaN 混入なら NaN）、path は x0 だけ。
    SECTION("degenerate data (empty, all zeros, NaN) returns init unchanged with converged = false") {
        const auto same = [&](const GarchFit& fit) {
            CHECK(fit.params.omega == init.omega);
            CHECK(fit.params.alpha == init.alpha);
            CHECK(fit.params.beta == init.beta);
            CHECK_FALSE(fit.optim.converged);
            CHECK(fit.optim.iters == 0);
            REQUIRE(fit.optim.path.size() == 1);
            CHECK(fit.optim.path.back() == fit.optim.x);
        };
        const std::vector<double> none;
        const GarchFit            e = garch_fit(none, init);
        same(e);
        CHECK(e.log_lik == 0.0);
        const std::vector<double> zeros(100, 0.0);
        const GarchFit            z = garch_fit(zeros, init);
        same(z);
        CHECK_THAT(z.log_lik, WithinRel(garch_log_likelihood(init, zeros), 1e-12));  // -mfma 対策で相対比較
        CHECK(std::isfinite(z.log_lik));
        std::vector<double> holed(r.begin(), r.begin() + 100);
        holed[50]        = kNaN;
        const GarchFit h = garch_fit(holed, init, OptimOptions{}, true);
        same(h);
        CHECK(std::isnan(h.log_lik));
    }
    SECTION("degenerate starts (beta = 0, alpha = beta = 0, absurd omega) still recover the truth") {
        for (const GarchParams bad :
             {GarchParams{1e-6, 0.10, 0.0}, GarchParams{5e-5, 0.0, 0.0}, GarchParams{1.0, 0.9, 0.05}}) {
            for (const bool use_bfgs : {false, true}) {
                const GarchFit fit = garch_fit(r, bad, OptimOptions{}, use_bfgs);
                CHECK_THAT(fit.params.alpha, WithinAbs(kTruth.alpha, 0.03));
                CHECK_THAT(fit.params.beta, WithinAbs(kTruth.beta, 0.03));
                CHECK(fit.log_lik >= ll_truth - 1e-6);
            }
        }
    }
}

TEST_CASE("GARCH-07: on a large sample the true parameters have a higher likelihood than perturbed ones",
          "[garch][property]") {
    // 摂動幅 0.05 は α̂, β̂ の漸近 SE ≈ 0.007 の 7 倍。ω は ×3 / ÷3（無条件分散の推定精度 ≈ 数 % に対し十分大きい）。
    // この seed では真値との尤度差は最小でも ≈ 78 nats（(0.13, 0.85)）で、MLE が真値に勝る典型量 ≈ 1 nat を大きく超える。
    constexpr std::size_t n = 20000;
    const auto            r  = simulate(kTruth, 777, n);
    const double          l0 = garch_log_likelihood(kTruth, r);
    REQUIRE(std::isfinite(l0));
    for (const GarchParams p : {GarchParams{1e-6, 0.03, 0.90}, GarchParams{1e-6, 0.08, 0.85},
                                GarchParams{1e-6, 0.13, 0.85}, GarchParams{1e-6, 0.03, 0.95},
                                GarchParams{3e-6, 0.08, 0.90}, GarchParams{1e-6 / 3.0, 0.08, 0.90}}) {
        CHECK(garch_log_likelihood(p, r) < l0);
    }
    SECTION("garch_simulate is deterministic for a given seed") {
        // 同じ呼び出し口から 2 回生成して比べる（別の呼び出し口だと -mfma の縮約が変わり 1 ulp ずれうる）
        std::array<std::vector<double>, 2> twice;
        for (auto& v : twice) v = simulate(kTruth, 777, 2000);
        CHECK(twice[0] == twice[1]);
        CHECK(simulate(kTruth, 778, 2000) != twice[0]);
        for (std::size_t t = 0; t < 2000; ++t) REQUIRE_THAT(twice[0][t], WithinRel(r[t], 1e-12));  // 長さ非依存
        std::vector<double> rr(2000), ss(2000);
        garch_simulate(kTruth, 777, rr, ss);
        CHECK_THAT(ss[0], WithinRel(garch_unconditional_variance(kTruth), 1e-12));  // σ²_0 = 無条件分散
        for (std::size_t t = 1; t < 2000; ++t) {
            const double r2     = rr[t - 1] * rr[t - 1];
            const double expect = kTruth.omega + kTruth.alpha * r2 + kTruth.beta * ss[t - 1];
            REQUIRE_THAT(ss[t], WithinRel(expect, 1e-12));
        }
    }
}

TEST_CASE("GARCH-08: the half-life ln(0.5)/ln(alpha+beta) increases monotonically with alpha+beta", "[garch][property]") {
    double prev = 0.0;
    for (int k = 1; k < 200; ++k) {
        const double      s = 0.005 * k;  // α+β ∈ [0.005, 0.995]
        const GarchParams p{1e-6, 0.3 * s, 0.7 * s};
        const double      h = garch_half_life(p);
        CHECK_THAT(h, WithinRel(std::log(0.5) / std::log(p.alpha + p.beta), 1e-12));
        CHECK(h > prev);
        prev = h;
    }
    CHECK(garch_half_life({1e-6, 0.10, 0.90}) == kInf);  // α+β = 1: ショックが減衰しない
}
