// VOLSURF-xx — core/pricing/vol_surface.hpp（Gatheral–Jacquier SSVI）の仕様テスト
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "quantviz/core/pricing/vol_surface.hpp"

using Catch::Matchers::WithinRel;
using quantviz::core::SsviParams;
using quantviz::core::ssvi_calendar_arbitrage_free;
using quantviz::core::ssvi_clamp;
using quantviz::core::ssvi_implied_vol;
using quantviz::core::ssvi_total_variance;

namespace {

// Task 6 のシーンが評価するのと同じ 64×32 の (k, T) 格子。k ∈ [−1, 1], T ∈ [0.02, 5]。
constexpr std::size_t kNk = 64;
constexpr std::size_t kNt = 32;

double grid_k(std::size_t i) noexcept {
    return -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(kNk - 1);
}

double grid_t(std::size_t j) noexcept {
    return 0.02 + (5.0 - 0.02) * static_cast<double>(j) / static_cast<double>(kNt - 1);
}

struct Variant {
    const char* name;
    SsviParams  p;
};

/// 既定 + 各フィールドをクランプ域の端に寄せた摂動 + 端の組み合わせ（コーナー）。
/// どれも ssvi_clamp を通しても値が変わらない＝許容域の内側または境界にある。
std::vector<Variant> variants() {
    const SsviParams d{};
    auto             with = [d](auto&& mutate) {
        SsviParams p = d;
        mutate(p);
        return p;
    };
    return {
        {"default", d},
        {"rho=+0.9", with([](SsviParams& p) { p.rho = 0.9; })},
        {"rho=-0.9", with([](SsviParams& p) { p.rho = -0.9; })},
        {"eta=0.1", with([](SsviParams& p) { p.eta = 0.1; })},
        {"eta=1.9", with([](SsviParams& p) { p.eta = 1.9; })},
        {"gamma=0", with([](SsviParams& p) { p.gamma = 0.0; })},
        {"gamma=1", with([](SsviParams& p) { p.gamma = 1.0; })},
        // --- クランプ域の端（ssvi_clamp が返しうる最悪の値）
        {"sigma_atm=min", with([](SsviParams& p) { p.sigma_atm = 1e-6; })},
        {"sigma_atm=max", with([](SsviParams& p) { p.sigma_atm = 5.0; })},
        {"eta=max", with([](SsviParams& p) { p.eta = 100.0; })},
        {"eta=min", with([](SsviParams& p) { p.eta = 1e-6; })},
        {"rho=+0.999", with([](SsviParams& p) { p.rho = 0.999; })},
        {"rho=-0.999", with([](SsviParams& p) { p.rho = -0.999; })},
        // --- コーナー（端の組み合わせ。桁落ちが最も厳しいのは ρ = −0.999 側）
        {"corner: sigma=5, eta=100, rho=-0.999, gamma=1", with([](SsviParams& p) {
             p.sigma_atm = 5.0;
             p.rho       = -0.999;
             p.eta       = 100.0;
             p.gamma     = 1.0;
         })},
        {"corner: sigma=5, eta=100, rho=+0.999, gamma=0", with([](SsviParams& p) {
             p.sigma_atm = 5.0;
             p.rho       = 0.999;
             p.eta       = 100.0;
             p.gamma     = 0.0;
         })},
        {"corner: sigma=1e-6, eta=100, rho=-0.999, gamma=1", with([](SsviParams& p) {
             p.sigma_atm = 1e-6;
             p.rho       = -0.999;
             p.eta       = 100.0;
             p.gamma     = 1.0;
         })},
        {"corner: sigma=1e-6, eta=1e-6, rho=+0.999, gamma=0", with([](SsviParams& p) {
             p.sigma_atm = 1e-6;
             p.rho       = 0.999;
             p.eta       = 1e-6;
             p.gamma     = 0.0;
         })},
    };
}

// ssvi_clamp は定数式で使える（シーンが constexpr な既定値を組み立てられるように）。
static_assert(ssvi_clamp(SsviParams{2.0, -5.0, 1e9, -1.0}).rho == -0.999);
static_assert(ssvi_clamp(SsviParams{2.0, -5.0, 1e9, -1.0}).eta == 100.0);
static_assert(ssvi_clamp(SsviParams{2.0, -5.0, 1e9, -1.0}).gamma == 0.0);
static_assert(ssvi_clamp(SsviParams{}).sigma_atm == 0.2);

}  // namespace

TEST_CASE("VOLSURF-01: implied vol is finite and strictly positive over the whole (k, T) grid",
          "[volsurf][property]") {
    SECTION("every clamped parameter corner gives a finite, strictly positive surface") {
        for (const Variant& v : variants()) {
            INFO("params: " << v.name);
            const SsviParams p = ssvi_clamp(v.p);
            // 摂動はすべて許容域の内側／境界なので、クランプは恒等写像であること。
            CHECK(p.sigma_atm == v.p.sigma_atm);
            CHECK(p.rho == v.p.rho);
            CHECK(p.eta == v.p.eta);
            CHECK(p.gamma == v.p.gamma);

            double min_iv     = std::numeric_limits<double>::infinity();
            bool   all_finite = true;
            for (std::size_t j = 0; j < kNt; ++j) {
                for (std::size_t i = 0; i < kNk; ++i) {
                    const double iv = ssvi_implied_vol(grid_k(i), grid_t(j), p);
                    if (!std::isfinite(iv)) all_finite = false;
                    if (iv < min_iv) min_iv = iv;
                }
            }
            CHECK(all_finite);
            CHECK(min_iv > 0.0);

            // ATM は θ の定義そのもの: w(0, T) = σ_atm² T、iv(0, T) = σ_atm（ρ, η, γ に依らない）。
            // ついでに w = iv² T の整合も見る。
            for (std::size_t j = 0; j < kNt; j += 7) {
                const double t = grid_t(j);
                INFO("T = " << t);
                CHECK_THAT(ssvi_total_variance(0.0, t, p), WithinRel(p.sigma_atm * p.sigma_atm * t, 1e-12));
                const double iv_atm = ssvi_implied_vol(0.0, t, p);
                CHECK_THAT(iv_atm, WithinRel(p.sigma_atm, 1e-12));
                CHECK_THAT(iv_atm * iv_atm * t, WithinRel(ssvi_total_variance(0.0, t, p), 1e-12));
            }
        }
    }

    // 以下 2 つは「クランプを通せば必ず有限・正」という上の不変条件を支える前提の固定。
    SECTION("ssvi_clamp: NaN falls back to the default and out-of-range values are clamped") {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();

        // 既定値はそのまま通る（クランプは恒等写像）。
        const SsviParams d = ssvi_clamp(SsviParams{});
        CHECK(d.sigma_atm == 0.2);
        CHECK(d.rho == -0.3);
        CHECK(d.eta == 1.0);
        CHECK(d.gamma == 0.5);

        // NaN は各フィールドの既定値へ。
        SsviParams all_nan{nan, nan, nan, nan};
        const auto fixed = ssvi_clamp(all_nan);
        CHECK(fixed.sigma_atm == 0.2);
        CHECK(fixed.rho == -0.3);
        CHECK(fixed.eta == 1.0);
        CHECK(fixed.gamma == 0.5);

        // 範囲外はクランプ。σ ∈ [1e-6, 5], ρ ∈ [−0.999, 0.999], η ∈ [1e-6, 100], γ ∈ [0, 1]。
        SsviParams lo{-1.0, -5.0, -3.0, -0.5};
        const auto c_lo = ssvi_clamp(lo);
        CHECK(c_lo.sigma_atm == 1e-6);
        CHECK(c_lo.rho == -0.999);
        CHECK(c_lo.eta == 1e-6);
        CHECK(c_lo.gamma == 0.0);

        // 上限は DBL_MAX ではなく金融的な上限。inf でも有限の端点に落ちる。
        SsviParams hi{inf, 5.0, inf, 2.0};
        const auto c_hi = ssvi_clamp(hi);
        CHECK(c_hi.sigma_atm == 5.0);
        CHECK(c_hi.rho == 0.999);
        CHECK(c_hi.eta == 100.0);
        CHECK(c_hi.gamma == 1.0);
        // その端点で面が壊れないこと（VOLSURF-01 の本体はこの p も舐めている）。
        CHECK(std::isfinite(ssvi_implied_vol(-1.0, 0.02, c_hi)));
        CHECK(ssvi_implied_vol(-1.0, 0.02, c_hi) > 0.0);

        // 0 も許容域外（σ, η は正であることを要求する）。
        SsviParams zeros{0.0, 0.0, 0.0, 0.0};
        const auto c_zeros = ssvi_clamp(zeros);
        CHECK(c_zeros.sigma_atm == 1e-6);
        CHECK(c_zeros.rho == 0.0);
        CHECK(c_zeros.eta == 1e-6);
        CHECK(c_zeros.gamma == 0.0);

        // 冪等。
        CHECK(ssvi_clamp(c_lo).rho == c_lo.rho);
        CHECK(ssvi_clamp(c_lo).eta == c_lo.eta);
        CHECK(ssvi_clamp(c_hi).gamma == c_hi.gamma);
        CHECK(ssvi_clamp(c_hi).sigma_atm == c_hi.sigma_atm);
    }

    SECTION("T <= 0 and T = NaN are floored instead of producing NaN") {
        const SsviParams p   = ssvi_clamp(SsviParams{});
        const double     nan = std::numeric_limits<double>::quiet_NaN();

        CHECK(std::isfinite(ssvi_total_variance(0.3, 0.0, p)));
        CHECK(std::isfinite(ssvi_implied_vol(0.3, 0.0, p)));
        CHECK(ssvi_implied_vol(0.3, 0.0, p) > 0.0);
        CHECK(std::isfinite(ssvi_total_variance(0.3, -1.0, p)));
        CHECK(std::isfinite(ssvi_implied_vol(0.3, -1.0, p)));
        CHECK(std::isfinite(ssvi_total_variance(0.3, nan, p)));
        CHECK(std::isfinite(ssvi_implied_vol(0.3, nan, p)));
    }
}

TEST_CASE("VOLSURF-02: the default surface is calendar-arbitrage free (w non-decreasing in T)",
          "[volsurf][property]") {
    const SsviParams p = ssvi_clamp(SsviParams{});
    CHECK(ssvi_calendar_arbitrage_free(p));

    std::size_t violations = 0;
    double      worst_drop = 0.0;
    for (std::size_t i = 0; i < kNk; ++i) {
        const double k    = grid_k(i);
        double       prev = ssvi_total_variance(k, grid_t(0), p);
        for (std::size_t j = 1; j < kNt; ++j) {
            const double w = ssvi_total_variance(k, grid_t(j), p);
            if (!(w >= prev)) {
                ++violations;
                if (prev - w > worst_drop) worst_drop = prev - w;
            }
            prev = w;
        }
    }
    CHECK(violations == 0u);
    CHECK(worst_drop == 0.0);
    // 単調性の検査が自明に通っていないこと（面が T 方向に実際に伸びている）。
    CHECK(ssvi_total_variance(0.0, grid_t(kNt - 1), p) > ssvi_total_variance(0.0, grid_t(0), p));
    CHECK(ssvi_total_variance(-0.8, grid_t(kNt - 1), p) > ssvi_total_variance(-0.8, grid_t(0), p));

    // 負の例: η が大きすぎると翼のバンド η(1+|ρ|) ≤ 2（Thm 4.2 / Remark 4.4 由来の保守的な制約）を
    // 破るので false。カレンダー条件 (A) 自体は η = 5 でも満たされている（σ>0, η>0, γ∈[0,1], |ρ|<1）
    // ことに注意 — この false は「カレンダー裁定がある」という意味ではなく、翼のバンド落ちである。
    SsviParams too_steep = p;
    too_steep.eta        = 5.0;  // 5.0 * (1 + 0.3) = 6.5 > 2
    CHECK_FALSE(ssvi_calendar_arbitrage_free(ssvi_clamp(too_steep)));

    // 境界: η(1+|ρ|) = 2 はちょうど許容、わずかに超えると false。
    SsviParams edge = p;
    edge.rho        = 0.0;
    edge.eta        = 2.0;
    CHECK(ssvi_calendar_arbitrage_free(edge));
    edge.eta = 2.0 + 1e-9;
    CHECK_FALSE(ssvi_calendar_arbitrage_free(edge));

    // 未クランプの不正パラメータはカレンダー条件 (A) 側で false（γ > 1、σ ≤ 0、|ρ| ≥ 1、NaN）。
    SsviParams bad_gamma = p;
    bad_gamma.gamma      = 1.5;
    CHECK_FALSE(ssvi_calendar_arbitrage_free(bad_gamma));
    SsviParams bad_sigma = p;
    bad_sigma.sigma_atm  = 0.0;
    CHECK_FALSE(ssvi_calendar_arbitrage_free(bad_sigma));
    SsviParams bad_rho = p;
    bad_rho.rho        = -1.0;
    CHECK_FALSE(ssvi_calendar_arbitrage_free(bad_rho));
    SsviParams nan_eta = p;
    nan_eta.eta        = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(ssvi_calendar_arbitrage_free(nan_eta));
}

TEST_CASE("VOLSURF-03: with zero skew the smile is symmetric about the money", "[volsurf][numeric]") {
    SsviParams p = SsviParams{};
    p.rho        = 0.0;
    p            = ssvi_clamp(p);
    REQUIRE(p.rho == 0.0);

    // 許容は計画書が指定する相対 1e-12。実際には ρ = 0 だと w は x = φk の偶関数として計算される
    // （1 + 0·x + sqrt(x² + 0 + 1)、x の符号は x*x で消える）ので誤差はビット単位で 0 になる。
    // 1e-12 は φ の形が将来変わっても意味を保つための余裕であって、緩めた結果ではない。
    for (std::size_t j = 0; j < kNt; ++j) {
        const double t = grid_t(j);
        for (std::size_t i = 0; i < kNk / 2; ++i) {
            const double k = grid_k(i);  // k < 0 側
            INFO("k = " << k << ", T = " << t);
            CHECK_THAT(ssvi_total_variance(k, t, p), WithinRel(ssvi_total_variance(-k, t, p), 1e-12));
            CHECK_THAT(ssvi_implied_vol(k, t, p), WithinRel(ssvi_implied_vol(-k, t, p), 1e-12));
        }
    }

    // ρ ≠ 0 なら対称ではない（テストが自明に通っていないことの確認）。
    SsviParams skewed = ssvi_clamp(SsviParams{});
    REQUIRE(skewed.rho < 0.0);
    CHECK(ssvi_total_variance(-0.5, 1.0, skewed) > ssvi_total_variance(0.5, 1.0, skewed));
}
