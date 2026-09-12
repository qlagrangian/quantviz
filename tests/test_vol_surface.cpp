// VOLSURF-xx — core/pricing/vol_surface.hpp（Gatheral–Jacquier SSVI）と、それを面に載せる
// scenes/vol_surface_model.hpp（VOLSURF-04）の仕様テスト
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/bridge/runner.hpp"
#include "quantviz/core/pricing/vol_surface.hpp"
#include "quantviz/scenes/vol_surface_model.hpp"

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

// ---------------------------------------------------------------------------------------------
// VOLSURF-04 — scenes/vol_surface_model.hpp（面を吐くシーン Model）の契約
// ---------------------------------------------------------------------------------------------

using quantviz::bridge::Command;
using quantviz::scenes::VolSurfaceModel;
using quantviz::scenes::VolSurfaceSnapshot;
using quantviz::scenes::VolSurfaceSurface;

namespace {

constexpr std::size_t kSurfK = VolSurfaceSurface::kK;
constexpr std::size_t kSurfT = VolSurfaceSurface::kT;
/// 「使い回しスロット」を模した毒値。surface() が書き落としたフィールドはこれが残る。
constexpr float kPoison = -12345.0f;

/// TripleBuffer の back() は 2 世代前の面が入った使い回しスロットなので、テストも
/// 「ゼロでも前回値でもない中身」を渡して surface() が全フィールドを書くことを検査する。
VolSurfaceSurface poisoned_surface() {
    VolSurfaceSurface s{};
    s.ks.fill(kPoison);
    s.ts.fill(kPoison);
    s.iv.fill(kPoison);
    return s;
}

/// 面のうち T が真ん中あたりの 1 行（スマイル 1 本）の先頭 index。
constexpr std::size_t mid_row() noexcept { return (kSurfT / 2) * kSurfK; }

}  // namespace

TEST_CASE("VOLSURF-04: the vol surface scene model satisfies Model/SurfaceModel, writes a fixed-size "
          "POD surface in full, and reshapes it when a parameter changes",
          "[volsurf][contract]") {
    // --- コンパイル時の契約 -------------------------------------------------------------
    STATIC_REQUIRE(quantviz::bridge::Model<VolSurfaceModel>);
    STATIC_REQUIRE(quantviz::bridge::SurfaceModel<VolSurfaceModel>);
    STATIC_REQUIRE(std::is_same_v<VolSurfaceModel::Snapshot, VolSurfaceSnapshot>);
    STATIC_REQUIRE(std::is_same_v<VolSurfaceModel::Surface, VolSurfaceSurface>);

    STATIC_REQUIRE(std::is_trivially_copyable_v<VolSurfaceSnapshot>);
    STATIC_REQUIRE(std::is_default_constructible_v<VolSurfaceSnapshot>);
    STATIC_REQUIRE(std::is_standard_layout_v<VolSurfaceSnapshot>);
    STATIC_REQUIRE(sizeof(VolSurfaceSnapshot) <= 128);  // Snapshot のサイズ方針（docs/01 §4）

    STATIC_REQUIRE(std::is_trivially_copyable_v<VolSurfaceSurface>);
    STATIC_REQUIRE(std::is_default_constructible_v<VolSurfaceSurface>);
    STATIC_REQUIRE(std::is_standard_layout_v<VolSurfaceSurface>);
    STATIC_REQUIRE(kSurfK == 64);
    STATIC_REQUIRE(kSurfT == 32);
    // 固定サイズ: 3 つの配列ぶんぴったり（可変長・ポインタが紛れ込んでいない）
    STATIC_REQUIRE(sizeof(VolSurfaceSurface) == sizeof(float) * (kSurfK + kSurfT + kSurfK * kSurfT));

    SECTION("surface() overwrites every field of a recycled slot with a finite, positive IV") {
        const VolSurfaceModel  m;
        const VolSurfaceSnapshot snap = m.snapshot();
        VolSurfaceSurface      s      = poisoned_surface();
        m.surface(s);

        CHECK(snap.seq == 0);  // まだステップしていない
        CHECK(snap.t == 0.0);

        // 軸は Config のレンジ（既定 k ∈ [−1, 1], T ∈ [0.05, 3]）を端まで覆い、狭義単調増加。
        CHECK(snap.k_min == -1.0);
        CHECK(snap.k_max == 1.0);
        CHECK(snap.t_min == 0.05);
        CHECK(snap.t_max == 3.0);
        CHECK_THAT(static_cast<double>(s.ks.front()), WithinRel(snap.k_min, 1e-6));
        CHECK_THAT(static_cast<double>(s.ks.back()), WithinRel(snap.k_max, 1e-6));
        CHECK_THAT(static_cast<double>(s.ts.front()), WithinRel(snap.t_min, 1e-6));
        CHECK_THAT(static_cast<double>(s.ts.back()), WithinRel(snap.t_max, 1e-6));

        bool axes_increasing = true;
        for (std::size_t i = 1; i < kSurfK; ++i) axes_increasing = axes_increasing && s.ks[i] > s.ks[i - 1];
        for (std::size_t j = 1; j < kSurfT; ++j) axes_increasing = axes_increasing && s.ts[j] > s.ts[j - 1];
        CHECK(axes_increasing);

        // 全要素が有限・正（VOLSURF-01 の格子版）かつ毒値が 1 つも残っていない。
        // 値そのものも core の閉形式と一致する（面はコアの純関数の値をそのまま並べたもの）。
        bool   all_ok   = true;
        bool   poisoned = false;
        double max_rel  = 0.0;
        for (std::size_t it = 0; it < kSurfT; ++it) {
            for (std::size_t ik = 0; ik < kSurfK; ++ik) {
                const float v = s.iv[it * kSurfK + ik];  // row-major [iT][iK]
                all_ok        = all_ok && std::isfinite(v) && v > 0.0f;
                poisoned      = poisoned || v == kPoison;
                const double want = ssvi_implied_vol(static_cast<double>(s.ks[ik]),
                                                     static_cast<double>(s.ts[it]), snap.params);
                max_rel = std::max(max_rel, std::fabs(static_cast<double>(v) - want) / want);
            }
        }
        CHECK(all_ok);
        CHECK_FALSE(poisoned);
        CHECK(max_rel < 1e-6);  // float に落とす丸めのぶんだけ（倍精度の値そのもの）
    }

    SECTION("step advances t and seq only; the surface does not depend on t") {
        VolSurfaceModel   m;
        VolSurfaceSurface before{};
        m.surface(before);

        constexpr double kDt = 1.0 / 252.0;
        for (int i = 0; i < 5; ++i) m.step(kDt);

        const VolSurfaceSnapshot snap = m.snapshot();
        CHECK(snap.seq == 5);
        CHECK_THAT(snap.t, WithinRel(5.0 * kDt, 1e-12));

        VolSurfaceSurface after = poisoned_surface();
        m.surface(after);
        CHECK(std::equal(before.iv.begin(), before.iv.end(), after.iv.begin()));
        CHECK(std::equal(before.ks.begin(), before.ks.end(), after.ks.begin()));
        CHECK(std::equal(before.ts.begin(), before.ts.end(), after.ts.begin()));
    }

    SECTION("SetParam(rho) reshapes the surface at the next surface() call and leaves seq alone") {
        VolSurfaceModel m;
        m.step(1.0 / 252.0);

        VolSurfaceSurface s0{};
        m.surface(s0);
        const VolSurfaceSnapshot before = m.snapshot();
        REQUIRE(before.seq == 1);
        REQUIRE(before.params.rho < 0.0);  // 既定 ρ = −0.3: put 側（k < 0）の翼が高い
        CHECK(s0.iv[mid_row()] > s0.iv[mid_row() + kSurfK - 1]);

        m.apply(Command::set_param(VolSurfaceModel::kRho, 0.8));
        const VolSurfaceSnapshot after = m.snapshot();
        CHECK(after.seq == before.seq);  // apply は通番も時刻も動かさない
        CHECK(after.t == before.t);
        CHECK_THAT(after.params.rho, WithinRel(0.8, 1e-15));

        VolSurfaceSurface s1 = poisoned_surface();
        m.surface(s1);
        CHECK_FALSE(std::equal(s0.iv.begin(), s0.iv.end(), s1.iv.begin()));
        CHECK(s1.iv[mid_row()] < s1.iv[mid_row() + kSurfK - 1]);  // ρ > 0 でスキューが反転
        CHECK(std::equal(s0.ks.begin(), s0.ks.end(), s1.ks.begin()));  // 軸は動かない
    }

    SECTION("apply clamps through ssvi_clamp and ignores unknown ids and clock commands") {
        VolSurfaceModel m;
        m.apply(Command::set_param(VolSurfaceModel::kRho, 5.0));
        CHECK_THAT(m.snapshot().params.rho, WithinRel(quantviz::core::kSsviMaxAbsRho, 1e-15));
        m.apply(Command::set_param(VolSurfaceModel::kGamma, -3.0));
        CHECK(m.snapshot().params.gamma == quantviz::core::kSsviMinGamma);
        m.apply(Command::set_param(VolSurfaceModel::kEta, 1e9));
        CHECK(m.snapshot().params.eta == quantviz::core::kSsviMaxEta);
        // NaN はそのフィールドの既定値へ（ssvi_clamp の規約）
        m.apply(Command::set_param(VolSurfaceModel::kSigmaAtm,
                                   std::numeric_limits<double>::quiet_NaN()));
        CHECK(m.snapshot().params.sigma_atm == SsviParams{}.sigma_atm);

        const VolSurfaceSnapshot keep = m.snapshot();
        m.apply(Command::set_param(0, 3.0));    // 0 は予約
        m.apply(Command::set_param(99, 3.0));   // 未知の param_id
        m.apply(Command::pause());              // 時計系は Runner が処理済み
        m.apply(Command::set_speed(4.0));
        const VolSurfaceSnapshot same = m.snapshot();
        CHECK(same.params.sigma_atm == keep.params.sigma_atm);
        CHECK(same.params.rho == keep.params.rho);
        CHECK(same.params.eta == keep.params.eta);
        CHECK(same.params.gamma == keep.params.gamma);
        CHECK(same.seq == keep.seq);

        // クランプ後のパラメータなら面はやはり有限・正のまま
        VolSurfaceSurface s = poisoned_surface();
        m.surface(s);
        bool all_ok = true;
        for (float v : s.iv) all_ok = all_ok && std::isfinite(v) && v > 0.0f;
        CHECK(all_ok);
    }

    SECTION("Reset rewinds t and seq, keeps the parameters, and rebuilds the surface") {
        VolSurfaceModel m;
        m.apply(Command::set_param(VolSurfaceModel::kEta, 1.4));
        m.apply(Command::set_param(VolSurfaceModel::kGamma, 0.25));
        for (int i = 0; i < 7; ++i) m.step(1.0 / 252.0);

        VolSurfaceSurface before{};
        m.surface(before);
        const VolSurfaceSnapshot pre = m.snapshot();
        REQUIRE(pre.seq == 7);
        REQUIRE(pre.t > 0.0);

        m.apply(Command::reset());
        const VolSurfaceSnapshot post = m.snapshot();
        CHECK(post.seq == 0);
        CHECK(post.t == 0.0);
        CHECK(post.params.sigma_atm == pre.params.sigma_atm);
        CHECK(post.params.rho == pre.params.rho);
        CHECK_THAT(post.params.eta, WithinRel(1.4, 1e-15));
        CHECK_THAT(post.params.gamma, WithinRel(0.25, 1e-15));

        // Reset でも面は作り直され、パラメータが同じなので同じ面になる
        VolSurfaceSurface after = poisoned_surface();
        m.surface(after);
        CHECK(std::equal(before.iv.begin(), before.iv.end(), after.iv.begin()));
    }

    SECTION("Runner::tick + poll_surface hands the newest surface to the drawing side") {
        using Runner = quantviz::bridge::Runner<VolSurfaceModel, 256, 256>;

        quantviz::bridge::RunnerConfig cfg;
        cfg.dt                     = 1.0 / 252.0;
        cfg.clock.steps_per_second = 100.0;
        cfg.publish_every          = 1;
        cfg.surface_every          = 1;
        Runner r(VolSurfaceModel{}, cfg);

        VolSurfaceSurface got = poisoned_surface();
        CHECK_FALSE(r.poll_surface(got));  // まだ何も publish されていない
        CHECK(r.surfaces_published() == 0);

        CHECK(r.tick(0.1) == 10);
        CHECK(r.surfaces_published() == 10);
        REQUIRE(r.poll_surface(got));
        CHECK_FALSE(r.poll_surface(got));  // 最新 1 枚だけ（古い 9 枚は捨てられる）

        VolSurfaceSurface want{};
        r.model().surface(want);  // tick は同期実行なので model() を読んでも競合しない
        CHECK(std::equal(got.ks.begin(), got.ks.end(), want.ks.begin()));
        CHECK(std::equal(got.ts.begin(), got.ts.end(), want.ts.begin()));
        CHECK(std::equal(got.iv.begin(), got.iv.end(), want.iv.begin()));

        // ρ を送ると、次の tick で publish される面が入れ替わる（Snapshot 側にも新しい ρ が載る）
        REQUIRE(r.send(Command::set_param(VolSurfaceModel::kRho, 0.8)));
        CHECK(r.tick(0.01) == 1);
        VolSurfaceSurface reskewed = poisoned_surface();
        REQUIRE(r.poll_surface(reskewed));
        CHECK_FALSE(std::equal(got.iv.begin(), got.iv.end(), reskewed.iv.begin()));
        CHECK(std::equal(got.ks.begin(), got.ks.end(), reskewed.ks.begin()));  // 軸は動かない

        VolSurfaceSnapshot snap{};
        bool               received = false;
        while (r.poll(snap)) received = true;
        REQUIRE(received);
        CHECK_THAT(snap.params.rho, WithinRel(0.8, 1e-15));
        CHECK(snap.seq == 11);
    }
}
