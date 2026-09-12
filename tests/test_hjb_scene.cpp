// HJBSCENE-01..02 — scenes/hjb_model.hpp の契約 / 単体テスト（docs/03_tdd_spec.md §7.3）
//
// このシーンは「1 step = HJB の後退 1 反復」なので、テストも反復数で書く（FDM シーンと同じ流儀）。
// 数値そのもの（π* が定数であること、閉形式への収束、時間整合）は HJB-01..06
// （tests/test_hjb_merton.cpp）が担う。ここで見るのは「配線」:
//   - Model / SurfaceModel の契約と POD の大きさ
//   - 素の `core::HjbMerton` と同じ格子・同じ順序で回し、Snapshot / Surface へ正しく写しているか
//   - 1 step = 1 反復、SetParam(γ) で満期へ巻き戻り π* が新しい解析値に動くこと
//
// 面（Surface）は 160 KB の POD なのでスタックに積まず make_unique で持つ。
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/bridge/runner.hpp"
#include "quantviz/core/exec/hjb_merton.hpp"
#include "quantviz/scenes/hjb_model.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::bridge::Command;
using quantviz::core::crra_utility;
using quantviz::core::HjbMerton;
using quantviz::core::HjbParams;
using quantviz::core::merton_fraction;
using quantviz::core::merton_value;
using quantviz::scenes::HjbModel;
using quantviz::scenes::HjbSnapshot;
using quantviz::scenes::HjbSurface;

namespace {

/// テスト用の格子。n_w は Snapshot の配列長で固定（256）なので、動かせるのは n_t だけ。
/// n_t = 面の行数（200）に合わせると「1 反復 = 面 1 行」が成り立ち、埋まり方の検査が読みやすい。
HjbModel::Config test_config() {
    HjbModel::Config cfg;
    cfg.mu    = 0.08;
    cfg.r     = 0.03;
    cfg.sigma = 0.20;
    cfg.gamma = 3.0;
    cfg.T     = 1.0;
    cfg.w_min = 0.2;
    cfg.w_max = 5.0;
    cfg.n_t   = 200;
    return cfg;
}

void advance(HjbModel& m, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) m.step(1.0 / 252.0);
}

/// 素の HjbMerton をモデルと同じパラメータで t = 0 まで回す（dual-run オラクル）。
/// パラメータはモデル自身の（sanitize 済みの）ものを使うので、入力の写し間違いは起こり得ない。
std::unique_ptr<HjbMerton> bare_run(const HjbParams& p, std::size_t steps) {
    auto h = std::make_unique<HjbMerton>();
    h->init(p);
    for (std::size_t i = 0; i < steps; ++i) h->step_backward();
    return h;
}

/// 面の行 row に載る時間レベル（モデルのヘッダ「時間レベルと面の行の対応」と同じ式）。
std::size_t level_of_row(std::size_t row, std::size_t n_t) {
    constexpr std::size_t d = HjbSurface::kT - 1;
    return (row * n_t + d / 2) / d;
}

/// 内側 90 %（両端 5 % を除く）での max |π_numeric − π_analytic|。
double max_pi_error_inner(const HjbSnapshot& s) {
    constexpr std::size_t lo = HjbSnapshot::kNodes * 5 / 100;
    double                worst = 0.0;
    for (std::size_t i = lo; i + lo < HjbSnapshot::kNodes; ++i)
        worst = std::max(worst, std::abs(s.pi_star[i] - s.analytic_pi[i]));
    return worst;
}

}  // namespace

TEST_CASE("HJBSCENE-01: satisfies the Model and SurfaceModel contracts and mirrors a bare HjbMerton run "
          "into the snapshot and the surface",
          "[hjbscene][contract]") {
    STATIC_REQUIRE(quantviz::bridge::Model<HjbModel>);
    STATIC_REQUIRE(quantviz::bridge::SurfaceModel<HjbModel>);

    STATIC_REQUIRE(std::is_trivially_copyable_v<HjbSnapshot>);
    STATIC_REQUIRE(std::is_standard_layout_v<HjbSnapshot>);
    STATIC_REQUIRE(std::is_default_constructible_v<HjbSnapshot>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<HjbSurface>);
    STATIC_REQUIRE(std::is_standard_layout_v<HjbSurface>);
    STATIC_REQUIRE(std::is_default_constructible_v<HjbSurface>);

    // Snapshot は 256 点 × 5 本の double ≈ 10 KB（M1 Greeks と同じ「32 KiB まで」の例外枠の内側。
    // 上限は計画書の 16 KiB で縛る）。面はリングに載せない（TripleBuffer で最新 1 枚、200×200 float）。
    STATIC_REQUIRE(HjbSnapshot::kNodes == 256);
    STATIC_REQUIRE(sizeof(HjbSnapshot) <= 16 * 1024);
    STATIC_REQUIRE(HjbSurface::kW == 200);
    STATIC_REQUIRE(HjbSurface::kT == 200);
    STATIC_REQUIRE(sizeof(HjbSurface) >= HjbSurface::kW * HjbSurface::kT * sizeof(float));
    STATIC_REQUIRE(sizeof(HjbSurface) <= 163840 + 4096);  // 160 KB + 軸と filled_rows のぶん
    // 面の列は格子の節点の間引き。両端は必ず格子の両端に当たる（= 面の w 軸は [w_min, w_max] を張る）。
    STATIC_REQUIRE(HjbModel::node_of_column(0) == 0);
    STATIC_REQUIRE(HjbModel::node_of_column(HjbSurface::kW - 1) == HjbSnapshot::kNodes - 1);

    const auto cfg = test_config();

    SECTION("a fresh model sits on the terminal utility: seq 0, the whole sweep still ahead") {
        const HjbModel    m(cfg);
        const HjbSnapshot s = m.snapshot();
        CHECK(s.seq == 0);
        CHECK(s.iteration == 0);
        CHECK(s.remaining == cfg.n_t);
        CHECK_THAT(s.t_remaining, WithinRel(cfg.T, 1e-15));
        CHECK_THAT(s.mu, WithinRel(cfg.mu, 1e-15));
        CHECK_THAT(s.r, WithinRel(cfg.r, 1e-15));
        CHECK_THAT(s.sigma, WithinRel(cfg.sigma, 1e-15));
        CHECK_THAT(s.gamma, WithinRel(cfg.gamma, 1e-15));
        CHECK_THAT(s.T, WithinRel(cfg.T, 1e-15));

        // 富の格子は対数等間隔・昇順、両端は w_min / w_max ちょうど。
        CHECK_THAT(s.wealth.front(), WithinRel(cfg.w_min, 1e-15));
        CHECK_THAT(s.wealth.back(), WithinRel(cfg.w_max, 1e-15));
        for (std::size_t i = 1; i < HjbSnapshot::kNodes; ++i) CHECK(s.wealth[i] > s.wealth[i - 1]);

        // 終端条件 V(w, T) = U(w)、閉形式の参照線は τ = 0 なので U(w) と一致する。
        for (std::size_t i = 0; i < HjbSnapshot::kNodes; ++i) {
            CHECK_THAT(s.values[i], WithinRel(crra_utility(s.wealth[i], cfg.gamma), 1e-15));
            CHECK_THAT(s.analytic_v[i], WithinRel(crra_utility(s.wealth[i], cfg.gamma), 1e-12));
        }
        // analytic_pi は定数（= merton_fraction）を配列に詰めたもの。
        const double pi_exact = merton_fraction(m.solver().params());
        for (std::size_t i = 0; i < HjbSnapshot::kNodes; ++i) CHECK(s.analytic_pi[i] == pi_exact);
        CHECK_THAT(pi_exact, WithinRel(0.05 / 0.12, 1e-14));
    }

    SECTION("after init the surface holds the terminal utility in row 0 and zeros everywhere else") {
        const HjbModel m(cfg);
        auto           surf = std::make_unique<HjbSurface>();
        m.surface(*surf);

        CHECK(surf->filled_rows == 1);
        CHECK_THAT(static_cast<double>(surf->times[0]), WithinRel(cfg.T, 1e-6));
        CHECK_THAT(static_cast<double>(surf->times[HjbSurface::kT - 1]), WithinAbs(0.0, 1e-7));
        for (std::size_t i = 1; i < HjbSurface::kT; ++i) CHECK(surf->times[i] < surf->times[i - 1]);

        // 面の列は格子の節点そのもの（補間しない）。行 0 は U(w)。
        const std::span<const double> w = m.solver().wealth();
        for (std::size_t j = 0; j < HjbSurface::kW; ++j) {
            const double wj = w[HjbModel::node_of_column(j)];
            CHECK_THAT(static_cast<double>(surf->wealth[j]), WithinRel(wj, 1e-6));
            CHECK_THAT(static_cast<double>(surf->values[j]), WithinRel(crra_utility(wj, cfg.gamma), 1e-6));
        }
        for (std::size_t j = 1; j < HjbSurface::kW; ++j) CHECK(surf->wealth[j] > surf->wealth[j - 1]);

        for (std::size_t row = 1; row < HjbSurface::kT; ++row)
            for (std::size_t j = 0; j < HjbSurface::kW; ++j)
                CHECK(surf->values[row * HjbSurface::kW + j] == 0.f);
    }

    SECTION("surface() overwrites every field of the recycled TripleBuffer slot") {
        // TripleBuffer のスロットは「2 回前に publish した面」であって、ゼロ初期化も前回内容の保持も
        // 期待できない（model_concept.hpp）。0xFF で埋めると float は NaN、filled_rows は 0xFFFFFFFF に
        // なるので、書き忘れたフィールドがあれば必ず見える。
        HjbModel m(cfg);
        advance(m, 30);
        auto surf = std::make_unique<HjbSurface>();
        std::memset(static_cast<void*>(surf.get()), 0xFF, sizeof(HjbSurface));
        REQUIRE(std::isnan(static_cast<double>(surf->values[0])));
        REQUIRE(surf->filled_rows == 0xFFFFFFFFu);

        m.surface(*surf);

        CHECK(surf->filled_rows == 31);  // n_t = kT なので 30 反復 + 終端行
        for (std::size_t j = 0; j < HjbSurface::kW; ++j) CHECK(std::isfinite(surf->wealth[j]));
        for (std::size_t i = 0; i < HjbSurface::kT; ++i) CHECK(std::isfinite(surf->times[i]));
        for (std::size_t row = 0; row < HjbSurface::kT; ++row) {
            const bool computed = row < surf->filled_rows;
            for (std::size_t j = 0; j < HjbSurface::kW; ++j) {
                const float x = surf->values[row * HjbSurface::kW + j];
                if (computed) {
                    CHECK(std::isfinite(x));
                } else {
                    CHECK(x == 0.f);  // 未計算行はゼロで塗り直される（NaN が残らない）
                }
            }
        }
    }

    SECTION("a broken maturity is sanitised once: the time axis and snapshot().T never disagree") {
        // T は UI に出ないが、Config から NaN / 0 / 負が来ても面と Snapshot が食い違ってはいけない
        // （生の Config を面が使うと、snapshot().T = 1e-6 なのに times[] だけ NaN、という状態になる）。
        const double kBadT[3] = {std::numeric_limits<double>::quiet_NaN(), 0.0, -5.0};
        for (const double bad : kBadT) {
            HjbModel::Config broken = cfg;
            broken.T                = bad;
            HjbModel          m(broken);
            const HjbSnapshot s = m.snapshot();
            INFO("cfg.T = " << bad << " -> snapshot().T = " << s.T);
            REQUIRE(std::isfinite(s.T));
            REQUIRE(s.T > 0.0);
            CHECK_THAT(s.t_remaining, WithinRel(s.T, 1e-15));

            auto surf = std::make_unique<HjbSurface>();
            m.surface(*surf);
            for (std::size_t i = 0; i < HjbSurface::kT; ++i) REQUIRE(std::isfinite(surf->times[i]));
            CHECK_THAT(static_cast<double>(surf->times[0]), WithinRel(s.T, 1e-6));
            CHECK_THAT(static_cast<double>(surf->times[HjbSurface::kT - 1]), WithinAbs(0.0, 1e-7 * s.T));
            for (std::size_t j = 0; j < HjbSurface::kW; ++j) REQUIRE(std::isfinite(surf->values[j]));
        }
    }

    SECTION("after the full sweep the snapshot is bit-identical to a bare HjbMerton run") {
        HjbModel m(cfg);
        advance(m, cfg.n_t);
        const HjbSnapshot s = m.snapshot();
        REQUIRE(s.remaining == 0);

        const auto bare = bare_run(m.solver().params(), cfg.n_t);
        REQUIRE(bare->remaining() == 0);

        // Snapshot の 3 本は格子の値をそのまま写したもの（補間なし）→ bit 一致。
        for (std::size_t i = 0; i < HjbSnapshot::kNodes; ++i) {
            CHECK(s.wealth[i] == bare->wealth()[i]);
            CHECK(s.values[i] == bare->values()[i]);
            CHECK(s.pi_star[i] == bare->optimal_fraction()[i]);
        }
        CHECK_THAT(s.t_remaining, WithinAbs(0.0, 1e-15));
        CHECK_THAT(max_pi_error_inner(s), WithinAbs(0.0, 1e-3));
        CHECK_THAT(s.max_abs_pi_error, WithinAbs(max_pi_error_inner(s), 1e-15));
        INFO("max |pi - analytic| (inner 90%) = " << s.max_abs_pi_error);
        CHECK(s.max_abs_pi_error < 1e-3);
    }

    SECTION("after the full sweep the last surface row is the t = 0 solution sampled the same way") {
        HjbModel m(cfg);
        advance(m, cfg.n_t);
        auto surf = std::make_unique<HjbSurface>();
        m.surface(*surf);
        CHECK(surf->filled_rows == HjbSurface::kT);

        const auto bare = bare_run(m.solver().params(), cfg.n_t);
        const std::size_t last = HjbSurface::kT - 1;
        for (std::size_t j = 0; j < HjbSurface::kW; ++j) {
            const std::size_t node = HjbModel::node_of_column(j);
            CHECK_THAT(static_cast<double>(surf->values[last * HjbSurface::kW + j]),
                       WithinRel(bare->values()[node], 1e-6));  // float の丸めぶん
        }

        // 中間行も 1 本ピン留めする（端だけ合わせる実装を通さないため）。
        constexpr std::size_t kRow  = 77;
        const std::size_t     level = level_of_row(kRow, cfg.n_t);
        REQUIRE(level > 0);
        REQUIRE(level < cfg.n_t);
        const auto mid = bare_run(m.solver().params(), level);
        CHECK_THAT(static_cast<double>(surf->times[kRow]), WithinAbs(mid->time(), 1e-6));
        for (std::size_t j = 0; j < HjbSurface::kW; ++j)
            CHECK_THAT(static_cast<double>(surf->values[kRow * HjbSurface::kW + j]),
                       WithinRel(mid->values()[HjbModel::node_of_column(j)], 1e-6));
    }

    SECTION("an unknown param_id is ignored") {
        HjbModel m(cfg);
        advance(m, 3);
        const HjbSnapshot before = m.snapshot();
        m.apply(Command::set_param(0, 1.0));     // 0 は予約
        m.apply(Command::set_param(999, 42.0));  // 未知
        const HjbSnapshot after = m.snapshot();
        CHECK(after.seq == before.seq);
        CHECK(after.iteration == before.iteration);
        CHECK(after.gamma == before.gamma);
        CHECK(after.values == before.values);
    }

    SECTION("SetParam clamps finite values to the documented range") {
        HjbModel m(cfg);
        m.apply(Command::set_param(HjbModel::kMu, 10.0));
        CHECK_THAT(m.snapshot().mu, WithinRel(HjbModel::kMaxMu, 1e-15));
        m.apply(Command::set_param(HjbModel::kMu, -10.0));
        CHECK_THAT(m.snapshot().mu, WithinRel(HjbModel::kMinMu, 1e-15));
        m.apply(Command::set_param(HjbModel::kRate, -1.0));
        CHECK_THAT(m.snapshot().r, WithinAbs(HjbModel::kMinRate, 1e-15));
        m.apply(Command::set_param(HjbModel::kSigma, 0.0));
        CHECK_THAT(m.snapshot().sigma, WithinRel(HjbModel::kMinSigma, 1e-15));
        m.apply(Command::set_param(HjbModel::kGamma, 1e9));
        CHECK_THAT(m.snapshot().gamma, WithinRel(HjbModel::kMaxGamma, 1e-15));
    }

    SECTION("gamma is snapped to exactly 1 near the log-utility branch") {
        HjbModel m(cfg);
        m.apply(Command::set_param(HjbModel::kGamma, 1.0 + 1e-7));
        CHECK(m.snapshot().gamma == 1.0);  // 厳密に 1（log 分岐に乗せる）
        // 終端は U(w) = ln w。
        const HjbSnapshot s = m.snapshot();
        for (std::size_t i = 0; i < HjbSnapshot::kNodes; ++i)
            CHECK_THAT(s.values[i], WithinAbs(std::log(s.wealth[i]), 1e-14));

        m.apply(Command::set_param(HjbModel::kGamma, 1.01));  // 吸着域の外
        CHECK_THAT(m.snapshot().gamma, WithinRel(1.01, 1e-15));
    }

    SECTION("SetParam rejects NaN and infinity (the sweep is not even rewound)") {
        HjbModel m(cfg);
        advance(m, 4);
        const HjbSnapshot       before = m.snapshot();
        constexpr std::uint32_t kIds[] = {HjbModel::kMu, HjbModel::kRate, HjbModel::kSigma,
                                          HjbModel::kGamma};
        for (const std::uint32_t id : kIds) {
            m.apply(Command::set_param(id, std::numeric_limits<double>::quiet_NaN()));
            m.apply(Command::set_param(id, std::numeric_limits<double>::infinity()));
            m.apply(Command::set_param(id, -std::numeric_limits<double>::infinity()));
        }
        const HjbSnapshot after = m.snapshot();
        CHECK(after.mu == before.mu);
        CHECK(after.r == before.r);
        CHECK(after.sigma == before.sigma);
        CHECK(after.gamma == before.gamma);
        CHECK(after.iteration == before.iteration);
        CHECK(after.seq == before.seq);
    }

    SECTION("Reset rewinds the sweep to the terminal utility") {
        HjbModel m(cfg);
        advance(m, 9);
        REQUIRE(m.snapshot().iteration == 9);
        m.apply(Command::reset());
        const HjbSnapshot s = m.snapshot();
        CHECK(s.seq == 0);
        CHECK(s.iteration == 0);
        CHECK(s.remaining == cfg.n_t);
        CHECK_THAT(s.t_remaining, WithinRel(cfg.T, 1e-15));
        for (std::size_t i = 0; i < HjbSnapshot::kNodes; ++i)
            CHECK_THAT(s.values[i], WithinRel(crra_utility(s.wealth[i], cfg.gamma), 1e-15));
    }

    SECTION("Runner: tick publishes snapshots on the ring and the newest surface on the triple buffer") {
        namespace bridge = quantviz::bridge;
        bridge::RunnerConfig rc;
        rc.dt                     = 1.0 / 252.0;
        rc.clock.steps_per_second = 100.0;
        rc.publish_every          = 1;
        rc.surface_every          = 1;

        // Runner は Snapshot リング（8 × 10 KB）と面 3 枚（3 × 158 KiB）を抱えるのでヒープに置く。
        auto r    = std::make_unique<bridge::Runner<HjbModel, 8, 8>>(HjbModel{cfg}, rc);
        auto surf = std::make_unique<HjbSurface>();
        CHECK_FALSE(r->poll_surface(*surf));  // まだ 1 枚も出ていない

        CHECK(r->tick(0.05) == 5);  // 100 steps/s × 0.05 s

        HjbSnapshot s{};
        std::size_t n = 0;
        while (r->poll(s)) ++n;
        CHECK(n == 5);
        CHECK(s.iteration == 5);
        CHECK(s.remaining == cfg.n_t - 5);

        REQUIRE(r->surfaces_published() == 5);
        REQUIRE(r->poll_surface(*surf));  // 最新 1 枚
        CHECK(surf->filled_rows > 1);
        CHECK_FALSE(r->poll_surface(*surf));
    }
}

TEST_CASE("HJBSCENE-02: one step is one backward time iteration and SetParam(gamma) restarts the sweep "
          "with a new analytic pi*",
          "[hjbscene][unit]") {
    const auto cfg = test_config();

    SECTION("each step advances the iteration, consumes one remaining step and fills the surface") {
        HjbModel m(cfg);
        auto     surf = std::make_unique<HjbSurface>();
        m.surface(*surf);
        std::uint32_t prev_rows = surf->filled_rows;
        REQUIRE(prev_rows == 1);

        // 行 j に載るレベルは round(j · n_t / (kT − 1))。n_t = kT = 200 では 201 個のレベルを 200 行に
        // 載せるので、ちょうど 1 つのレベル（100）がどの行にも乗らない = その反復だけ行数が増えない。
        // それ以外は「1 反復 = 面 1 行」。掃引全体を回して、単調・最大 +1・停滞はちょうど 1 回を見る。
        const double dt_grid = cfg.T / static_cast<double>(cfg.n_t);
        std::size_t  stalls  = 0;
        for (std::size_t k = 1; k <= cfg.n_t; ++k) {
            m.step(1.0 / 252.0);
            const HjbSnapshot s = m.snapshot();
            REQUIRE(s.iteration == k);
            REQUIRE(s.remaining == cfg.n_t - k);
            REQUIRE(s.seq == k);
            CHECK_THAT(s.t_remaining, WithinAbs(cfg.T - static_cast<double>(k) * dt_grid, 1e-12));

            m.surface(*surf);
            REQUIRE(surf->filled_rows >= prev_rows);
            REQUIRE(surf->filled_rows <= prev_rows + 1);
            if (surf->filled_rows == prev_rows) ++stalls;
            prev_rows = surf->filled_rows;
        }
        CHECK(stalls == 1);
        CHECK(prev_rows == HjbSurface::kT);
    }

    SECTION("after the whole sweep further steps hold the solution but keep advancing seq") {
        HjbModel m(cfg);
        advance(m, cfg.n_t);
        const HjbSnapshot done = m.snapshot();
        CHECK(done.iteration == cfg.n_t);
        CHECK(done.remaining == 0);
        CHECK_THAT(done.t_remaining, WithinAbs(0.0, 1e-15));

        advance(m, 3);
        const HjbSnapshot after = m.snapshot();
        CHECK(after.iteration == cfg.n_t);
        CHECK(after.remaining == 0);
        CHECK(after.seq == done.seq + 3);  // Runner から見れば「ステップした」のは事実
        CHECK(after.values == done.values);
        CHECK(after.pi_star == done.pi_star);

        auto surf = std::make_unique<HjbSurface>();
        m.surface(*surf);
        CHECK(surf->filled_rows == HjbSurface::kT);
    }

    SECTION("SetParam(gamma) rewinds to the terminal condition and moves pi* to the new analytic value") {
        HjbModel m(cfg);
        advance(m, cfg.n_t);
        const HjbSnapshot before = m.snapshot();
        REQUIRE(before.remaining == 0);

        constexpr double kNewGamma = 5.0;  // pi* = 0.05 / (5 * 0.04) = 0.25（クランプ域の内側）
        m.apply(Command::set_param(HjbModel::kGamma, kNewGamma));

        const HjbSnapshot restarted = m.snapshot();
        CHECK(restarted.seq == 0);
        CHECK(restarted.iteration == 0);
        CHECK(restarted.remaining == cfg.n_t);
        CHECK_THAT(restarted.gamma, WithinRel(kNewGamma, 1e-15));
        CHECK_THAT(restarted.t_remaining, WithinRel(cfg.T, 1e-15));
        for (std::size_t i = 0; i < HjbSnapshot::kNodes; ++i)
            CHECK_THAT(restarted.values[i], WithinRel(crra_utility(restarted.wealth[i], kNewGamma), 1e-15));

        auto surf = std::make_unique<HjbSurface>();
        m.surface(*surf);
        CHECK(surf->filled_rows == 1);  // 面も満期の 1 行だけに戻る
        for (std::size_t j = 0; j < HjbSurface::kW; ++j)
            CHECK(surf->values[HjbSurface::kW + j] == 0.f);

        advance(m, cfg.n_t);
        const HjbSnapshot after = m.snapshot();
        const double      pi_new = merton_fraction(m.solver().params());
        CHECK_THAT(pi_new, WithinRel(0.25, 1e-14));
        CHECK(std::abs(pi_new - before.analytic_pi[0]) > 0.1);  // 確かに別の解になっている
        for (std::size_t i = 0; i < HjbSnapshot::kNodes; ++i) CHECK(after.analytic_pi[i] == pi_new);

        INFO("max |pi - analytic| (inner 90%) after gamma = 5 is " << after.max_abs_pi_error);
        CHECK(after.max_abs_pi_error < 1e-3);
        CHECK(std::abs(after.pi_star[HjbSnapshot::kNodes / 2] -
                       before.pi_star[HjbSnapshot::kNodes / 2]) > 0.1);
    }

    SECTION("a SetParam that does not move the value is a no-op") {
        HjbModel m(cfg);
        advance(m, 6);
        REQUIRE(m.snapshot().iteration == 6);

        m.apply(Command::set_param(HjbModel::kGamma, cfg.gamma));  // 同じ値
        CHECK(m.snapshot().iteration == 6);
        CHECK(m.snapshot().seq == 6);
        m.apply(Command::set_param(HjbModel::kSigma, 1e9));  // クランプで上限に着く
        REQUIRE(m.snapshot().iteration == 0);
        advance(m, 6);
        m.apply(Command::set_param(HjbModel::kSigma, 1e9));  // 同じ上限 → 動かない
        CHECK(m.snapshot().iteration == 6);
        CHECK(m.snapshot().seq == 6);
    }

    SECTION("the analytic reference curve follows the current time slice") {
        HjbModel m(cfg);
        advance(m, 50);
        const HjbSnapshot s = m.snapshot();
        const HjbParams   p = m.solver().params();
        for (std::size_t i = 0; i < HjbSnapshot::kNodes; i += 17)
            CHECK_THAT(s.analytic_v[i], WithinRel(merton_value(p, s.wealth[i], s.t_remaining), 1e-14));
        // 数値解も同じ断面に居る（HJB-05 の実測: 200×200 で相対 5e-6 程度。ここは配線の確認なので緩く）。
        for (std::size_t i = 0; i < HjbSnapshot::kNodes; i += 17)
            CHECK_THAT(s.values[i], WithinRel(s.analytic_v[i], 1e-3));
    }
}
