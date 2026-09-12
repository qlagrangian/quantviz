// FDMSCENE-01..04 — scenes/fdm_american_model.hpp の契約 / 単体 / 決定性テスト（docs/03_tdd_spec.md §5.7）
//
// このシーンは「1 step = CN/PSOR の後ろ向き反復 1 回」なので、テストも反復数で書く。
// 面（Surface）は TripleBuffer 経由で渡す大きな POD なので、スタックに積まず make_unique で持つ。
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/bridge/runner.hpp"
#include "quantviz/core/pricing/fdm_cn.hpp"
#include "quantviz/scenes/fdm_american_model.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::bridge::Command;
using quantviz::core::FdmCn;
using quantviz::core::FdmGrid;
using quantviz::core::FdmParams;
using quantviz::core::OptionType;
using quantviz::scenes::FdmAmericanModel;
using quantviz::scenes::FdmSceneSnapshot;
using quantviz::scenes::FdmSceneSurface;

namespace {

/// テスト用の格子は小さめ（100 × 50）。1 テストあたりの全掃引が数 ms で終わる。
FdmAmericanModel::Config test_config() {
    FdmAmericanModel::Config cfg;
    cfg.n_space  = 100;
    cfg.n_time   = 50;
    cfg.K        = 100.0;
    cfg.T        = 1.0;
    cfg.r        = 0.05;
    cfg.sigma    = 0.20;
    cfg.q        = 0.0;
    cfg.american = true;
    cfg.type     = OptionType::Put;
    cfg.omega    = 1.2;
    cfg.s0       = 100.0;
    return cfg;
}

/// Config から「素の FdmCn」の格子とパラメータを組む（モデルと同じ規則: S_max = kSmaxMultiple · K）。
FdmGrid grid_of(const FdmAmericanModel::Config& c) {
    return FdmGrid{FdmAmericanModel::kSmaxMultiple * c.K, c.n_space, c.n_time};
}
FdmParams params_of(const FdmAmericanModel::Config& c) {
    FdmParams p{};
    p.K           = c.K;
    p.T           = c.T;
    p.r           = c.r;
    p.sigma       = c.sigma;
    p.q           = c.q;
    p.type        = c.type;
    p.american    = c.american;
    p.psor_omega  = c.omega;
    return p;
}

double intrinsic(const FdmAmericanModel::Config& c, double s) {
    return c.type == OptionType::Call ? std::max(s - c.K, 0.0) : std::max(c.K - s, 0.0);
}

/// n 回 step する（dt はモデルが無視するので値は何でもよい）。
void advance(FdmAmericanModel& m, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) m.step(1.0 / 252.0);
}

}  // namespace

TEST_CASE("FDMSCENE-01: satisfies the Model and SurfaceModel contracts with fixed-size POD payloads",
          "[fdmscene][contract]") {
    STATIC_REQUIRE(quantviz::bridge::Model<FdmAmericanModel>);
    STATIC_REQUIRE(quantviz::bridge::SurfaceModel<FdmAmericanModel>);

    STATIC_REQUIRE(std::is_trivially_copyable_v<FdmSceneSnapshot>);
    STATIC_REQUIRE(std::is_standard_layout_v<FdmSceneSnapshot>);
    STATIC_REQUIRE(std::is_default_constructible_v<FdmSceneSnapshot>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<FdmSceneSurface>);
    STATIC_REQUIRE(std::is_standard_layout_v<FdmSceneSurface>);
    STATIC_REQUIRE(std::is_default_constructible_v<FdmSceneSurface>);

    // Snapshot は 3 本の 256 点配列で ≈ 6 KB（M1 の Greeks と同じ「32 KiB 上限」の例外枠）。
    STATIC_REQUIRE(sizeof(FdmSceneSnapshot) <= 32 * 1024);
    STATIC_REQUIRE(FdmSceneSnapshot::kCurve == 256);
    // 面はリングに載せない（TripleBuffer で最新 1 枚）。200 × 200 の float で ≈ 157 KiB。
    STATIC_REQUIRE(FdmSceneSurface::kS == 200);
    STATIC_REQUIRE(FdmSceneSurface::kT == 200);
    STATIC_REQUIRE(sizeof(FdmSceneSurface) >= FdmSceneSurface::kS * FdmSceneSurface::kT * sizeof(float));

    const auto cfg = test_config();

    SECTION("a fresh model sits at maturity: seq 0, iteration 0, the full sweep still ahead") {
        const FdmAmericanModel m(cfg);
        const FdmSceneSnapshot s = m.snapshot();
        CHECK(s.seq == 0);
        CHECK(s.iteration == 0);
        CHECK(s.remaining == cfg.n_time);
        CHECK(s.status == FdmSceneSnapshot::kStatusOk);
        CHECK(s.boundary_len == 0);
        CHECK_THAT(s.t_remaining, WithinRel(cfg.T, 1e-15));
        CHECK_THAT(s.K, WithinRel(cfg.K, 1e-15));
        CHECK_THAT(s.r, WithinRel(cfg.r, 1e-15));
        CHECK_THAT(s.sigma, WithinRel(cfg.sigma, 1e-15));
        CHECK(s.q == 0.0);
        CHECK(s.american);
        CHECK(s.type == OptionType::Put);
        CHECK_THAT(s.s0, WithinRel(cfg.s0, 1e-15));
        // S 軸は [0, S_max] を 256 点で等分する（昇順）。
        CHECK(s.spots.front() == 0.0);
        CHECK_THAT(s.spots.back(), WithinRel(FdmAmericanModel::kSmaxMultiple * cfg.K, 1e-15));
        for (std::size_t i = 1; i < FdmSceneSnapshot::kCurve; ++i) CHECK(s.spots[i] > s.spots[i - 1]);
    }

    SECTION("Reset rewinds seq and the iteration counter") {
        FdmAmericanModel m(cfg);
        advance(m, 7);
        REQUIRE(m.snapshot().seq == 7);
        REQUIRE(m.snapshot().iteration == 7);
        m.apply(Command::reset());
        const FdmSceneSnapshot s = m.snapshot();
        CHECK(s.seq == 0);
        CHECK(s.iteration == 0);
        CHECK(s.remaining == cfg.n_time);
    }

    SECTION("an unknown param_id is ignored") {
        FdmAmericanModel m(cfg);
        advance(m, 3);
        const FdmSceneSnapshot before = m.snapshot();
        m.apply(Command::set_param(0, 1.0));     // 0 は予約
        m.apply(Command::set_param(999, 42.0));  // 未知
        const FdmSceneSnapshot after = m.snapshot();
        CHECK(after.seq == before.seq);
        CHECK(after.iteration == before.iteration);
        CHECK(after.K == before.K);
        CHECK(after.v_at_s0 == before.v_at_s0);
    }

    SECTION("SetParam clamps finite values to the documented range") {
        FdmAmericanModel m(cfg);
        m.apply(Command::set_param(FdmAmericanModel::kStrike, 1e9));
        CHECK_THAT(m.snapshot().K, WithinRel(FdmAmericanModel::kMaxStrike, 1e-15));
        m.apply(Command::set_param(FdmAmericanModel::kSigma, -1.0));
        CHECK_THAT(m.snapshot().sigma, WithinRel(FdmAmericanModel::kMinSigma, 1e-15));
        m.apply(Command::set_param(FdmAmericanModel::kRate, -10.0));
        CHECK_THAT(m.snapshot().r, WithinRel(FdmAmericanModel::kMinRate, 1e-15));
        m.apply(Command::set_param(FdmAmericanModel::kDividend, 5.0));
        CHECK_THAT(m.snapshot().q, WithinRel(FdmAmericanModel::kMaxDividend, 1e-15));
        m.apply(Command::set_param(FdmAmericanModel::kOmega, 9.0));
        CHECK_THAT(m.solver().params().psor_omega, WithinRel(FdmAmericanModel::kMaxOmega, 1e-15));
        m.apply(Command::set_param(FdmAmericanModel::kOmega, 0.0));
        CHECK_THAT(m.solver().params().psor_omega, WithinRel(FdmAmericanModel::kMinOmega, 1e-15));
    }

    SECTION("SetParam rejects NaN and infinity (the sweep is not even rewound)") {
        FdmAmericanModel m(cfg);
        advance(m, 4);
        const FdmSceneSnapshot   before = m.snapshot();
        const double             omega  = m.solver().params().psor_omega;
        constexpr std::uint32_t  kIds[] = {FdmAmericanModel::kStrike,   FdmAmericanModel::kRate,
                                           FdmAmericanModel::kSigma,    FdmAmericanModel::kDividend,
                                           FdmAmericanModel::kAmerican, FdmAmericanModel::kOptionType,
                                           FdmAmericanModel::kOmega};
        for (const std::uint32_t id : kIds) {
            m.apply(Command::set_param(id, std::numeric_limits<double>::quiet_NaN()));
            m.apply(Command::set_param(id, std::numeric_limits<double>::infinity()));
            m.apply(Command::set_param(id, -std::numeric_limits<double>::infinity()));
        }
        const FdmSceneSnapshot after = m.snapshot();
        CHECK(after.K == before.K);
        CHECK(after.r == before.r);
        CHECK(after.sigma == before.sigma);
        CHECK(after.q == before.q);
        CHECK(after.american == before.american);
        CHECK(after.type == before.type);
        CHECK(m.solver().params().psor_omega == omega);
        CHECK(after.iteration == before.iteration);  // 拒否した = 掃引をやり直していない
        CHECK(after.seq == before.seq);
    }

    SECTION("kAmerican = 0 hands the step to the European solver: no boundary, no PSOR sweeps") {
        FdmAmericanModel m(cfg);
        advance(m, 10);
        CHECK(std::isfinite(m.snapshot().exercise_boundary_now));
        CHECK(m.snapshot().psor_iterations > 0);

        m.apply(Command::set_param(FdmAmericanModel::kAmerican, 0.0));
        CHECK_FALSE(m.snapshot().american);
        CHECK(m.snapshot().iteration == 0);  // 係数が変わるので掃引はやり直し
        advance(m, 10);
        const FdmSceneSnapshot e = m.snapshot();
        CHECK(std::isnan(e.exercise_boundary_now));
        CHECK(e.psor_iterations == 0);  // 三重対角ソルバは反復しない
        CHECK(e.boundary_len == 10);    // 長さは伸びるが中身は NaN
        for (std::uint32_t i = 0; i < e.boundary_len; ++i) CHECK(std::isnan(e.boundary_t[i]));
    }

    SECTION("kOptionType swaps the payoff the sweep starts from") {
        FdmAmericanModel m(cfg);  // 既定は put
        const FdmSceneSnapshot put = m.snapshot();
        m.apply(Command::set_param(FdmAmericanModel::kOptionType, 0.0));  // 0 = Call
        const FdmSceneSnapshot call = m.snapshot();

        CHECK(call.type == OptionType::Call);
        CHECK(call.iteration == 0);
        CHECK(call.values.front() == 0.0);  // call は S = 0 で無価値
        CHECK(put.values.front() > 0.0);    // put は S = 0 で K
        for (std::size_t i = 0; i < FdmSceneSnapshot::kCurve; ++i)
            CHECK_THAT(call.values[i], WithinAbs(std::max(call.spots[i] - cfg.K, 0.0), 1e-9));

        m.apply(Command::set_param(FdmAmericanModel::kOptionType, 1.0));  // 1 = Put
        CHECK(m.snapshot().type == OptionType::Put);
    }

    SECTION("Runner: tick publishes snapshots on the ring and the newest surface on the triple buffer") {
        namespace bridge = quantviz::bridge;
        bridge::RunnerConfig rc;
        rc.dt                       = 1.0 / 252.0;
        rc.clock.steps_per_second   = 100.0;
        rc.publish_every            = 1;
        rc.surface_every            = 1;

        // Runner は Snapshot リング（8 × 6 KB）と面 3 枚（3 × 157 KiB）を抱えるのでヒープに置く。
        auto r = std::make_unique<bridge::Runner<FdmAmericanModel, 8, 8>>(FdmAmericanModel{cfg}, rc);
        auto surf = std::make_unique<FdmSceneSurface>();
        CHECK_FALSE(r->poll_surface(*surf));  // まだ 1 枚も出ていない

        CHECK(r->tick(0.05) == 5);  // 100 steps/s × 0.05 s

        FdmSceneSnapshot s{};
        std::size_t      n = 0;
        while (r->poll(s)) ++n;
        CHECK(n == 5);
        CHECK(s.iteration == 5);
        CHECK(s.remaining == cfg.n_time - 5);

        REQUIRE(r->surfaces_published() == 5);
        REQUIRE(r->poll_surface(*surf));  // 最新 1 枚だけ
        CHECK(surf->filled_rows > 1);
        CHECK_FALSE(r->poll_surface(*surf));
    }
}

TEST_CASE("FDMSCENE-02: one step is exactly one backward iteration and the sweep stops at t = 0",
          "[fdmscene][unit]") {
    const auto       cfg = test_config();
    FdmAmericanModel m(cfg);

    const double dtau = cfg.T / static_cast<double>(cfg.n_time);

    SECTION("each step decrements remaining, increments iteration and walks t back by T / n_time") {
        for (std::size_t k = 1; k <= 5; ++k) {
            m.step(1.0 / 252.0);
            const FdmSceneSnapshot s = m.snapshot();
            CHECK(s.iteration == k);
            CHECK(s.remaining == cfg.n_time - k);
            CHECK(s.seq == k);
            CHECK_THAT(s.t_remaining, WithinAbs(cfg.T - static_cast<double>(k) * dtau, 1e-12));
            CHECK(s.boundary_len == k);  // 完了した反復ごとに S* が 1 点積まれる
        }
    }

    SECTION("after n_time steps nothing is left and further steps are no-ops (but seq still advances)") {
        advance(m, cfg.n_time);
        const FdmSceneSnapshot done = m.snapshot();
        CHECK(done.iteration == cfg.n_time);
        CHECK(done.remaining == 0);
        CHECK_THAT(done.t_remaining, WithinAbs(0.0, 1e-15));

        advance(m, 3);
        const FdmSceneSnapshot after = m.snapshot();
        CHECK(after.iteration == cfg.n_time);  // 反復は進まない
        CHECK(after.remaining == 0);
        CHECK(after.seq == done.seq + 3);  // seq は進む（Runner から見て「ステップした」ことは真）
        CHECK(after.values == done.values);
        CHECK(after.boundary_len == done.boundary_len);
    }

    SECTION("SetParam restarts the sweep; a SetParam that does not move the value leaves it alone") {
        advance(m, 10);
        REQUIRE(m.snapshot().iteration == 10);
        REQUIRE(m.snapshot().seq == 10);

        m.apply(Command::set_param(FdmAmericanModel::kSigma, 0.35));
        const FdmSceneSnapshot restarted = m.snapshot();
        CHECK(restarted.iteration == 0);  // PDE の途中で係数は差し替えられない → 満期からやり直す
        CHECK(restarted.remaining == cfg.n_time);
        CHECK(restarted.seq == 0);
        CHECK(restarted.boundary_len == 0);
        CHECK_THAT(restarted.sigma, WithinRel(0.35, 1e-15));
        CHECK_THAT(restarted.t_remaining, WithinRel(cfg.T, 1e-15));

        advance(m, 4);
        REQUIRE(m.snapshot().iteration == 4);
        m.apply(Command::set_param(FdmAmericanModel::kSigma, 0.35));  // 同じ値
        CHECK(m.snapshot().iteration == 4);                           // 巻き戻さない
        CHECK(m.snapshot().seq == 4);
    }
}

// これは「配線」のテストであって数値のテストではない: モデルが素の `FdmCn` と同じ格子・同じ順序で
// 同じ解を作り、それを Snapshot / Surface へ正しく写しているか（dual-run オラクル）だけを見る。
// CN/PSOR そのものの精度・収束次数・American の性質は FDM-01..11（tests/test_fdm_cn.cpp）が担う。
TEST_CASE("FDMSCENE-03: after the full sweep the scene matches a bare FdmCn run and the surface holds "
          "the payoff in row 0 and the final values in the last filled row",
          "[fdmscene][determinism]") {
    const auto       cfg = test_config();
    FdmAmericanModel m(cfg);
    advance(m, cfg.n_time);
    const FdmSceneSnapshot s = m.snapshot();

    // 素の FdmCn を同じ格子・同じパラメータで t = 0 まで回す。
    FdmCn bare;
    bare.init(grid_of(cfg), params_of(cfg));
    while (bare.step_backward()) {
    }
    REQUIRE(bare.remaining() == 0);

    CHECK_THAT(s.v_at_s0, WithinRel(bare.value_at(cfg.s0), 1e-12));
    CHECK_THAT(s.delta_at_s0, WithinRel(bare.delta_at(cfg.s0), 1e-12));
    CHECK_THAT(s.exercise_boundary_now, WithinRel(bare.exercise_boundary(), 1e-12));
    for (std::size_t i = 0; i < FdmSceneSnapshot::kCurve; ++i)
        CHECK_THAT(s.values[i], WithinRel(bare.value_at(s.spots[i]), 1e-12));

    auto surf = std::make_unique<FdmSceneSurface>();
    m.surface(*surf);
    CHECK(surf->filled_rows == FdmSceneSurface::kT);  // 掃引が終わったので全行が埋まっている

    // 行 0 = 満期（ペイオフ）、最後に埋まった行 = t = 0 の値。float なので許容は相対 1e-6。
    FdmCn payoff;
    payoff.init(grid_of(cfg), params_of(cfg));
    for (std::size_t j = 0; j < FdmSceneSurface::kS; ++j) {
        const double sj = static_cast<double>(surf->spots[j]);
        CHECK_THAT(static_cast<double>(surf->values[j]), WithinAbs(intrinsic(cfg, sj), 1e-4));
        CHECK_THAT(static_cast<double>(surf->values[j]), WithinAbs(payoff.value_at(sj), 1e-4));
    }
    const std::size_t last = FdmSceneSurface::kT - 1;
    for (std::size_t j = 0; j < FdmSceneSurface::kS; ++j) {
        const double sj = static_cast<double>(surf->spots[j]);
        CHECK_THAT(static_cast<double>(surf->values[last * FdmSceneSurface::kS + j]),
                   WithinAbs(bare.value_at(sj), 1e-4));
    }
    // 中間行も 1 本ピン留めする（端だけ合わせる実装を通さないため）。行 j に載るのはレベル
    // round(j · M / (kT − 1))（モデルのヘッダ「時間レベルと面の行の対応」）。
    {
        constexpr std::size_t kRow  = 100;
        constexpr std::size_t kDen  = FdmSceneSurface::kT - 1;
        const std::size_t     level = (kRow * cfg.n_time + kDen / 2) / kDen;
        REQUIRE(level > 0);
        REQUIRE(level < cfg.n_time);  // 端ではない = 中間行の検査になっている

        FdmCn mid;
        mid.init(grid_of(cfg), params_of(cfg));
        for (std::size_t i = 0; i < level; ++i) REQUIRE(mid.step_backward());
        CHECK_THAT(static_cast<double>(surf->times[kRow]), WithinAbs(mid.time(), 1e-6));
        for (std::size_t j = 0; j < FdmSceneSurface::kS; ++j) {
            const double sj = static_cast<double>(surf->spots[j]);
            CHECK_THAT(static_cast<double>(surf->values[kRow * FdmSceneSurface::kS + j]),
                       WithinAbs(mid.value_at(sj), 1e-4));
        }
    }

    // 時間軸は満期 → 今日（降順、最後は t = 0）。
    CHECK_THAT(static_cast<double>(surf->times[0]), WithinRel(cfg.T, 1e-6));
    CHECK_THAT(static_cast<double>(surf->times[last]), WithinAbs(0.0, 1e-7));
    for (std::size_t i = 1; i < FdmSceneSurface::kT; ++i) CHECK(surf->times[i] <= surf->times[i - 1]);
}

TEST_CASE("FDMSCENE-04: Reset puts the scene back on the terminal payoff", "[fdmscene][unit]") {
    const auto       cfg = test_config();
    FdmAmericanModel m(cfg);
    advance(m, 20);
    REQUIRE(m.snapshot().iteration == 20);
    REQUIRE(m.snapshot().boundary_len == 20);

    m.apply(Command::reset());
    const FdmSceneSnapshot s = m.snapshot();

    CHECK(s.iteration == 0);
    CHECK(s.remaining == cfg.n_time);
    CHECK(s.seq == 0);
    CHECK(s.boundary_len == 0);
    CHECK(s.status == FdmSceneSnapshot::kStatusOk);
    CHECK_THAT(s.t_remaining, WithinRel(cfg.T, 1e-15));
    // V(S) は本源的価値そのもの（K は格子点に乗るので線形補間でも折れ点が再現される）。
    for (std::size_t i = 0; i < FdmSceneSnapshot::kCurve; ++i)
        CHECK_THAT(s.values[i], WithinAbs(intrinsic(cfg, s.spots[i]), 1e-9));
    CHECK_THAT(s.v_at_s0, WithinAbs(intrinsic(cfg, cfg.s0), 1e-9));

    // 面も満期 1 行だけに戻る（未計算行はゼロ）。
    auto surf = std::make_unique<FdmSceneSurface>();
    m.surface(*surf);
    CHECK(surf->filled_rows >= 1);
    CHECK(surf->filled_rows < FdmSceneSurface::kT);
    for (std::size_t j = 0; j < FdmSceneSurface::kS; ++j)
        CHECK_THAT(static_cast<double>(surf->values[j]),
                   WithinAbs(intrinsic(cfg, static_cast<double>(surf->spots[j])), 1e-4));
    const std::size_t first_empty = surf->filled_rows;
    for (std::size_t j = 0; j < FdmSceneSurface::kS; ++j)
        CHECK(surf->values[first_empty * FdmSceneSurface::kS + j] == 0.f);
}
