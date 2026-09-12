// ACSCENE-01..02 — scenes/exec_model.hpp の契約 / 単体テスト（docs/03_tdd_spec.md §7.2）
//
// このシーンは「掃引」ではない: 軌道・コスト・フロンティア・面はパラメータの純関数で、`step()` は
// アニメーションの再生ヘッド t を進めるだけ。したがってテストも「配線」を見る:
// コア（`core/exec/almgren_chriss.hpp`）の純関数が返す値がそのまま Snapshot / Surface に載っているか、
// SetParam がその場で効くか、step が t と seq 以外を動かさないか。数値そのもの（閉形式・TWAP 極限・
// フロンティアの単調性）は AC-01..07（tests/test_almgren_chriss.cpp）が担う。
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/bridge/runner.hpp"
#include "quantviz/core/exec/almgren_chriss.hpp"
#include "quantviz/scenes/exec_model.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::bridge::Command;
using quantviz::core::AcCost;
using quantviz::core::AcParams;
using quantviz::scenes::ExecModel;
using quantviz::scenes::ExecSnapshot;
using quantviz::scenes::ExecSurface;

namespace {

/// 素のコアで λ だけ差し替えた軌道を引く（dual-run オラクル）。
std::array<double, ExecSnapshot::kN> bare_trajectory(const AcParams& base, double lambda) {
    AcParams p = base;
    p.lambda   = lambda;
    std::array<double, ExecSnapshot::kN> x{};
    REQUIRE(quantviz::core::ac_trajectory(p, x) == ExecSnapshot::kN);
    return x;
}

void advance(ExecModel& m, std::size_t n, double dt = 0.02) {
    for (std::size_t i = 0; i < n; ++i) m.step(dt);
}

}  // namespace

TEST_CASE("ACSCENE-01: satisfies the Model and SurfaceModel contracts and mirrors the bare Almgren-Chriss "
          "core in fixed-size POD payloads",
          "[acscene][contract]") {
    STATIC_REQUIRE(quantviz::bridge::Model<ExecModel>);
    STATIC_REQUIRE(quantviz::bridge::SurfaceModel<ExecModel>);

    STATIC_REQUIRE(std::is_trivially_copyable_v<ExecSnapshot>);
    STATIC_REQUIRE(std::is_standard_layout_v<ExecSnapshot>);
    STATIC_REQUIRE(std::is_default_constructible_v<ExecSnapshot>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<ExecSurface>);
    STATIC_REQUIRE(std::is_standard_layout_v<ExecSurface>);
    STATIC_REQUIRE(std::is_default_constructible_v<ExecSurface>);

    // 状態の上限（docs/01_design.md §4「Snapshot のサイズ方針」の例外枠）。
    STATIC_REQUIRE(sizeof(ExecSnapshot) <= 8192);
    STATIC_REQUIRE(sizeof(ExecSurface) <= 16384);
    // 軌道 1 本 = ac_trajectory の出力そのもの（n_steps + 1 点）。面の t 軸も同じ節点。
    STATIC_REQUIRE(ExecSnapshot::kN == ExecModel::kSteps + 1);
    STATIC_REQUIRE(ExecSurface::kT == ExecSnapshot::kN);

    const ExecModel    m;
    const ExecSnapshot s = m.snapshot();

    SECTION("a fresh model has not stepped: seq 0, the playhead at t = 0") {
        CHECK(s.seq == 0);
        CHECK(s.t == 0.0);
        CHECK(s.params.n_steps == ExecModel::kSteps);
        CHECK_THAT(s.x_now, WithinRel(s.params.X, 1e-15));  // t = 0 ではまだ何も売っていない
        CHECK_THAT(s.kappa, WithinRel(quantviz::core::ac_kappa(s.params), 1e-15));
    }

    SECTION("the lambda ladder is ascending, log-spaced and straddles the current lambda") {
        for (std::size_t l = 1; l < ExecSnapshot::kL; ++l) CHECK(s.lambdas[l] > s.lambdas[l - 1]);
        CHECK(s.lambdas.front() < s.params.lambda);
        CHECK(s.lambdas.back() > s.params.lambda);
        // 等比数列（log10 の等間隔）
        const double ratio = s.lambdas[1] / s.lambdas[0];
        for (std::size_t l = 2; l < ExecSnapshot::kL; ++l)
            CHECK_THAT(s.lambdas[l] / s.lambdas[l - 1], WithinRel(ratio, 1e-12));
        // 張る幅はちょうど 2 · kLadderDecades 桁（現在の λ の周り ±kLadderDecades 桁）。
        CHECK_THAT(s.lambdas.back() / s.lambdas.front(),
                   WithinRel(std::pow(10.0, 2.0 * ExecModel::kLadderDecades), 1e-12));
        CHECK_THAT(s.params.lambda / s.lambdas.front(),
                   WithinRel(std::pow(10.0, ExecModel::kLadderDecades), 1e-12));
    }

    SECTION("every ladder row is bit-identical to ac_trajectory of the bare core") {
        for (std::size_t l = 0; l < ExecSnapshot::kL; ++l) {
            const auto want = bare_trajectory(s.params, s.lambdas[l]);
            for (std::size_t n = 0; n < ExecSnapshot::kN; ++n)
                CHECK(s.trajectories[l * ExecSnapshot::kN + n] == want[n]);
            const AcCost c = [&] {
                AcParams p = s.params;
                p.lambda   = s.lambdas[l];
                return quantviz::core::ac_cost(p);
            }();
            CHECK(s.costs[l].expected == c.expected);
            CHECK(s.costs[l].variance == c.variance);
        }
        // 現在の λ の軌道・コストも同じ規則（パネルが太線で描く 1 本）。
        const auto want = bare_trajectory(s.params, s.params.lambda);
        for (std::size_t n = 0; n < ExecSnapshot::kN; ++n) CHECK(s.trajectory[n] == want[n]);
        const AcCost c = quantviz::core::ac_cost(s.params);
        CHECK(s.cost.expected == c.expected);
        CHECK(s.cost.variance == c.variance);
    }

    SECTION("the frontier grid is the bare ac_frontier over a fixed log-spaced lambda range") {
        for (std::size_t i = 1; i < ExecSnapshot::kFrontier; ++i)
            CHECK(s.frontier_lambdas[i] > s.frontier_lambdas[i - 1]);
        CHECK_THAT(s.frontier_lambdas.front(), WithinRel(ExecModel::kMinLambda, 1e-12));
        CHECK_THAT(s.frontier_lambdas.back(), WithinRel(ExecModel::kMaxLambda, 1e-12));

        std::array<AcCost, ExecSnapshot::kFrontier> want{};
        REQUIRE(quantviz::core::ac_frontier(s.params, s.frontier_lambdas, want) ==
                ExecSnapshot::kFrontier);
        for (std::size_t i = 0; i < ExecSnapshot::kFrontier; ++i) {
            CHECK(s.frontier[i].expected == want[i].expected);
            CHECK(s.frontier[i].variance == want[i].variance);
        }
        // 効率的フロンティア: λ↑ で期待コスト↑・分散↓（AC-06 の性質がそのまま載っている）。
        for (std::size_t i = 1; i < ExecSnapshot::kFrontier; ++i) {
            CHECK(s.frontier[i].expected > s.frontier[i - 1].expected);
            CHECK(s.frontier[i].variance < s.frontier[i - 1].variance);
        }
    }

    SECTION("the surface is finite everywhere, starts at X and is fully liquidated at T") {
        auto surf = std::make_unique<ExecSurface>();
        m.surface(*surf);

        CHECK(surf->ts.front() == 0.f);
        CHECK_THAT(static_cast<double>(surf->ts.back()), WithinRel(s.params.T, 1e-6));
        for (std::size_t i = 1; i < ExecSurface::kT; ++i) CHECK(surf->ts[i] > surf->ts[i - 1]);
        for (std::size_t l = 1; l < ExecSurface::kL; ++l) CHECK(surf->lambdas[l] > surf->lambdas[l - 1]);

        for (std::size_t l = 0; l < ExecSurface::kL; ++l) {
            const float* const row = surf->x.data() + l * ExecSurface::kT;
            CHECK_THAT(static_cast<double>(row[0]), WithinRel(s.params.X, 1e-6));  // t = 0 で全量
            CHECK_THAT(static_cast<double>(row[ExecSurface::kT - 1]), WithinAbs(0.0, 1e-6));  // t = T
            for (std::size_t i = 0; i < ExecSurface::kT; ++i) {
                REQUIRE(std::isfinite(row[i]));          // メッシュに NaN を渡さない
                CHECK(row[i] >= 0.f);
                if (i > 0) CHECK(row[i] <= row[i - 1]);  // 売り切りなので残量は単調非増加
            }
        }

        // 面の 1 行 = その行の λ で引いた ac_trajectory そのもの（面は独自の λ 軸 32 本を張るので、
        // Snapshot の梯子 8 本とは別の経路。ここを見ないと面側の λ 軸がテストされない）。
        // 許容は float 格納ぶん + λ を float で読み直すぶん（相対 1e-6。x ~ 1e6 なので絶対 1e-3 も許す）。
        for (std::size_t l = 0; l < ExecSurface::kL; ++l) {
            const auto         want = bare_trajectory(s.params, static_cast<double>(surf->lambdas[l]));
            const float* const row  = surf->x.data() + l * ExecSurface::kT;
            for (std::size_t i = 0; i < ExecSurface::kT; ++i)
                CHECK_THAT(static_cast<double>(row[i]),
                           WithinRel(want[i], 1e-6) || WithinAbs(want[i], 1e-3));
        }
    }

    SECTION("an unknown param_id is ignored") {
        ExecModel          mm;
        advance(mm, 3);
        const ExecSnapshot before = mm.snapshot();
        mm.apply(Command::set_param(0, 1.0));     // 0 は予約
        mm.apply(Command::set_param(999, 42.0));  // 未知
        const ExecSnapshot after = mm.snapshot();
        CHECK(after.seq == before.seq);
        CHECK(after.params.lambda == before.params.lambda);
        CHECK(after.trajectory == before.trajectory);
    }

    SECTION("SetParam clamps finite values and rejects NaN / infinity") {
        ExecModel mm;
        mm.apply(Command::set_param(ExecModel::kLambda, 1e9));
        CHECK_THAT(mm.snapshot().params.lambda, WithinRel(ExecModel::kMaxLambda, 1e-15));
        mm.apply(Command::set_param(ExecModel::kSigma, -1.0));
        CHECK_THAT(mm.snapshot().params.sigma, WithinRel(ExecModel::kMinSigma, 1e-15));
        mm.apply(Command::set_param(ExecModel::kEta, 1e9));
        CHECK_THAT(mm.snapshot().params.eta, WithinRel(ExecModel::kMaxEta, 1e-15));
        mm.apply(Command::set_param(ExecModel::kGamma, -1.0));
        CHECK_THAT(mm.snapshot().params.gamma, WithinAbs(ExecModel::kMinGamma, 1e-18));
        mm.apply(Command::set_param(ExecModel::kT, 1e9));
        CHECK_THAT(mm.snapshot().params.T, WithinRel(ExecModel::kMaxT, 1e-15));

        const ExecSnapshot before = mm.snapshot();
        constexpr std::uint32_t kIds[] = {ExecModel::kLambda, ExecModel::kEta, ExecModel::kGamma,
                                          ExecModel::kSigma, ExecModel::kT};
        for (const std::uint32_t id : kIds) {
            mm.apply(Command::set_param(id, std::numeric_limits<double>::quiet_NaN()));
            mm.apply(Command::set_param(id, std::numeric_limits<double>::infinity()));
            mm.apply(Command::set_param(id, -std::numeric_limits<double>::infinity()));
        }
        const ExecSnapshot after = mm.snapshot();
        CHECK(after.params.lambda == before.params.lambda);
        CHECK(after.params.eta == before.params.eta);
        CHECK(after.params.gamma == before.params.gamma);
        CHECK(after.params.sigma == before.params.sigma);
        CHECK(after.params.T == before.params.T);
        CHECK(after.trajectory == before.trajectory);
    }

    SECTION("Reset rewinds the playhead and the sequence number but keeps the parameters") {
        ExecModel mm;
        mm.apply(Command::set_param(ExecModel::kLambda, 5e-6));
        advance(mm, 10);
        REQUIRE(mm.snapshot().seq == 10);
        REQUIRE(mm.snapshot().t > 0.0);

        mm.apply(Command::reset());
        const ExecSnapshot r = mm.snapshot();
        CHECK(r.seq == 0);
        CHECK(r.t == 0.0);
        CHECK_THAT(r.params.lambda, WithinRel(5e-6, 1e-15));
        CHECK_THAT(r.x_now, WithinRel(r.params.X, 1e-15));
    }

    SECTION("Runner: tick publishes snapshots on the ring and the newest surface on the triple buffer") {
        namespace bridge = quantviz::bridge;
        bridge::RunnerConfig rc;
        rc.dt                     = 0.02;
        rc.clock.steps_per_second = 100.0;
        rc.publish_every          = 1;
        rc.surface_every          = 1;

        auto r    = std::make_unique<bridge::Runner<ExecModel, 8, 8>>(ExecModel{}, rc);
        auto surf = std::make_unique<ExecSurface>();
        CHECK_FALSE(r->poll_surface(*surf));  // まだ 1 枚も出ていない

        CHECK(r->tick(0.05) == 5);  // 100 steps/s × 0.05 s

        ExecSnapshot got{};
        std::size_t  n = 0;
        while (r->poll(got)) ++n;
        CHECK(n == 5);
        CHECK(got.seq == 5);
        CHECK_THAT(got.t, WithinRel(5.0 * rc.dt, 1e-12));

        REQUIRE(r->surfaces_published() == 5);
        REQUIRE(r->poll_surface(*surf));  // 最新 1 枚だけ
        CHECK_FALSE(r->poll_surface(*surf));
    }
}

TEST_CASE("ACSCENE-02: SetParam reshapes the trajectories and the surface at once, while a step only "
          "moves the playhead",
          "[acscene][unit]") {
    SECTION("SetParam(lambda) front-loads the program: every interior point drops") {
        ExecModel m;
        m.apply(Command::set_param(ExecModel::kLambda, 1e-7));
        const ExecSnapshot slow = m.snapshot();
        auto               a    = std::make_unique<ExecSurface>();
        m.surface(*a);

        m.apply(Command::set_param(ExecModel::kLambda, 1e-4));  // リスク回避を上げる
        const ExecSnapshot fast = m.snapshot();
        auto               b    = std::make_unique<ExecSurface>();
        m.surface(*b);

        CHECK_THAT(fast.params.lambda, WithinRel(1e-4, 1e-15));
        CHECK(fast.seq == slow.seq);  // 掃引ではないので通番は巻き戻らない
        // 端点は両方とも X と 0。内点は全て小さくなる（前倒し = AC-04 の性質）。
        CHECK(fast.trajectory.front() == slow.trajectory.front());
        CHECK(fast.trajectory.back() == slow.trajectory.back());
        for (std::size_t n = 1; n + 1 < ExecSnapshot::kN; ++n)
            CHECK(fast.trajectory[n] < slow.trajectory[n]);
        // コストの向き: 期待コスト↑・分散↓
        CHECK(fast.cost.expected > slow.cost.expected);
        CHECK(fast.cost.variance < slow.cost.variance);

        // 面も同じフレームで作り直される（λ 軸は現在の λ を中心に張るので中身も軸も動く）。
        CHECK(b->lambdas != a->lambdas);
        CHECK(b->x != a->x);
        const std::size_t mid = ExecSurface::kT / 2;
        for (std::size_t l = 0; l < ExecSurface::kL; ++l)
            CHECK(b->x[l * ExecSurface::kT + mid] < a->x[l * ExecSurface::kT + mid]);
    }

    SECTION("a SetParam that does not move the value changes nothing") {
        ExecModel m;
        advance(m, 4);
        const ExecSnapshot before = m.snapshot();
        auto               a      = std::make_unique<ExecSurface>();
        m.surface(*a);
        // 構築時に 1 回、面は最初の surface() で 1 回。step は どちらも走らせない。
        REQUIRE(m.recompute_count() == 1);
        REQUIRE(m.surface_builds() == 1);
        m.surface(*a);
        CHECK(m.surface_builds() == 1);  // dirty が立っていなければ作り直さない

        m.apply(Command::set_param(ExecModel::kLambda, before.params.lambda));
        m.apply(Command::set_param(ExecModel::kSigma, before.params.sigma));
        m.apply(Command::set_param(0, 1.0));     // 予約
        m.apply(Command::set_param(999, 42.0));  // 未知
        const ExecSnapshot after = m.snapshot();
        auto               b     = std::make_unique<ExecSurface>();
        m.surface(*b);

        // 「結果が同じ」だけでなく「仕事をしていない」ことを見る（同じ値の再送で 2000 点以上を
        // 舐め直さない、という設計の主張そのもの）。
        CHECK(m.recompute_count() == 1);
        CHECK(m.surface_builds() == 1);

        CHECK(after.seq == before.seq);
        CHECK(after.t == before.t);
        CHECK(after.trajectory == before.trajectory);
        CHECK(after.trajectories == before.trajectories);
        CHECK(after.lambdas == before.lambdas);
        CHECK(after.cost.expected == before.cost.expected);
        CHECK(after.cost.variance == before.cost.variance);
        CHECK(b->x == a->x);
        CHECK(b->lambdas == a->lambdas);

        // 本当に動く値なら、ちょうど 1 回ずつ走る（面は次の surface() まで遅延する）。
        m.apply(Command::set_param(ExecModel::kSigma, before.params.sigma * 1.1));
        CHECK(m.recompute_count() == 2);
        CHECK(m.surface_builds() == 1);  // まだ誰も面を読んでいない
        m.surface(*b);
        CHECK(m.surface_builds() == 2);
        CHECK(b->x != a->x);
        m.step(0.02);  // step は軌道にも面にも触れない
        CHECK(m.recompute_count() == 2);
        m.surface(*b);
        CHECK(m.surface_builds() == 2);
    }

    SECTION("SetParam(eta / gamma / sigma / T) reshapes the program too") {
        ExecModel          m;
        const ExecSnapshot base = m.snapshot();

        m.apply(Command::set_param(ExecModel::kEta, base.params.eta * 4.0));
        CHECK(m.snapshot().trajectory != base.trajectory);  // η↑ で κ↓ → TWAP 寄り
        CHECK(m.snapshot().kappa < base.kappa);

        ExecModel m2;
        m2.apply(Command::set_param(ExecModel::kSigma, base.params.sigma * 1.5));
        CHECK(m2.snapshot().kappa > base.kappa);  // σ↑ でリスクが増える → 前倒し

        ExecModel m3;
        m3.apply(Command::set_param(ExecModel::kT, base.params.T * 2.0));
        const ExecSnapshot longer = m3.snapshot();
        CHECK_THAT(longer.params.T, WithinRel(base.params.T * 2.0, 1e-15));
        auto surf = std::make_unique<ExecSurface>();
        m3.surface(*surf);
        CHECK_THAT(static_cast<double>(surf->ts.back()), WithinRel(longer.params.T, 1e-6));

        ExecModel m4;
        m4.apply(Command::set_param(ExecModel::kGamma, base.params.gamma * 4.0));
        // 恒久インパクトは軌道の形をほぼ変えない（η~ 経由で僅かに効く）が、期待コストは上がる。
        CHECK(m4.snapshot().cost.expected > base.cost.expected);
    }

    SECTION("step only moves the playhead and the sequence number") {
        ExecModel          m;
        const ExecSnapshot before = m.snapshot();
        m.step(0.25);
        const ExecSnapshot after = m.snapshot();

        CHECK(after.seq == before.seq + 1);
        CHECK_THAT(after.t, WithinRel(0.25, 1e-15));
        CHECK(after.trajectory == before.trajectory);
        CHECK(after.trajectories == before.trajectories);
        CHECK(after.lambdas == before.lambdas);
        CHECK(after.frontier_lambdas == before.frontier_lambdas);
        CHECK(after.params.lambda == before.params.lambda);
        CHECK(after.kappa == before.kappa);
        // 再生ヘッドの残量は軌道の線形補間（t = 0 では X、T では 0）。
        CHECK(after.x_now < before.x_now);
        CHECK(after.x_now < after.params.X);

        // 面は t に依らない（x(t; λ) はパラメータの関数）。
        auto a = std::make_unique<ExecSurface>();
        auto b = std::make_unique<ExecSurface>();
        m.surface(*a);
        advance(m, 5);
        m.surface(*b);
        CHECK(b->x == a->x);
    }

    SECTION("the playhead wraps modulo T instead of running off the end") {
        ExecModel m;
        const double T = m.snapshot().params.T;
        advance(m, 400, T * 0.137);  // T を割り切らない刻みで 54 周ぶん
        const ExecSnapshot s = m.snapshot();
        CHECK(s.seq == 400);
        CHECK(s.t > 0.0);
        CHECK(s.t < T);
        CHECK(std::isfinite(s.x_now));
        CHECK(s.x_now >= 0.0);
        CHECK(s.x_now <= s.params.X);
    }
}
