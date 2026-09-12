// LSMSCENE-01..03 — scenes/lsm_model.hpp の契約 / 単体 / 決定性テスト（docs/03_tdd_spec.md §7.1）
//
// このシーンは「1 step = 後ろ向き回帰 1 時点」なので、テストも時点数で書く。数値（LSM の価格の
// 正しさ・収束・アンチセティックの性質）は LSM-01..07（tests/test_lsm.cpp）が担う。ここで見るのは
// 「モデルが素の `core::Lsm` と同じ入力で同じ掃引を回し、それを Snapshot へ正しく写しているか」
// （dual-run オラクル）と Model 契約だけ。
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/bridge/runner.hpp"
#include "quantviz/core/pricing/lsm.hpp"
#include "quantviz/scenes/lsm_model.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::bridge::Command;
using quantviz::core::Lsm;
using quantviz::core::LsmBasis;
using quantviz::core::LsmParams;
using quantviz::core::OptionType;
using quantviz::scenes::LsmModel;
using quantviz::scenes::LsmSceneSnapshot;

namespace {

/// テスト用は 1000 パス（掃引 64 時点が数 ms で終わる）。価格の正しさではなく配線を見るので本数は自由。
LsmModel::Config test_config() {
    LsmModel::Config cfg;
    cfg.K       = 100.0;
    cfg.T       = 1.0;
    cfg.r       = 0.05;
    cfg.sigma   = 0.20;
    cfg.s0      = 100.0;
    cfg.type    = OptionType::Put;
    cfg.n_paths = 1000;
    cfg.n_basis = 3;
    cfg.basis   = LsmBasis::Laguerre;
    cfg.seed    = 777;
    return cfg;
}

/// Config から「素の Lsm」のパラメータを組む（モデルと同じ規則: n_steps = Snapshot::kSteps）。
LsmParams params_of(const LsmModel::Config& c) {
    LsmParams p{};
    p.s0      = c.s0;
    p.K       = c.K;
    p.T       = c.T;
    p.r       = c.r;
    p.sigma   = c.sigma;
    p.type    = c.type;
    p.n_paths = c.n_paths;
    p.n_steps = LsmSceneSnapshot::kSteps;
    p.n_basis = c.n_basis;
    p.basis   = c.basis;
    p.seed    = c.seed;
    return p;
}

/// n 回 step する（dt はモデルが無視するので値は何でもよい）。
void advance(LsmModel& m, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) m.step(1.0 / 252.0);
}

/// 素の Lsm を t = 0 まで回す。
void run_bare(Lsm& bare, const LsmParams& p) {
    bare.init(p);
    while (bare.step_backward()) {
    }
}

}  // namespace

TEST_CASE("LSMSCENE-01: satisfies the Model contract with a fixed-size POD snapshot that starts unstepped",
          "[lsmscene][contract]") {
    STATIC_REQUIRE(quantviz::bridge::Model<LsmModel>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<LsmSceneSnapshot>);
    STATIC_REQUIRE(std::is_standard_layout_v<LsmSceneSnapshot>);
    STATIC_REQUIRE(std::is_default_constructible_v<LsmSceneSnapshot>);
    // 16 本 × 65 点 + 5 分位 × 65 点 + フィット 2 本 × 64 点 ≈ 12 KB（Greeks と同じ「32 KiB まで」の
    // 例外枠のうち、この Snapshot には 16 KiB を上限として課す）。
    STATIC_REQUIRE(sizeof(LsmSceneSnapshot) <= 16 * 1024);
    STATIC_REQUIRE(LsmSceneSnapshot::kShow == 16);
    STATIC_REQUIRE(LsmSceneSnapshot::kSteps == 64);
    STATIC_REQUIRE(LsmSceneSnapshot::kPts == LsmSceneSnapshot::kSteps + 1);
    // このシーンは面を持たない（パスは Snapshot 内で 16 本 + 分位帯に縮約する）。
    STATIC_REQUIRE_FALSE(quantviz::bridge::SurfaceModel<LsmModel>);

    const auto cfg = test_config();

    SECTION("a fresh model sits at maturity: seq 0, the whole sweep ahead, price == European MC") {
        const LsmModel         m(cfg);
        const LsmSceneSnapshot s = m.snapshot();
        CHECK(s.seq == 0);
        CHECK(s.t_index == LsmSceneSnapshot::kSteps);
        CHECK(s.remaining == LsmSceneSnapshot::kSteps);
        CHECK(s.exercised_paths == 0);
        CHECK(s.itm_now == 0);
        CHECK(s.itm_paths == 0);
        CHECK(s.fit_rank == 0);
        CHECK_FALSE(s.fit_valid);  // まだ 1 度も回帰していない = 保持しているフィットも無い
        CHECK_FALSE(s.fit_stale);
        // まだ 1 時点も回帰していない = 誰も早期行使していないので、価格は同じパスの European MC。
        CHECK(s.price == s.european);
        CHECK(s.std_error > 0.0);
        CHECK_THAT(s.t_now, WithinRel(cfg.T, 1e-15));

        CHECK_THAT(s.K, WithinRel(cfg.K, 1e-15));
        CHECK_THAT(s.r, WithinRel(cfg.r, 1e-15));
        CHECK_THAT(s.sigma, WithinRel(cfg.sigma, 1e-15));
        CHECK_THAT(s.T, WithinRel(cfg.T, 1e-15));
        CHECK_THAT(s.s0, WithinRel(cfg.s0, 1e-15));
        CHECK(s.n_paths == cfg.n_paths);
        CHECK(s.n_basis == cfg.n_basis);
        CHECK(s.basis == cfg.basis);
        CHECK(s.type == cfg.type);

        // 16 本ともパスは S0 から出て、行使時点は「満期まで持った」を指す。
        for (std::size_t i = 0; i < LsmSceneSnapshot::kShow; ++i) {
            CHECK_THAT(s.shown_paths[i * LsmSceneSnapshot::kPts], WithinRel(cfg.s0, 1e-15));
            CHECK(s.shown_exercise[i] == static_cast<std::int32_t>(LsmSceneSnapshot::kSteps));
        }
        // 分位帯は t = 0 では潰れ（全パスが S0）、満期に向かって広がる。
        for (std::size_t q = 0; q < LsmSceneSnapshot::kQuant; ++q)
            CHECK_THAT(s.quantiles[q * LsmSceneSnapshot::kPts], WithinRel(cfg.s0, 1e-15));
        const std::size_t last = LsmSceneSnapshot::kPts - 1;
        CHECK(s.quantiles[0 * LsmSceneSnapshot::kPts + last] <
              s.quantiles[(LsmSceneSnapshot::kQuant - 1) * LsmSceneSnapshot::kPts + last]);

        // 継続価値カーブの S 軸は昇順で K を含む（フィットはまだ無いので値は NaN）。
        for (std::size_t j = 1; j < LsmSceneSnapshot::kFit; ++j) CHECK(s.fit_s[j] > s.fit_s[j - 1]);
        CHECK(s.fit_s.front() <= cfg.K);
        CHECK(s.fit_s.back() >= cfg.K);
        for (std::size_t j = 0; j < LsmSceneSnapshot::kFit; ++j) CHECK(std::isnan(s.fit_v[j]));
    }

    SECTION("Reset rewinds the sweep and seq") {
        LsmModel m(cfg);
        advance(m, 7);
        REQUIRE(m.snapshot().seq == 7);
        REQUIRE(m.snapshot().t_index == LsmSceneSnapshot::kSteps - 7);

        m.apply(Command::reset());
        const LsmSceneSnapshot s = m.snapshot();
        CHECK(s.seq == 0);
        CHECK(s.t_index == LsmSceneSnapshot::kSteps);
        CHECK(s.remaining == LsmSceneSnapshot::kSteps);
        CHECK(s.exercised_paths == 0);
        CHECK(m.seed() == cfg.seed);  // seed 0 = 今の seed で再生
    }

    SECTION("Reset with a non-zero seed swaps the paths") {
        LsmModel m(cfg);
        const double before = m.snapshot().shown_paths[LsmSceneSnapshot::kPts - 1];
        m.apply(Command::reset(4242));
        CHECK(m.seed() == 4242);
        CHECK(m.snapshot().shown_paths[LsmSceneSnapshot::kPts - 1] != before);
        CHECK(m.snapshot().seq == 0);
    }

    SECTION("an unknown param_id is ignored") {
        LsmModel m(cfg);
        advance(m, 3);
        const LsmSceneSnapshot before = m.snapshot();
        m.apply(Command::set_param(0, 1.0));     // 0 は予約
        m.apply(Command::set_param(999, 42.0));  // 未知
        const LsmSceneSnapshot after = m.snapshot();
        CHECK(after.seq == before.seq);
        CHECK(after.t_index == before.t_index);
        CHECK(after.K == before.K);
        CHECK(after.price == before.price);
    }

    SECTION("SetParam clamps finite values to the documented range") {
        LsmModel m(cfg);
        m.apply(Command::set_param(LsmModel::kStrike, 1e9));
        CHECK_THAT(m.snapshot().K, WithinRel(LsmModel::kMaxStrike, 1e-15));
        m.apply(Command::set_param(LsmModel::kSigma, -1.0));
        CHECK(m.snapshot().sigma == LsmModel::kMinSigma);
        m.apply(Command::set_param(LsmModel::kRate, -10.0));
        CHECK_THAT(m.snapshot().r, WithinRel(LsmModel::kMinRate, 1e-15));
        m.apply(Command::set_param(LsmModel::kNBasis, 99.0));
        CHECK(m.snapshot().n_basis == LsmModel::kMaxBasis);
        m.apply(Command::set_param(LsmModel::kNBasis, 0.0));
        CHECK(m.snapshot().n_basis == LsmModel::kMinBasis);
        m.apply(Command::set_param(LsmModel::kNPaths, 1.0));
        CHECK(m.snapshot().n_paths == LsmModel::kMinPaths);
        m.apply(Command::set_param(LsmModel::kNPaths, 1e9));
        CHECK(m.snapshot().n_paths == LsmModel::kMaxPaths);
        m.apply(Command::set_param(LsmModel::kNPaths, 999.0));  // 奇数は偶数へ切り上げ
        CHECK(m.snapshot().n_paths == 1000);
        m.apply(Command::set_param(LsmModel::kBasis, 0.0));
        CHECK(m.snapshot().basis == LsmBasis::Power);
        m.apply(Command::set_param(LsmModel::kBasis, 1.0));
        CHECK(m.snapshot().basis == LsmBasis::Laguerre);
    }

    SECTION("SetParam rejects NaN and infinity (the sweep is not even rewound)") {
        LsmModel m(cfg);
        advance(m, 4);
        const LsmSceneSnapshot  before = m.snapshot();
        constexpr std::uint32_t kIds[] = {LsmModel::kStrike, LsmModel::kSigma,  LsmModel::kRate,
                                          LsmModel::kBasis,  LsmModel::kNBasis, LsmModel::kNPaths};
        for (const std::uint32_t id : kIds) {
            m.apply(Command::set_param(id, std::numeric_limits<double>::quiet_NaN()));
            m.apply(Command::set_param(id, std::numeric_limits<double>::infinity()));
            m.apply(Command::set_param(id, -std::numeric_limits<double>::infinity()));
        }
        const LsmSceneSnapshot after = m.snapshot();
        CHECK(after.K == before.K);
        CHECK(after.sigma == before.sigma);
        CHECK(after.r == before.r);
        CHECK(after.basis == before.basis);
        CHECK(after.n_basis == before.n_basis);
        CHECK(after.n_paths == before.n_paths);
        CHECK(after.t_index == before.t_index);  // 拒否した = 掃引をやり直していない
        CHECK(after.seq == before.seq);
    }

    SECTION("Runner: a tick publishes one snapshot per backward point") {
        namespace bridge = quantviz::bridge;
        bridge::RunnerConfig rc;
        rc.dt                     = 1.0 / 252.0;
        rc.clock.steps_per_second = 100.0;
        rc.publish_every          = 1;

        // Snapshot が 12 KB なのでリング（8 枚）ごとヒープに置く。
        auto r = std::make_unique<bridge::Runner<LsmModel, 8, 8>>(LsmModel{cfg}, rc);
        CHECK(r->tick(0.05) == 5);  // 100 steps/s × 0.05 s

        LsmSceneSnapshot s{};
        std::size_t      n = 0;
        while (r->poll(s)) ++n;
        CHECK(n == 5);
        CHECK(s.seq == 5);
        CHECK(s.t_index == LsmSceneSnapshot::kSteps - 5);
        CHECK(s.remaining == LsmSceneSnapshot::kSteps - 5);
    }
}

TEST_CASE("LSMSCENE-02: one step is exactly one backward time point and the sweep holds at t = 0",
          "[lsmscene][unit]") {
    const auto cfg = test_config();
    LsmModel   m(cfg);

    const double dt = cfg.T / static_cast<double>(LsmSceneSnapshot::kSteps);

    SECTION("each step walks t_index and remaining back by exactly one point") {
        for (std::size_t k = 1; k <= 5; ++k) {
            m.step(1.0 / 252.0);
            const LsmSceneSnapshot s = m.snapshot();
            CHECK(s.seq == k);
            CHECK(s.t_index == LsmSceneSnapshot::kSteps - k);
            CHECK(s.remaining == LsmSceneSnapshot::kSteps - k);
            CHECK_THAT(s.t_now, WithinAbs(cfg.T - static_cast<double>(k) * dt, 1e-12));
        }
    }

    SECTION("after kSteps steps nothing is left and further steps are no-ops (but seq still advances)") {
        advance(m, LsmSceneSnapshot::kSteps);
        const LsmSceneSnapshot done = m.snapshot();
        CHECK(done.t_index == 0);
        CHECK(done.remaining == 0);
        CHECK(done.seq == LsmSceneSnapshot::kSteps);
        CHECK_THAT(done.t_now, WithinAbs(0.0, 1e-15));

        advance(m, 3);
        const LsmSceneSnapshot after = m.snapshot();
        CHECK(after.t_index == 0);  // 反復は進まない
        CHECK(after.remaining == 0);
        CHECK(after.seq == done.seq + 3);  // seq は進む（Runner から見て「ステップした」ことは真）
        CHECK(after.price == done.price);
        CHECK(after.exercised_paths == done.exercised_paths);
        CHECK(after.coeffs == done.coeffs);
        CHECK(after.fit_v == done.fit_v);
    }

    SECTION("the American price only beats the European one once the sweep has walked back") {
        const LsmSceneSnapshot start = m.snapshot();
        REQUIRE(start.price == start.european);
        advance(m, LsmSceneSnapshot::kSteps);
        const LsmSceneSnapshot done = m.snapshot();
        CHECK(done.european == start.european);  // European は同じパスの満期ペイオフ（掃引で変わらない）
        CHECK(done.price > done.european);       // 早期行使の権利ぶん高い
        CHECK(done.exercised_paths > 0);
    }

    // 保持するフィット（lsm_model.hpp「フィットは保持する」）の不変条件を掃引全体で見る。
    // K = 50 は「どの行使日にも ITM パスが 1 本も無い」= 一度も回帰できない設定（put が深い OTM。
    // この seed / σ では 64 時点すべてで itm_now == 0）、K = 60 は「回帰できる時点とできない時点が
    // 混ざる」設定（この seed で 26 時点はフィットあり、38 時点は保持に回る）。
    SECTION("the held fit never claims a point it does not belong to") {
        auto invariants = [](const LsmSceneSnapshot& s) {
            if (s.fit_valid) {
                // 保持に回ったフィットは必ず「もっと満期側の時点」のもの（掃引は後ろ向き）。
                if (s.fit_stale)
                    CHECK(s.fit_t_index > s.t_index);
                else
                    CHECK(s.fit_t_index == s.t_index);
            } else {
                CHECK_FALSE(s.fit_stale);  // 1 つも持っていないなら「保持」でもない
                const bool all_nan = std::all_of(s.fit_v.begin(), s.fit_v.end(),
                                                 [](double v) { return std::isnan(v); });
                CHECK(all_nan);
            }
            // この時点で ITM パスがあれば回帰できている = 保持ではなく現在の時点のフィット。
            if (s.itm_now > 0) {
                CHECK(s.fit_valid);
                CHECK_FALSE(s.fit_stale);
                CHECK(s.fit_t_index == s.t_index);
            }
        };

        auto deep_otm     = cfg;
        deep_otm.K        = 50.0;
        LsmModel    m50(deep_otm);
        std::size_t fitted50 = 0;
        for (std::size_t k = 0; k < LsmSceneSnapshot::kSteps; ++k) {
            m50.step(1.0);
            const LsmSceneSnapshot s = m50.snapshot();
            invariants(s);
            CHECK(s.itm_now == 0);   // 深い OTM: どの時点にも ITM パスが無い
            CHECK_FALSE(s.fit_valid);
            if (s.fit_valid) ++fitted50;
        }
        CHECK(fitted50 == 0);
        CHECK(m50.snapshot().price == 0.0);            // 誰も行使せず満期ペイオフも 0
        CHECK(m50.snapshot().exercised_paths == 0);

        auto mixed = cfg;
        mixed.K    = 60.0;
        LsmModel    m60(mixed);
        std::size_t fresh = 0, held = 0;
        for (std::size_t k = 0; k < LsmSceneSnapshot::kSteps; ++k) {
            m60.step(1.0);
            const LsmSceneSnapshot s = m60.snapshot();
            invariants(s);
            if (s.fit_valid && !s.fit_stale) ++fresh;
            if (s.fit_stale) ++held;
        }
        CHECK(fresh > 0);  // 両方の枝を通ったことを確かめる（通らないと不変条件が空回りする）
        CHECK(held > 0);
    }

    SECTION("SetParam restarts the sweep; a SetParam that does not move the value is a no-op") {
        advance(m, 10);
        REQUIRE(m.snapshot().seq == 10);

        m.apply(Command::set_param(LsmModel::kSigma, 0.35));
        const LsmSceneSnapshot restarted = m.snapshot();
        CHECK(restarted.seq == 0);  // 回帰の途中で σ は差し替えられない → 満期からやり直す
        CHECK(restarted.t_index == LsmSceneSnapshot::kSteps);
        CHECK(restarted.remaining == LsmSceneSnapshot::kSteps);
        CHECK(restarted.exercised_paths == 0);
        CHECK_THAT(restarted.sigma, WithinRel(0.35, 1e-15));
        CHECK_THAT(restarted.t_now, WithinRel(cfg.T, 1e-15));

        advance(m, 4);
        REQUIRE(m.snapshot().seq == 4);
        const LsmSceneSnapshot before = m.snapshot();
        m.apply(Command::set_param(LsmModel::kSigma, 0.35));   // 同じ値
        m.apply(Command::set_param(LsmModel::kStrike, 100.0)); // 同じ値
        m.apply(Command::set_param(LsmModel::kNPaths, 999.0)); // 偶数へ丸めると同じ値（1000）
        const LsmSceneSnapshot after = m.snapshot();
        CHECK(after.seq == 4);  // 巻き戻さない
        CHECK(after.t_index == before.t_index);
        CHECK(after.price == before.price);
    }
}

TEST_CASE("LSMSCENE-03: the finished sweep reproduces a bare core::Lsm run bit for bit",
          "[lsmscene][determinism]") {
    const auto cfg = test_config();

    Lsm bare;
    run_bare(bare, params_of(cfg));
    const auto expected = bare.result();

    LsmModel m(cfg);
    advance(m, LsmSceneSnapshot::kSteps);
    const LsmSceneSnapshot s = m.snapshot();
    REQUIRE(s.remaining == 0);

    // 配線のテストなので許容はゼロ: 同じ入力・同じ順序なら bit 一致でなければならない。
    CHECK(s.price == expected.price);
    CHECK(s.std_error == expected.std_error);
    CHECK(s.european == expected.european_price);
    CHECK(s.exercised_paths == expected.exercised_paths);

    // 保持しているフィット: 既定は ATM（S0 = K）なので最後の時点 k = 0 には ITM パスが無く、
    // 掃引を終えた Snapshot は k = 1 のフィットを保持している（lsm_model.hpp「フィットは保持する」）。
    CHECK(s.itm_now == 0);
    CHECK(s.fit_valid);
    CHECK(s.fit_stale);
    CHECK(s.fit_t_index == 1);
    // その時点まで回した素の Lsm と係数・カーブが bit 一致すること。
    Lsm held;
    held.init(params_of(cfg));
    while (held.remaining() > s.fit_t_index) REQUIRE(held.step_backward());
    CHECK(s.fit_rank == static_cast<std::uint32_t>(held.last_fit_rank()));
    CHECK(s.itm_paths == static_cast<std::uint32_t>(held.last_itm_paths()));
    CHECK(s.fit_fallback == held.last_fit_fallback());
    CHECK(s.fit_rank > 0);
    for (std::size_t j = 0; j < s.n_basis; ++j) CHECK(s.coeffs[j] == held.continuation_coeffs()[j]);
    for (std::size_t j = 0; j < LsmSceneSnapshot::kFit; ++j) {
        REQUIRE(std::isfinite(s.fit_v[j]));
        CHECK(s.fit_v[j] == held.continuation_value(s.fit_s[j]));
    }
    // 分位帯も素の Lsm と同じ（描画用の縮約が別の列を読んでいないことの確認）。
    for (std::size_t q = 0; q < LsmSceneSnapshot::kQuant; ++q)
        for (std::size_t k = 0; k < LsmSceneSnapshot::kPts; k += 8)
            CHECK(s.quantiles[q * LsmSceneSnapshot::kPts + k] ==
                  bare.quantile_at(k, LsmSceneSnapshot::kQuantLevels[q]));

    SECTION("Reset replays the same seed and lands on the same price") {
        m.apply(Command::reset());
        REQUIRE(m.snapshot().seq == 0);
        advance(m, LsmSceneSnapshot::kSteps);
        const LsmSceneSnapshot again = m.snapshot();
        CHECK(again.price == s.price);
        CHECK(again.std_error == s.std_error);
        CHECK(again.shown_paths == s.shown_paths);
        CHECK(again.quantiles == s.quantiles);
    }

    SECTION("SetParam re-inits with the same seed: seq 0 and a different price") {
        m.apply(Command::set_param(LsmModel::kSigma, 0.40));
        CHECK(m.snapshot().seq == 0);
        advance(m, LsmSceneSnapshot::kSteps);
        const LsmSceneSnapshot hot = m.snapshot();
        CHECK(hot.price != s.price);
        CHECK(hot.price > s.price);  // σ を上げれば put は高くなる

        LsmModel::Config hot_cfg = cfg;
        hot_cfg.sigma            = 0.40;
        Lsm hot_bare;
        run_bare(hot_bare, params_of(hot_cfg));
        CHECK(hot.price == hot_bare.result().price);  // 同じ seed の別パラメータでも bit 一致
    }
}
