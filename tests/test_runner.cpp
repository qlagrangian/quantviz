// RUNNER-xx — bridge/runner.hpp の仕様テスト。
// 時間に依存しない tick() を主に使い、スレッド起動は統合テスト（RUNNER-07/08 と RUNNER-11 の
// 最後の SECTION）だけに絞る。スレッドを使うケースは TSan の対象。
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>
#include <type_traits>
#include <vector>

#include "quantviz/bridge/runner.hpp"

using namespace quantviz::bridge;

namespace {

struct CounterSnapshot {
    std::uint64_t steps   = 0;
    double        last_dt = 0.0;
    double        param   = 0.0;
    std::uint64_t resets  = 0;
};

class CounterModel {
public:
    using Snapshot = CounterSnapshot;
    void     step(double dt) { ++s_.steps; s_.last_dt = dt; }
    Snapshot snapshot() const noexcept { return s_; }
    void     apply(const Command& c) {
        if (c.type == CommandType::SetParam && c.param_id == 1) s_.param = c.value;
        if (c.type == CommandType::Reset) { s_.steps = 0; ++s_.resets; }
    }
private:
    Snapshot s_{};
};
static_assert(Model<CounterModel>);

template <class R>
std::vector<CounterSnapshot> drain(R& r) {
    std::vector<CounterSnapshot> out;
    CounterSnapshot              s;
    while (r.poll(s)) out.push_back(s);
    return out;
}

RunnerConfig cfg(double sps, std::size_t publish_every = 1) {
    RunnerConfig c;
    c.dt                     = 0.25;
    c.clock.steps_per_second = sps;
    c.publish_every          = publish_every;
    return c;
}

}  // namespace

TEST_CASE("RUNNER-01: tick runs the due steps with the configured dt and publishes one snapshot each",
          "[runner][unit]") {
    Runner<CounterModel> r(CounterModel{}, cfg(100.0));
    CHECK(r.tick(1.0) == 100);
    CHECK(r.total_steps() == 100);
    const auto snaps = drain(r);
    REQUIRE(snaps.size() == 100);
    for (std::size_t i = 0; i < snaps.size(); ++i) {
        CHECK(snaps[i].steps == i + 1);
        CHECK(snaps[i].last_dt == 0.25);
    }
}

TEST_CASE("RUNNER-02: clock commands are applied before stepping (Pause / StepOnce / Resume)",
          "[runner][unit]") {
    Runner<CounterModel> r(CounterModel{}, cfg(100.0));
    REQUIRE(r.send(Command::pause()));
    CHECK(r.tick(1.0) == 0);
    CHECK(r.clock().paused());

    REQUIRE(r.send(Command::step_once()));
    CHECK(r.tick(1.0) == 1);
    CHECK(drain(r).size() == 1);

    REQUIRE(r.send(Command::resume()));
    CHECK(r.tick(0.5) == 50);
}

TEST_CASE("RUNNER-03: SetSpeed reaches the clock; SetParam and Reset reach the model", "[runner][unit]") {
    Runner<CounterModel> r(CounterModel{}, cfg(100.0));
    REQUIRE(r.send(Command::set_speed(2.0)));
    CHECK(r.tick(1.0) == 200);

    REQUIRE(r.send(Command::set_param(1, 3.5)));
    REQUIRE(r.send(Command::reset()));
    r.tick(0.01);  // 2 steps
    CHECK(r.model().snapshot().param == 3.5);
    CHECK(r.model().snapshot().resets == 1);
    CHECK(r.model().snapshot().steps == 2);  // Reset は step の前に消化されている
}

TEST_CASE("RUNNER-04: publish_every decimates snapshots", "[runner][unit]") {
    Runner<CounterModel> r(CounterModel{}, cfg(100.0, 10));
    r.tick(1.0);
    const auto snaps = drain(r);
    REQUIRE(snaps.size() == 10);
    for (std::size_t i = 0; i < snaps.size(); ++i) CHECK(snaps[i].steps == 10 * (i + 1));
}

TEST_CASE("RUNNER-05: when the snapshot ring is full the core keeps going and counts drops",
          "[runner][unit]") {
    Runner<CounterModel, 4, 8> r(CounterModel{}, cfg(100.0));
    CHECK(r.tick(1.0) == 100);
    CHECK(r.total_steps() == 100);
    CHECK(r.queued_snapshots() == 4);
    CHECK(r.dropped_snapshots() == 96);
    const auto snaps = drain(r);
    REQUIRE(snaps.size() == 4);
    CHECK(snaps.front().steps == 1);  // 古いものが残り、新しいものが捨てられる
    CHECK(snaps.back().steps == 4);
}

TEST_CASE("RUNNER-06: send fails (returns false) when the command ring is full, never blocks",
          "[runner][unit]") {
    Runner<CounterModel, 4, 2> r(CounterModel{}, cfg(100.0));
    CHECK(r.send(Command::pause()));
    CHECK(r.send(Command::pause()));
    CHECK_FALSE(r.send(Command::pause()));
}

TEST_CASE("RUNNER-07: threaded run produces monotonically increasing snapshots and stops cleanly",
          "[runner][concurrency]") {
    Runner<CounterModel> r(CounterModel{}, cfg(20000.0));
    CHECK_FALSE(r.running());
    r.start();
    CHECK(r.running());

    std::vector<CounterSnapshot> got;
    const auto                   deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (got.size() < 500 && std::chrono::steady_clock::now() < deadline) {
        CounterSnapshot s;
        while (r.poll(s)) got.push_back(s);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(r.send(Command::set_param(1, 9.0)));
    r.stop();
    CHECK_FALSE(r.running());

    REQUIRE(got.size() >= 500);
    for (std::size_t i = 1; i < got.size(); ++i) REQUIRE(got[i].steps == got[i - 1].steps + 1);
    // stop() は join まで行うので、以後 model() を読んでも競合しない
    CHECK(r.model().snapshot().param == 9.0);
    CHECK(r.total_steps() >= got.back().steps);
}

TEST_CASE("RUNNER-08: start is idempotent and the destructor joins a running thread", "[runner][concurrency]") {
    {
        Runner<CounterModel> r(CounterModel{}, cfg(1000.0));
        r.start();
        r.start();
        CHECK(r.running());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }  // ここでハングやクラッシュがなければ合格
    SUCCEED("destructor joined the worker thread");
}

// --------------------------------------------------------------------------- M2: 面チャネル

namespace {

/// 面（グリッド大の状態の代用）。step 数と、それから決まるセル値・チェックサムを持つ。
struct CounterSurface {
    std::uint64_t          steps    = 0;
    std::array<double, 16> cells{};
    double                 checksum = 0.0;
};

/// CounterModel と同じ状態を持ち、surface() だけを足した Model（sizeof(M) は CounterModel と同一）。
class SurfaceCounterModel {
public:
    using Snapshot = CounterSnapshot;
    using Surface  = CounterSurface;

    void     step(double dt) { ++s_.steps; s_.last_dt = dt; }
    Snapshot snapshot() const noexcept { return s_; }
    void     apply(const Command& c) {
        if (c.type == CommandType::SetParam && c.param_id == 1) s_.param = c.value;
        if (c.type == CommandType::Reset) { s_.steps = 0; ++s_.resets; }
    }
    /// 現在の面を out に書く（ヒープなし・例外なし）。
    void surface(Surface& out) const noexcept {
        out.steps  = s_.steps;
        double sum = 0.0;
        for (std::size_t i = 0; i < out.cells.size(); ++i) {
            out.cells[i] = static_cast<double>(s_.steps) + static_cast<double>(i);
            sum += out.cells[i];
        }
        out.checksum = sum;
    }

private:
    Snapshot s_{};
};
static_assert(Model<SurfaceCounterModel>);

/// 期待チェックサム: Σ_{i<16} (steps + i) = 16·steps + 120
double expected_checksum(std::uint64_t steps) {
    return 16.0 * static_cast<double>(steps) + 120.0;
}

/// 空基底最適化（EBO）の確認用。非 SurfaceModel の面チャネルが本当に「サイズ 0 の基底」なら、
/// それに char を 1 つ足しただけの型は 1 バイトに収まる。崩れた場合は RUNNER-09 の
/// STATIC_REQUIRE がコンパイル時に失敗する（実行時のテスト失敗ではなくビルドエラーになる）。
struct EboProbe : detail::SurfaceChannel<CounterModel> {
    char c;
};

}  // namespace

TEST_CASE("RUNNER-09: a SurfaceModel runner publishes the latest surface every surface_every steps; "
          "a plain model grows no channel",
          "[runner][unit]") {
    STATIC_REQUIRE(SurfaceModel<SurfaceCounterModel>);
    STATIC_REQUIRE_FALSE(SurfaceModel<CounterModel>);
    // 非 SurfaceModel の面チャネルは空クラスで、空基底として畳まれる（Runner は 1 バイトも太らない）
    STATIC_REQUIRE(std::is_empty_v<detail::SurfaceChannel<CounterModel>>);
    STATIC_REQUIRE(sizeof(EboProbe) == 1);
    // 面チャネルは SurfaceModel のときだけ生える（両 Model は同じ状態なので sizeof(M) は等しい）
    STATIC_REQUIRE(sizeof(CounterModel) == sizeof(SurfaceCounterModel));
    STATIC_REQUIRE(sizeof(Runner<SurfaceCounterModel>) > sizeof(Runner<CounterModel>));

    SECTION("surface_every = 1: poll_surface returns only the newest surface, older ones are dropped") {
        Runner<SurfaceCounterModel> r(SurfaceCounterModel{}, cfg(100.0));
        CounterSurface              surf{};
        CHECK_FALSE(r.poll_surface(surf));  // まだ何も出ていない
        CHECK(r.surfaces_published() == 0);

        CHECK(r.tick(0.1) == 10);
        CHECK(r.total_steps() == 10);
        CHECK(r.surfaces_published() == 10);

        REQUIRE(r.poll_surface(surf));
        CHECK(surf.steps == 10);  // 最新 1 枚だけ。途中の 9 枚は捨てられる
        CHECK(surf.checksum == expected_checksum(10));
        CHECK_FALSE(r.poll_surface(surf));  // 新しいものが無ければ false
        CHECK(surf.steps == 10);

        // Snapshot リングは従来どおり全ステップ分出る（面チャネルは干渉しない）
        CHECK(drain(r).size() == 10);
    }

    SECTION("surface_every = 3 publishes only on multiples of 3") {
        RunnerConfig c   = cfg(100.0);
        c.surface_every  = 3;
        Runner<SurfaceCounterModel> r(SurfaceCounterModel{}, c);

        CHECK(r.tick(0.1) == 10);  // 3, 6, 9 の 3 回だけ publish
        CHECK(r.surfaces_published() == 3);

        CounterSurface surf{};
        REQUIRE(r.poll_surface(surf));
        CHECK(surf.steps == 9);
        CHECK(surf.checksum == expected_checksum(9));
        CHECK_FALSE(r.poll_surface(surf));

        CHECK(r.tick(0.1) == 10);  // 12, 15, 18 → 累計 6
        CHECK(r.surfaces_published() == 6);
        REQUIRE(r.poll_surface(surf));
        CHECK(surf.steps == 18);
    }

    SECTION("surface_every = 0 is clamped to 1, like publish_every") {
        RunnerConfig c  = cfg(100.0);
        c.surface_every = 0;
        Runner<SurfaceCounterModel> r(SurfaceCounterModel{}, c);
        CHECK(r.config().surface_every == 1);
        r.tick(0.05);
        CHECK(r.surfaces_published() == 5);
    }
}

// --------------------------------------------------------------------------- M3: 一時停止中の反映

// CounterModel / SurfaceCounterModel には `seq` フィールドが無いので、`steps` が seq の役を担う
// （step で +1、Reset で 0 に戻る）。
TEST_CASE("RUNNER-10: a model command applied on a tick that runs no steps republishes the snapshot",
          "[runner][unit]") {
    RunnerConfig paused       = cfg(100.0);
    paused.clock.start_paused = true;

    SECTION("SetParam while paused publishes exactly one snapshot, with the seq unchanged") {
        Runner<CounterModel> r(CounterModel{}, paused);
        REQUIRE(r.send(Command::step_once()));
        REQUIRE(r.send(Command::step_once()));
        REQUIRE(r.send(Command::step_once()));
        REQUIRE(r.tick(1.0) == 3);
        REQUIRE(drain(r).size() == 3);

        REQUIRE(r.send(Command::set_param(1, 2.5)));
        CHECK(r.tick(1.0) == 0);  // 一時停止中: ステップは走らない
        const auto snaps = drain(r);
        REQUIRE(snaps.size() == 1);  // それでも新しい状態が 1 枚だけ出る
        CHECK(snaps[0].param == 2.5);
        CHECK(snaps[0].steps == 3);  // seq は据え置き（巻き戻らない）
    }

    SECTION("Reset while paused publishes one snapshot with seq 0") {
        Runner<CounterModel> r(CounterModel{}, paused);
        REQUIRE(r.send(Command::step_once()));
        REQUIRE(r.tick(1.0) == 1);
        REQUIRE(drain(r).size() == 1);

        REQUIRE(r.send(Command::reset()));
        CHECK(r.tick(1.0) == 0);
        const auto snaps = drain(r);
        REQUIRE(snaps.size() == 1);
        CHECK(snaps[0].steps == 0);   // Reset は seq を 0 に戻す
        CHECK(snaps[0].resets == 1);
    }

    SECTION("Reset and SetParam drained in the same paused tick publish exactly one snapshot") {
        Runner<CounterModel> r(CounterModel{}, paused);
        REQUIRE(r.send(Command::step_once()));
        REQUIRE(r.tick(1.0) == 1);
        REQUIRE(drain(r).size() == 1);

        REQUIRE(r.send(Command::reset()));
        REQUIRE(r.send(Command::set_param(1, 7.0)));
        CHECK(r.tick(1.0) == 0);
        const auto snaps = drain(r);
        REQUIRE(snaps.size() == 1);   // 2 つのモデルコマンドでも再送は 1 回
        CHECK(snaps[0].steps == 0);
        CHECK(snaps[0].param == 7.0);
    }

    SECTION("a tick with only clock commands publishes nothing") {
        Runner<CounterModel> r(CounterModel{}, paused);
        REQUIRE(r.send(Command::set_speed(2.0)));
        REQUIRE(r.send(Command::pause()));
        CHECK(r.tick(1.0) == 0);
        CHECK(drain(r).empty());
    }

    SECTION("a tick that also runs steps publishes only the per-step snapshots") {
        Runner<CounterModel> r(CounterModel{}, paused);
        REQUIRE(r.send(Command::set_param(1, 7.0)));
        REQUIRE(r.send(Command::step_once()));
        CHECK(r.tick(1.0) == 1);
        const auto snaps = drain(r);
        REQUIRE(snaps.size() == 1);  // 追加の 1 枚は出さない
        CHECK(snaps[0].steps == 1);
        CHECK(snaps[0].param == 7.0);
    }

    SECTION("a SurfaceModel republishes the surface too, whatever the surface_every phase") {
        // ここは走行中に 2 ステップ進めてから止める。一時停止中の StepOnce で進めると R12
        //（手動ステップの最後の 1 歩は位相に関わらず publish）が先に面を出してしまい、
        // 「位相外なのでまだ面が無い」という前提が作れない。
        RunnerConfig c  = cfg(2.0);  // 壁 1 秒で 2 ステップ
        c.surface_every = 3;
        Runner<SurfaceCounterModel> r(SurfaceCounterModel{}, c);
        REQUIRE(r.tick(1.0) == 2);
        REQUIRE(drain(r).size() == 2);       // 2 ステップぶんの Snapshot は取り除いておく
        CHECK(r.surfaces_published() == 0);  // まだ 3 の倍数に達していない
        CounterSurface surf{};
        CHECK_FALSE(r.poll_surface(surf));

        REQUIRE(r.send(Command::pause()));
        REQUIRE(r.send(Command::set_param(1, 1.0)));
        CHECK(r.tick(1.0) == 0);
        CHECK(r.surfaces_published() == 1);
        REQUIRE(r.poll_surface(surf));
        CHECK(surf.steps == 2);
        CHECK(surf.checksum == expected_checksum(2));
        CHECK(drain(r).size() == 1);
    }
}

// --------------------------------------------------------------------------- M4: 手動ステップの見え方

// R12: 一時停止中の StepOnce（pending ステップ）で進んだ tick は、最後のステップを publish_every /
// surface_every の位相に関わらず publish する。走行中の間引きは変えない。
TEST_CASE("RUNNER-12: steps taken from a paused StepOnce publish whatever the publish_every phase",
          "[runner][unit]") {
    RunnerConfig paused       = cfg(100.0, 4);  // publish_every = 4
    paused.clock.start_paused = true;

    SECTION("five StepOnce in five separate ticks publish five snapshots with seq 1..5") {
        Runner<CounterModel> r(CounterModel{}, paused);
        for (int i = 0; i < 5; ++i) {
            REQUIRE(r.send(Command::step_once()));
            CHECK(r.tick(1.0) == 1);
        }
        CHECK(r.total_steps() == 5);
        const auto snaps = drain(r);
        REQUIRE(snaps.size() == 5);
        for (std::size_t i = 0; i < snaps.size(); ++i) CHECK(snaps[i].steps == i + 1);
    }

    SECTION("five StepOnce drained in one tick publish on the phase (4) and on the last step (5)") {
        Runner<CounterModel> r(CounterModel{}, paused);
        for (int i = 0; i < 5; ++i) REQUIRE(r.send(Command::step_once()));
        CHECK(r.tick(1.0) == 5);
        const auto snaps = drain(r);
        REQUIRE(snaps.size() == 2);  // 4 は位相ぶん、5 は最後のステップぶん（重複は出さない）
        CHECK(snaps[0].steps == 4);
        CHECK(snaps[1].steps == 5);
    }

    SECTION("a last step that lands on the phase is published exactly once") {
        Runner<CounterModel> r(CounterModel{}, paused);
        for (int i = 0; i < 4; ++i) REQUIRE(r.send(Command::step_once()));
        CHECK(r.tick(1.0) == 4);
        const auto snaps = drain(r);
        REQUIRE(snaps.size() == 1);
        CHECK(snaps[0].steps == 4);
    }

    SECTION("a running clock keeps the publish_every decimation") {
        Runner<CounterModel> r(CounterModel{}, cfg(100.0, 4));
        CHECK(r.tick(0.1) == 10);  // 4, 8 の 2 枚だけ。10 は位相外なので出さない
        const auto snaps = drain(r);
        REQUIRE(snaps.size() == 2);
        CHECK(snaps[0].steps == 4);
        CHECK(snaps[1].steps == 8);
    }

    SECTION("a paused StepOnce republishes the surface off-phase too") {
        RunnerConfig c  = paused;
        c.surface_every = 4;
        Runner<SurfaceCounterModel> r(SurfaceCounterModel{}, c);

        REQUIRE(r.send(Command::step_once()));
        CHECK(r.tick(1.0) == 1);
        CHECK(r.surfaces_published() == 1);  // 位相（4 の倍数）に達していなくても出す
        CounterSurface surf{};
        REQUIRE(r.poll_surface(surf));
        CHECK(surf.steps == 1);
        CHECK(surf.checksum == expected_checksum(1));
        CHECK(drain(r).size() == 1);

        REQUIRE(r.send(Command::resume()));
        CHECK(r.tick(0.1) == 10);            // steps 2..11 → 面は 4, 8 の 2 回だけ（走行中は間引く）
        CHECK(r.surfaces_published() == 3);
        REQUIRE(r.poll_surface(surf));
        CHECK(surf.steps == 8);
        CHECK(drain(r).size() == 2);
    }
}

// --------------------------------------------------------------------------- M5: step 時間の計測

namespace {

/// 1 step が確実に測れる長さ（~50 us）かかる Model。計測の有無を実測で見分けるために使う。
class SlowModel {
public:
    using Snapshot = CounterSnapshot;
    void     step(double dt) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        ++s_.steps;
        s_.last_dt = dt;
    }
    Snapshot snapshot() const noexcept { return s_; }
    void     apply(const Command&) {}

private:
    Snapshot s_{};
};
static_assert(Model<SlowModel>);

std::uint64_t bin_sum(const StepHistogramSample& s) {
    std::uint64_t sum = 0;
    for (const std::uint64_t b : s.bins) sum += b;
    return sum;
}

}  // namespace

// RUNNER-11: step の所要時間を対数ビンのヒストグラム（atomic, relaxed）に記録する。measure_every = k は
// 「k ステップを 1 サンプル（= elapsed/k ns per step）にまとめる」。k = 0 なら steady_clock を一切読まない。
TEST_CASE("RUNNER-11: tick records one step-time sample per measure_every steps in a log histogram",
          "[runner][unit]") {
    STATIC_REQUIRE(kStepHistogramBins == 32);
    // 32 ビン + count/total/max = 280 B。キャッシュラインを跨がないよう alignas してある。
    STATIC_REQUIRE(alignof(StepHistogram) == kCacheLineSize);
    STATIC_REQUIRE(sizeof(StepHistogram) % kCacheLineSize == 0);
    STATIC_REQUIRE(sizeof(StepHistogram) >= 32 * sizeof(std::uint64_t) + 3 * sizeof(std::uint64_t));
    STATIC_REQUIRE(std::is_trivially_copyable_v<StepHistogramSample>);

    // 既定は 16（軽い step では clock 読みの下駄が 25/k ns になるため）。以下の SECTION は
    // 既定値に依存しないよう measure_every を毎回明示する。
    CHECK(RunnerConfig{}.measure_every == 16);

    SECTION("measure_every = 1 records one sample per step") {
        RunnerConfig c  = cfg(100.0);
        c.measure_every = 1;
        Runner<CounterModel> r(CounterModel{}, c);
        CHECK(r.config().measure_every == 1);
        CHECK(r.step_histogram().snapshot().count == 0);

        CHECK(r.tick(1.0) == 100);
        const auto s = r.step_histogram().snapshot();
        CHECK(s.count == 100);
        CHECK(bin_sum(s) == s.count);
        CHECK(s.max_ns >= s.total_ns / s.count);  // 最大 >= 平均
    }

    SECTION("a 50 us step is timed: the totals are consistent and the sample lands above bin 0") {
        RunnerConfig c  = cfg(100.0);
        c.measure_every = 1;
        Runner<SlowModel> r(SlowModel{}, c);
        CHECK(r.tick(0.1) == 10);
        const auto s = r.step_histogram().snapshot();
        REQUIRE(s.count == 10);
        CHECK(bin_sum(s) == 10);
        CHECK(s.total_ns > 0);
        CHECK(s.max_ns >= s.total_ns / s.count);
        CHECK(s.max_ns >= 10'000);  // 50 us の sleep が 10 us 未満で返ることはない
        CHECK(s.bins[0] == 0);      // 100 ns 未満のビンには入らない
        CHECK(StepHistogram::bin_of(s.max_ns) >= StepHistogram::bin_of(10'000));
    }

    SECTION("measure_every = 4: 40 steps give 10 samples") {
        RunnerConfig c  = cfg(100.0);
        c.measure_every = 4;
        Runner<CounterModel> r(CounterModel{}, c);
        CHECK(r.tick(0.4) == 40);
        const auto s = r.step_histogram().snapshot();
        CHECK(s.count == 10);
        CHECK(bin_sum(s) == 10);
    }

    SECTION("a group that the tick cannot fill is still recorded, divided by its real step count") {
        RunnerConfig c  = cfg(100.0);
        c.measure_every = 4;
        Runner<CounterModel> r(CounterModel{}, c);
        CHECK(r.tick(0.06) == 6);  // 4 + 端数 2
        CHECK(r.step_histogram().snapshot().count == 2);
        CHECK(r.tick(0.01) == 1);  // 端数だけの tick でも 1 サンプル（グループは tick を跨がない）
        CHECK(r.step_histogram().snapshot().count == 3);
    }

    SECTION("measure_every = 0 leaves the histogram untouched and the stepping unchanged") {
        RunnerConfig c  = cfg(100.0);
        c.measure_every = 0;
        Runner<SlowModel> r(SlowModel{}, c);
        CHECK(r.tick(0.1) == 10);
        const auto s = r.step_histogram().snapshot();
        CHECK(s.count == 0);
        CHECK(s.total_ns == 0);
        CHECK(s.max_ns == 0);
        CHECK(bin_sum(s) == 0);
        // 計測を切っても step / publish の振る舞いは同じ
        CHECK(r.total_steps() == 10);
        CHECK(drain(r).size() == 10);
    }

    SECTION("the histogram is telemetry: Reset does not clear it, reset_step_histogram does") {
        RunnerConfig c  = cfg(100.0);
        c.measure_every = 1;
        Runner<CounterModel> r(CounterModel{}, c);
        CHECK(r.tick(0.1) == 10);
        REQUIRE(r.step_histogram().snapshot().count == 10);

        REQUIRE(r.send(Command::reset()));
        CHECK(r.tick(0.05) == 5);
        CHECK(r.model().snapshot().steps == 5);              // モデルは巻き戻る
        CHECK(r.step_histogram().snapshot().count == 15);    // 計測は積み上がったまま

        r.reset_step_histogram();
        const auto s = r.step_histogram().snapshot();
        CHECK(s.count == 0);
        CHECK(s.total_ns == 0);
        CHECK(s.max_ns == 0);
        CHECK(bin_sum(s) == 0);
    }

    SECTION("bin_of and bin_lower_ns are inverse over a log sweep of 1 ns .. 1e10 ns") {
        CHECK(StepHistogram::bin_lower_ns(0) == 100);  // ビン 0 の名目下端は 100 ns
        // 上端は開いている（1 s 以上は全部最上位ビン）ので、上限の番号は「無限大」を返す
        CHECK(StepHistogram::bin_lower_ns(kStepHistogramBins) == std::numeric_limits<std::uint64_t>::max());

        for (std::size_t i = 0; i < kStepHistogramBins; ++i) {
            // 表は 100 ns から 1 ビンあたり 10^(7/32) ≈ 1.655 倍（100 ns → 1 s を 32 ビン）
            const double want = 100.0 * std::pow(10.0, 7.0 * static_cast<double>(i) / 32.0);
            CHECK(StepHistogram::bin_lower_ns(i) == static_cast<std::uint64_t>(std::llround(want)));
            if (i > 0) CHECK(StepHistogram::bin_lower_ns(i) > StepHistogram::bin_lower_ns(i - 1));
            CHECK(StepHistogram::bin_of(StepHistogram::bin_lower_ns(i)) == i);
            if (i > 1) CHECK(StepHistogram::bin_of(StepHistogram::bin_lower_ns(i) - 1) == i - 1);
        }

        for (int e = 0; e <= 100; ++e) {  // 10^0 .. 10^10 を 0.1 桁刻みで
            const auto ns = static_cast<std::uint64_t>(std::llround(std::pow(10.0, 0.1 * e)));
            const std::size_t b = StepHistogram::bin_of(ns);
            REQUIRE(b < kStepHistogramBins);
            // ビン 0 はアンダーフロー（100 ns 未満も飲み込む）ので、下側の関係はそこだけ成り立たない
            if (ns >= StepHistogram::bin_lower_ns(0)) CHECK(StepHistogram::bin_lower_ns(b) <= ns);
            CHECK(ns < StepHistogram::bin_lower_ns(b + 1));
        }

        CHECK(StepHistogram::bin_of(0) == 0);
        CHECK(StepHistogram::bin_of(99) == 0);
        CHECK(StepHistogram::bin_of(1'000'000'000ULL) == kStepHistogramBins - 1);      // 1 s
        CHECK(StepHistogram::bin_of(1'000'000'000'000ULL) == kStepHistogramBins - 1);  // 1e12 も飽和
    }

    SECTION("the compute thread fills the histogram while this thread reads it (TSan)") {
        // 既定の measure_every のまま回す。RUNNER-07 と同じく「条件が満たされるまで待つ」形に
        // して、遅いマシンや TSan 下でも時間で落ちないようにする。
        Runner<CounterModel> r(CounterModel{}, cfg(20000.0));
        r.start();
        StepHistogramSample mid{};
        const auto          deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (mid.count == 0 && std::chrono::steady_clock::now() < deadline) {
            mid = r.step_histogram().snapshot();  // 走行中の読みは relaxed（だいたいの姿）
            CounterSnapshot s;
            while (r.poll(s)) {}
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        r.stop();
        REQUIRE(mid.count > 0);
        const auto after = r.step_histogram().snapshot();
        CHECK(after.count >= mid.count);       // 単調増加
        CHECK(bin_sum(after) == after.count);  // 停止後なら厳密に一致する
        CHECK(after.max_ns >= after.total_ns / after.count);
    }
}
