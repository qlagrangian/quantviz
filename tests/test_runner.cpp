// RUNNER-xx — bridge/runner.hpp の仕様テスト。
// 時間に依存しない tick() を主に使い、スレッド起動は 1 ケースだけ統合テストとして持つ。
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
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
