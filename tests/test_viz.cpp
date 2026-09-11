// VIZ-xx — viz/scene_registry.hpp, viz/clock_controls.hpp（vizcore: GUI 非依存部）の仕様テスト。
//
// ImGui に依存する描画そのものはテストしない（手動チェックリスト）。ここで固定するのは
//   * シーンの登録順と選択のライフサイクル（生成 → start / 前シーンは stop してから破棄）
//   * 共通 Control の状態遷移と、そこから生成される Command
// の 2 点だけ。VIZ-01〜03 は偽シーンだけを使う純ロジック、VIZ-04 だけが本物の Runner
// （計算スレッド）を起動するので `[concurrency]`（TSan 対象）に分けてある。
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/runner.hpp"
#include "quantviz/viz/clock_controls.hpp"
#include "quantviz/viz/scene_registry.hpp"

using quantviz::bridge::Command;
using quantviz::bridge::CommandType;
using quantviz::viz::ClockControlState;
using quantviz::viz::RunnerScene;
using quantviz::viz::Scene;
using quantviz::viz::SceneRegistry;

namespace {

// ------------------------------------------------------------------ 偽シーン（観測用）
/// シーンは select() で破棄されるため、観測は外部の log に書き出す。
struct SceneLog {
    int  starts                 = 0;
    int  stops                  = 0;
    int  draws                  = 0;
    int  destroys               = 0;
    bool running                = false;
    bool running_at_destruction = true;  ///< 破棄時点で stop 済みか（false が期待値）
};

class FakeScene final : public Scene {
public:
    explicit FakeScene(SceneLog& log) : log_(&log) {}
    ~FakeScene() override {
        ++log_->destroys;
        log_->running_at_destruction = log_->running;
    }
    FakeScene(const FakeScene&)            = delete;
    FakeScene& operator=(const FakeScene&) = delete;

    void start() override {
        ++log_->starts;
        log_->running = true;
    }
    void stop() override {
        ++log_->stops;
        log_->running = false;
    }
    bool running() const override { return log_->running; }
    void draw() override { ++log_->draws; }

private:
    SceneLog* log_;
};

// ------------------------------------------------------------------ 本物の Runner を載せた最小シーン
struct CounterSnapshot {
    std::uint64_t steps = 0;
};

class CounterModel {
public:
    using Snapshot = CounterSnapshot;
    void     step(double) { ++s_.steps; }
    Snapshot snapshot() const noexcept { return s_; }
    void     apply(const Command&) {}

private:
    Snapshot s_{};
};
static_assert(quantviz::bridge::Model<CounterModel>);

/// パネル契約は `void draw(Runner&)` だけ。ここでは呼ばれた回数と、渡された Runner を検査する。
struct CountingPanel {
    int* draws = nullptr;
    template <class R>
    void draw(R& runner) {
        CHECK(runner.queued_snapshots() == 0);  // 一時停止で開始しているので Snapshot は出ていない
        if (draws != nullptr) ++*draws;
    }
};

using CounterScene = RunnerScene<CounterModel, CountingPanel, 64, 16>;

quantviz::bridge::RunnerConfig counter_cfg() {
    quantviz::bridge::RunnerConfig c;
    c.dt                     = 0.25;
    c.clock.steps_per_second = 10.0;
    c.clock.start_paused     = true;  // スレッドを空回しさせない
    return c;
}

/// 本物の Runner を内側に持ち、「破棄の瞬間に計算スレッドが join 済みか」を記録するプローブ。
/// メンバ inner_ の破棄はこのデストラクタ本体の**後**なので、ここで running() を読めば
/// 「SceneRegistry が stop() してから破棄したか」がそのまま観測できる（VIZ-04）。
class ProbeScene final : public Scene {
public:
    ProbeScene(SceneLog& log, int* draws)
        : log_(&log), inner_(CounterModel{}, counter_cfg(), std::in_place, CountingPanel{draws}) {}
    ~ProbeScene() override {
        ++log_->destroys;
        log_->running_at_destruction = inner_.running();
    }

    void start() override {
        ++log_->starts;
        inner_.start();
    }
    void stop() override {
        ++log_->stops;
        inner_.stop();
    }
    bool running() const override { return inner_.running(); }
    void draw() override {
        ++log_->draws;
        inner_.draw();
    }

private:
    SceneLog*    log_;
    CounterScene inner_;
};

}  // namespace

// ====================================================================== VIZ-01
TEST_CASE("VIZ-01: the registry lists every registered scene name in registration order", "[viz][unit]") {
    SceneLog      a, b, c;
    SceneRegistry reg;
    CHECK(reg.size() == 0);
    CHECK(reg.names().empty());

    reg.add("Streaming", [&a] { return std::unique_ptr<Scene>(std::make_unique<FakeScene>(a)); });
    reg.add("Greeks", [&b] { return std::unique_ptr<Scene>(std::make_unique<FakeScene>(b)); });
    reg.add("GARCH", [&c] { return std::unique_ptr<Scene>(std::make_unique<FakeScene>(c)); });

    REQUIRE(reg.size() == 3);
    const std::vector<std::string>& names = reg.names();
    REQUIRE(names.size() == 3);
    CHECK(names[0] == "Streaming");
    CHECK(names[1] == "Greeks");
    CHECK(names[2] == "GARCH");
}

// ====================================================================== VIZ-02
TEST_CASE("VIZ-02: selecting a scene builds and starts it, stopping the previous runner before destroying it",
          "[viz][unit]") {
    SceneLog      a, b;
    SceneRegistry reg;
    reg.add("A", [&a] { return std::unique_ptr<Scene>(std::make_unique<FakeScene>(a)); });
    reg.add("B", [&b] { return std::unique_ptr<Scene>(std::make_unique<FakeScene>(b)); });

    CHECK(reg.current() == nullptr);
    CHECK(reg.current_index() == SceneRegistry::kNone);
    CHECK(std::as_const(reg).current() == nullptr);  // const 版も同じものを指す

    SECTION("select creates the scene and starts it") {
        CHECK(reg.select(0));  // 選択できたら true
        REQUIRE(reg.current() != nullptr);
        CHECK(std::as_const(reg).current() == reg.current());
        CHECK(reg.current_index() == 0);
        CHECK(a.starts == 1);
        CHECK(a.destroys == 0);
        CHECK(reg.current()->running());

        reg.current()->draw();
        CHECK(a.draws == 1);
    }

    SECTION("switching scenes stops the previous runner before destroying it") {
        reg.select(0);
        reg.select(1);

        CHECK(a.stops == 1);
        CHECK(a.destroys == 1);
        CHECK_FALSE(a.running);
        CHECK_FALSE(a.running_at_destruction);  // stop() は破棄より前

        CHECK(b.starts == 1);
        CHECK(b.destroys == 0);
        CHECK(reg.current_index() == 1);
        REQUIRE(reg.current() != nullptr);
        CHECK(reg.current()->running());
    }

    SECTION("re-selecting the same index rebuilds the scene (equivalent to a reset)") {
        reg.select(0);
        reg.select(0);

        CHECK(a.starts == 2);
        CHECK(a.stops == 1);
        CHECK(a.destroys == 1);
        CHECK(reg.current_index() == 0);
        REQUIRE(reg.current() != nullptr);
        CHECK(reg.current()->running());
    }

    SECTION("an out-of-range index is ignored and leaves the current scene alone") {
        REQUIRE(reg.select(0));
        CHECK_FALSE(reg.select(2));
        CHECK_FALSE(reg.select(SceneRegistry::kNone));

        CHECK(a.starts == 1);
        CHECK(a.stops == 0);
        CHECK(a.destroys == 0);
        CHECK(reg.current_index() == 0);
        REQUIRE(reg.current() != nullptr);
        CHECK(reg.current()->running());
    }
}

// ====================================================================== VIZ-04
TEST_CASE("VIZ-04: selecting a scene backed by a real Runner joins the previous compute thread "
          "(running() false)",
          "[viz][concurrency]") {
    SceneLog probe;
    int      draws        = 0;
    int      unused_draws = 0;

    SceneRegistry reg;
    reg.add("Probe",
            [&probe, &draws] { return std::unique_ptr<Scene>(std::make_unique<ProbeScene>(probe, &draws)); });
    reg.add("Plain", [&unused_draws] {
        // by-value のコンストラクタ（計画の型契約）も 1 か所で通しておく。
        return std::unique_ptr<Scene>(
            std::make_unique<CounterScene>(CounterModel{}, counter_cfg(), CountingPanel{&unused_draws}));
    });

    SECTION("start / draw / stop are delegated to the Runner and the Panel") {
        REQUIRE(reg.select(0));
        REQUIRE(reg.current() != nullptr);
        CHECK(reg.current()->running());  // 計算スレッドが起動している
        reg.current()->draw();
        CHECK(draws == 1);

        reg.current()->stop();  // join
        CHECK_FALSE(reg.current()->running());
    }

    SECTION("switching away joins the previous compute thread before the scene is destroyed") {
        REQUIRE(reg.select(0));
        REQUIRE(reg.current()->running());

        REQUIRE(reg.select(1));
        CHECK(probe.stops == 1);
        CHECK(probe.destroys == 1);
        CHECK_FALSE(probe.running_at_destruction);  // 破棄時点で join 済み
        CHECK(reg.current()->running());
    }

    SECTION("a registry holding a running scene tears it down cleanly") {
        {
            SceneRegistry scoped;
            scoped.add("Probe", [&probe, &draws] {
                return std::unique_ptr<Scene>(std::make_unique<ProbeScene>(probe, &draws));
            });
            REQUIRE(scoped.select(0));
            REQUIRE(scoped.current()->running());
        }  // stop() を呼ばずにスコープを抜けても RunnerScene のデストラクタが join する
        CHECK(probe.destroys == 1);
    }
}

// ====================================================================== VIZ-03
TEST_CASE("VIZ-03: the shared clock control state is a pure function producing the matching Command",
          "[viz][unit]") {
    ClockControlState st;
    CHECK(st.speed == 1.0f);
    CHECK_FALSE(st.paused);
    CHECK_FALSE(quantviz::viz::step_allowed(st));  // 走行中は Step 不可

    SECTION("toggle_pause flips paused and alternates Pause / Resume") {
        const Command pause = quantviz::viz::toggle_pause(st);
        CHECK(pause.type == CommandType::Pause);
        CHECK(st.paused);
        CHECK(quantviz::viz::step_allowed(st));  // 一時停止中だけ Step 可

        const Command resume = quantviz::viz::toggle_pause(st);
        CHECK(resume.type == CommandType::Resume);
        CHECK_FALSE(st.paused);
        CHECK_FALSE(quantviz::viz::step_allowed(st));
    }

    SECTION("set_speed stores the multiplier and emits SetSpeed with it") {
        const Command c = quantviz::viz::set_speed(st, 2.5f);
        CHECK(c.type == CommandType::SetSpeed);
        CHECK(c.value == 2.5);  // 2.5 は float/double で厳密に一致する
        CHECK(st.speed == 2.5f);

        const Command slow = quantviz::viz::set_speed(st, 0.25f);
        CHECK(slow.value == 0.25);
        CHECK(st.speed == 0.25f);
        CHECK_FALSE(st.paused);  // speed の変更は pause 状態に影響しない
    }

    SECTION("step_once and reset are stateless command factories") {
        const Command s = quantviz::viz::step_once();
        CHECK(s.type == CommandType::StepOnce);

        const Command r = quantviz::viz::reset();
        CHECK(r.type == CommandType::Reset);
        CHECK(r.seed == 0);  // 0 = 現在の seed で再生
    }
}
