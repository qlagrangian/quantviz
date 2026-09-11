// CLOCK-xx — bridge/sim_clock.hpp の仕様テスト
#include <catch2/catch_test_macros.hpp>

#include <limits>

#include "quantviz/bridge/sim_clock.hpp"

using quantviz::bridge::SimClock;

namespace {
SimClock make(double sps, double speed = 1.0, std::size_t cap = 10000, bool paused = false) {
    return SimClock{SimClock::Config{sps, speed, cap, paused}};
}
}  // namespace

TEST_CASE("CLOCK-01: one wall second at 1000 steps/s yields exactly 1000 steps", "[clock][unit]") {
    auto c = make(1000.0);
    CHECK(c.due_steps(1.0) == 1000);
}

TEST_CASE("CLOCK-02: fractional steps accumulate across calls instead of being lost", "[clock][unit]") {
    auto c = make(1000.0);
    CHECK(c.due_steps(0.0005) == 0);  // 0.5 step → まだ出さない
    CHECK(c.due_steps(0.0005) == 1);  // 合計 1.0 step
    CHECK(c.due_steps(0.5) == 500);
    CHECK(c.due_steps(0.5) == 500);
}

TEST_CASE("CLOCK-03: while paused no steps are due and wall time is not banked", "[clock][unit]") {
    auto c = make(1000.0);
    c.pause();
    CHECK(c.paused());
    CHECK(c.due_steps(10.0) == 0);
    c.resume();
    CHECK_FALSE(c.paused());
    CHECK(c.due_steps(0.0) == 0);     // 再開直後のバースト無し
    CHECK(c.due_steps(0.001) == 1);   // 通常運転
}

TEST_CASE("CLOCK-04: speed multiplies the step rate; speed 0 stops time", "[clock][unit]") {
    auto c = make(1000.0, 2.0);
    CHECK(c.due_steps(1.0) == 2000);
    c.set_speed(0.5);
    CHECK(c.due_steps(1.0) == 500);
    c.set_speed(0.0);
    CHECK(c.due_steps(1.0) == 0);
}

TEST_CASE("CLOCK-05: request_step advances exactly one step even while paused", "[clock][unit]") {
    auto c = make(1000.0);
    c.pause();
    c.request_step();
    CHECK(c.pending_steps() == 1);
    CHECK(c.due_steps(5.0) == 1);
    CHECK(c.due_steps(5.0) == 0);
    c.request_step();
    c.request_step();
    c.request_step();
    CHECK(c.due_steps(0.0) == 3);  // 複数リクエストは加算
}

TEST_CASE("CLOCK-06: request_step while running adds to the wall-clock steps", "[clock][unit]") {
    auto c = make(1000.0);
    c.request_step();
    CHECK(c.due_steps(0.001) == 2);
}

TEST_CASE("CLOCK-07: max_steps_per_tick caps a tick and discards the excess", "[clock][unit]") {
    auto c = make(1000.0, 1.0, 100);
    CHECK(c.due_steps(10.0) == 100);  // 10000 due → 100
    CHECK(c.due_steps(0.0) == 0);     // 追いつこうとしない（超過分は捨てた）
}

TEST_CASE("CLOCK-08: negative and NaN elapsed are treated as zero; bad speed clamps to zero", "[clock][unit]") {
    auto c = make(1000.0);
    CHECK(c.due_steps(-1.0) == 0);
    CHECK(c.due_steps(std::numeric_limits<double>::quiet_NaN()) == 0);
    c.set_speed(-3.0);
    CHECK(c.speed() == 0.0);
    c.set_speed(std::numeric_limits<double>::quiet_NaN());
    CHECK(c.speed() == 0.0);
}

TEST_CASE("CLOCK-09: start_paused config begins in the paused state", "[clock][unit]") {
    auto c = make(1000.0, 1.0, 10000, true);
    CHECK(c.paused());
    CHECK(c.due_steps(1.0) == 0);
}
