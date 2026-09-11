// HIST-xx — viz/history.hpp（描画側の循環履歴）の仕様テスト
#include <catch2/catch_test_macros.hpp>

#include "quantviz/viz/history.hpp"

using quantviz::viz::History;

TEST_CASE("HIST-01: a new history is empty with offset 0", "[history][unit]") {
    History<4> h;
    CHECK(h.empty());
    CHECK(h.count() == 0);
    CHECK(h.offset() == 0);
    CHECK(h.latest_x() == 0.0);
}

TEST_CASE("HIST-02: before wrapping, offset stays 0 and data is in insertion order", "[history][unit]") {
    History<4> h;
    h.push(1.0, 10.0);
    h.push(2.0, 20.0);
    h.push(3.0, 30.0);
    CHECK(h.count() == 3);
    CHECK(h.offset() == 0);
    CHECK(h.xs()[0] == 1.0);
    CHECK(h.ys()[2] == 30.0);
    CHECK(h.latest_x() == 3.0);
    CHECK(h.latest_y() == 30.0);
    CHECK(h.oldest_x() == 1.0);
}

TEST_CASE("HIST-03: once full, offset points at the oldest sample (ImPlot circular-buffer convention)",
          "[history][unit]") {
    History<4> h;
    for (int i = 1; i <= 6; ++i) h.push(i, i * 10.0);
    CHECK(h.count() == 4);
    CHECK(h.offset() == 2);         // 5,6 が index 0,1 を上書き → 最古の 3 は index 2
    CHECK(h.xs()[h.offset()] == 3.0);
    CHECK(h.oldest_x() == 3.0);
    CHECK(h.latest_x() == 6.0);
    CHECK(h.latest_y() == 60.0);
}

TEST_CASE("HIST-04: clear() empties without reallocating; pushes restart from index 0", "[history][unit]") {
    History<4> h;
    for (int i = 0; i < 10; ++i) h.push(i, i);
    h.clear();
    CHECK(h.empty());
    CHECK(h.offset() == 0);
    h.push(42.0, 1.0);
    CHECK(h.xs()[0] == 42.0);
    CHECK(h.count() == 1);
}
