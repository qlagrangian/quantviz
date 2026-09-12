// HEAT-xx -- viz/history2d.hpp (2D circular history for the price x time heatmap)
#include <array>
#include <cstddef>

#include <catch2/catch_test_macros.hpp>

#include "quantviz/viz/history2d.hpp"

using quantviz::viz::History2D;

TEST_CASE("HEAT-01: dimensions are fixed and count saturates at Cols as columns wrap",
          "[heat][unit]") {
    History2D<4, 3> h;  // Rows=4, Cols=3
    CHECK(h.count() == 0);
    CHECK(h.ordered().empty());

    std::array<float, 4> col0{1.0f, 2.0f, 3.0f, 4.0f};
    h.push_column(col0);
    CHECK(h.count() == 1);

    std::array<float, 4> col1{5.0f, 6.0f, 7.0f, 8.0f};
    h.push_column(col1);
    CHECK(h.count() == 2);

    std::array<float, 4> col2{9.0f, 10.0f, 11.0f, 12.0f};
    h.push_column(col2);
    CHECK(h.count() == 3);  // == Cols, buffer now full

    std::array<float, 4> col3{13.0f, 14.0f, 15.0f, 16.0f};
    h.push_column(col3);  // wraps: overwrites the oldest column (col0)
    CHECK(h.count() == 3);  // saturates -- never exceeds Cols

    std::array<float, 4> col4{17.0f, 18.0f, 19.0f, 20.0f};
    h.push_column(col4);  // wraps again
    CHECK(h.count() == 3);

    SECTION("a column shorter than Rows is ignored") {
        History2D<4, 3> h2;
        std::array<float, 3> short_col{1.0f, 2.0f, 3.0f};
        h2.push_column(short_col);
        CHECK(h2.count() == 0);
        CHECK(h2.ordered().empty());
    }

    SECTION("a column longer than Rows uses only the first Rows elements") {
        History2D<4, 3> h3;
        std::array<float, 6> long_col{1.0f, 2.0f, 3.0f, 4.0f, 999.0f, 999.0f};
        h3.push_column(long_col);
        CHECK(h3.count() == 1);
        auto o = h3.ordered();
        REQUIRE(o.size() == 4);
        CHECK(o[0] == 1.0f);
        CHECK(o[1] == 2.0f);
        CHECK(o[2] == 3.0f);
        CHECK(o[3] == 4.0f);
    }
}

TEST_CASE("HEAT-02: after wrapping, ordered() lists columns oldest to newest in row-major layout",
          "[heat][unit]") {
    constexpr std::size_t kRows = 2;
    constexpr std::size_t kCols = 3;
    History2D<kRows, kCols> h;

    // Push 5 identifiable columns: column k's cells are k*10 + r (r = row index).
    // Capacity is 3, so columns 1 and 2 get evicted, leaving 3, 4, 5 oldest->newest.
    for (int k = 1; k <= 5; ++k) {
        std::array<float, kRows> col{};
        for (std::size_t r = 0; r < kRows; ++r) {
            col[r] = static_cast<float>(k * 10) + static_cast<float>(r);
        }
        h.push_column(col);
    }

    REQUIRE(h.count() == kCols);
    auto o = h.ordered();
    REQUIRE(o.size() == kRows * kCols);

    const int expected_k[kCols] = {3, 4, 5};
    for (std::size_t r = 0; r < kRows; ++r) {
        for (std::size_t c = 0; c < kCols; ++c) {
            const float expected = static_cast<float>(expected_k[c] * 10) + static_cast<float>(r);
            CHECK(o[r * kCols + c] == expected);
        }
    }
}

TEST_CASE("HEAT-03: clear() resets to empty and ordered()'s buffer is stable (no allocation)",
          "[heat][unit]") {
    History2D<3, 2> h;
    std::array<float, 3> col{1.0f, 2.0f, 3.0f};
    h.push_column(col);
    h.push_column(col);

    const float* first_ptr = h.ordered().data();

    h.push_column(col);  // wraps once more; storage identity must not change
    CHECK(h.ordered().data() == first_ptr);

    h.clear();
    CHECK(h.count() == 0);
    CHECK(h.ordered().empty());

    // clear() does not reallocate either: the scratch buffer keeps its address.
    h.push_column(col);
    CHECK(h.ordered().data() == first_ptr);
}
