// RING-xx — bridge/spsc_ring.hpp の仕様テスト（docs/03_tdd_spec.md 参照）
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <thread>
#include <vector>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/spsc_ring.hpp"

using quantviz::bridge::Command;
using quantviz::bridge::SpscRing;

TEST_CASE("RING-01: a new ring is empty and pop fails", "[ring][unit]") {
    SpscRing<int, 8> ring;
    int              v = -1;
    CHECK(ring.empty_approx());
    CHECK(ring.size_approx() == 0);
    CHECK_FALSE(ring.try_pop(v));
    CHECK(v == -1);
}

TEST_CASE("RING-02: values come out in FIFO order", "[ring][unit]") {
    SpscRing<int, 8> ring;
    for (int i = 0; i < 5; ++i) REQUIRE(ring.try_push(i));
    CHECK(ring.size_approx() == 5);
    for (int i = 0; i < 5; ++i) {
        int v = -1;
        REQUIRE(ring.try_pop(v));
        CHECK(v == i);
    }
    CHECK(ring.empty_approx());
}

TEST_CASE("RING-03: exactly Capacity elements fit, the next push fails without blocking", "[ring][unit]") {
    SpscRing<int, 4> ring;
    for (int i = 0; i < 4; ++i) REQUIRE(ring.try_push(i));
    CHECK_FALSE(ring.try_push(99));
    CHECK(ring.size_approx() == 4);
    int v = 0;
    REQUIRE(ring.try_pop(v));
    CHECK(v == 0);
    CHECK(ring.try_push(99));  // 1 つ空いたので入る
}

TEST_CASE("RING-04: wraparound keeps order across many cycles of the buffer", "[ring][unit]") {
    SpscRing<std::uint32_t, 4> ring;
    std::uint32_t              next_push = 0, next_pop = 0;
    for (int cycle = 0; cycle < 1000; ++cycle) {
        for (int k = 0; k < 3; ++k) REQUIRE(ring.try_push(next_push++));
        for (int k = 0; k < 3; ++k) {
            std::uint32_t v = 0;
            REQUIRE(ring.try_pop(v));
            REQUIRE(v == next_pop++);
        }
    }
    CHECK(ring.empty_approx());
}

TEST_CASE("RING-05: capacity() is a compile-time constant and the type constraints hold", "[ring][unit]") {
    STATIC_REQUIRE(SpscRing<int, 16>::capacity() == 16);
    STATIC_REQUIRE(std::is_trivially_copyable_v<Command>);
    // 非 2 冪の Capacity と非 trivially-copyable な T は static_assert でコンパイルエラーになる（仕様）
}

TEST_CASE("RING-06: POD structs (Command) round-trip bit-exactly", "[ring][unit]") {
    SpscRing<Command, 2> ring;
    REQUIRE(ring.try_push(Command::set_param(7, 1.25)));
    REQUIRE(ring.try_push(Command::reset(123456789ULL)));
    Command c{};
    REQUIRE(ring.try_pop(c));
    CHECK(c.type == quantviz::bridge::CommandType::SetParam);
    CHECK(c.param_id == 7);
    CHECK(c.value == 1.25);
    REQUIRE(ring.try_pop(c));
    CHECK(c.type == quantviz::bridge::CommandType::Reset);
    CHECK(c.seed == 123456789ULL);
}

TEST_CASE("RING-07: producer/consumer threads transfer 1M items with no loss, dup or reorder",
          "[ring][concurrency]") {
    constexpr std::uint64_t kN = 1'000'000;
    SpscRing<std::uint64_t, 1024> ring;

    std::thread producer([&] {
        for (std::uint64_t i = 0; i < kN; ++i) {
            while (!ring.try_push(i)) std::this_thread::yield();
        }
    });

    std::uint64_t expected = 0;
    bool          ordered  = true;
    while (expected < kN) {
        std::uint64_t v = 0;
        if (ring.try_pop(v)) {
            if (v != expected) {
                ordered = false;
                break;
            }
            ++expected;
        } else {
            std::this_thread::yield();
        }
    }
    producer.join();
    CHECK(ordered);
    CHECK(expected == kN);
    CHECK(ring.empty_approx());
}

TEST_CASE("RING-08: hot fields live on separate cache lines (no false sharing by layout)", "[ring][unit]") {
    // head/tail/両キャッシュ/バッファがそれぞれ 64B 境界に置かれるため、
    // オブジェクト全体は少なくとも 4 ラインを超える大きさになる。
    STATIC_REQUIRE(sizeof(SpscRing<int, 2>) >= 4 * quantviz::bridge::kCacheLineSize);
    STATIC_REQUIRE(alignof(SpscRing<int, 2>) == quantviz::bridge::kCacheLineSize);
}
