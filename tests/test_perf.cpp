// PERF-xx — viz/perf.hpp（vizcore: GUI 非依存）の仕様テスト。
//
// パフォーマンスパネルが使う「純粋な計算」だけをここで固定する（描画そのものは手動チェックリスト）。
// どれも時間源も乱数も持たないので、完全に決定的なテストになる。
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "quantviz/viz/perf.hpp"

using Catch::Matchers::WithinAbs;
using quantviz::viz::dropped_ratio;
using quantviz::viz::Ema;
using quantviz::viz::FixedHistogram;
using quantviz::viz::percentile_sorted;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

}  // namespace

// ctest に登録される名前に [ ] は使えない（CatchAddTests.cmake が CMake のリスト展開で
// 括弧を食い、以降のテスト名まで 1 つに繋がってしまう）。区間は言葉で書く。
TEST_CASE("PERF-01: FixedHistogram bins each value half-open from lo up to hi, clamps the ends, drops NaN",
          "[perf][unit]") {
    SECTION("uniform edges 0,1,2,3,4") {
        FixedHistogram<4> h{{0.0, 1.0, 2.0, 3.0, 4.0}};
        CHECK(h.total() == 0);

        h.add(0.0);    // 下端はビンに含む
        h.add(0.999);  // ビン 0
        h.add(1.0);    // 内側の境界は上のビンへ（[lo, hi)）
        h.add(2.5);    // ビン 2
        h.add(3.999);  // ビン 3
        h.add(4.0);    // 上端ちょうどは最後のビンへ（最後だけ閉区間）
        h.add(10.0);   // 上端超えはクランプして最後のビンへ
        h.add(-1.0);   // 下端未満はクランプしてビン 0 へ

        CHECK(h.counts[0] == 3);
        CHECK(h.counts[1] == 1);
        CHECK(h.counts[2] == 1);
        CHECK(h.counts[3] == 3);
        CHECK(h.total() == 8);
        CHECK(h.dropped == 0);

        h.add(kNaN);  // NaN はどのビンにも入れず dropped で数える
        CHECK(h.dropped == 1);
        CHECK(h.total() == 8);

        h.add(kInf);   // ±inf はクランプ（NaN 扱いにしない）
        h.add(-kInf);
        CHECK(h.counts[0] == 4);
        CHECK(h.counts[3] == 4);
        CHECK(h.total() == 10);

        h.reset();
        CHECK(h.total() == 0);
        CHECK(h.dropped == 0);
        CHECK(h.edges[0] == 0.0);  // 境界は reset で消えない
        CHECK(h.edges[4] == 4.0);
    }

    SECTION("non-uniform edges are searched, not computed") {
        FixedHistogram<4> h{{0.0, 0.5, 10.0, 1000.0, 1.0e6}};
        h.add(0.0);       // [0, 0.5)
        h.add(0.4999);    // [0, 0.5)
        h.add(0.5);       // [0.5, 10)
        h.add(9.999);     // [0.5, 10)
        h.add(10.0);      // [10, 1000)
        h.add(999.0);     // [10, 1000)
        h.add(1000.0);    // [1000, 1e6]
        h.add(999999.0);  // [1000, 1e6]
        CHECK(h.counts[0] == 2);
        CHECK(h.counts[1] == 2);
        CHECK(h.counts[2] == 2);
        CHECK(h.counts[3] == 2);
        CHECK(h.total() == 8);
    }

    SECTION("a single-bin histogram catches everything") {
        FixedHistogram<1> h{{1.0, 2.0}};
        h.add(0.0);
        h.add(1.5);
        h.add(99.0);
        CHECK(h.counts[0] == 3);
        CHECK(h.total() == 3);
    }
}

TEST_CASE("PERF-02: percentile_sorted interpolates linearly between the two neighbouring ranks",
          "[perf][unit]") {
    SECTION("five equally spaced samples: p50 / p95 / p99 by hand") {
        // pos = p (n-1) = 4p、x = xs[floor(pos)] + frac (xs[floor+1] - xs[floor])
        const std::vector<double> xs{1.0, 2.0, 3.0, 4.0, 5.0};
        const std::span<const double> s(xs);
        CHECK_THAT(percentile_sorted(s, 0.50), WithinAbs(3.0, 1e-12));   // pos = 2.0  → 3
        CHECK_THAT(percentile_sorted(s, 0.95), WithinAbs(4.8, 1e-12));   // pos = 3.8  → 4 + .8
        CHECK_THAT(percentile_sorted(s, 0.99), WithinAbs(4.96, 1e-12));  // pos = 3.96 → 4 + .96
        CHECK_THAT(percentile_sorted(s, 0.0), WithinAbs(1.0, 1e-12));
        CHECK_THAT(percentile_sorted(s, 1.0), WithinAbs(5.0, 1e-12));
    }

    SECTION("unevenly spaced samples interpolate inside the straddled interval") {
        const std::vector<double> xs{1.0, 2.0, 4.0, 8.0};
        const std::span<const double> s(xs);
        CHECK_THAT(percentile_sorted(s, 0.50), WithinAbs(3.0, 1e-12));   // pos = 1.5  → 2 + .5·2
        CHECK_THAT(percentile_sorted(s, 0.95), WithinAbs(7.4, 1e-12));   // pos = 2.85 → 4 + .85·4
        CHECK_THAT(percentile_sorted(s, 0.99), WithinAbs(7.88, 1e-12));  // pos = 2.97 → 4 + .97·4
    }

    SECTION("one sample: every percentile is that sample") {
        const std::vector<double> xs{42.0};
        const std::span<const double> s(xs);
        CHECK_THAT(percentile_sorted(s, 0.0), WithinAbs(42.0, 1e-12));
        CHECK_THAT(percentile_sorted(s, 0.5), WithinAbs(42.0, 1e-12));
        CHECK_THAT(percentile_sorted(s, 1.0), WithinAbs(42.0, 1e-12));
    }

    SECTION("p outside [0,1] is clamped; NaN p is taken as 0; an empty span is NaN") {
        const std::vector<double> xs{1.0, 2.0, 3.0};
        const std::span<const double> s(xs);
        CHECK_THAT(percentile_sorted(s, -1.0), WithinAbs(1.0, 1e-12));
        CHECK_THAT(percentile_sorted(s, 2.0), WithinAbs(3.0, 1e-12));
        CHECK_THAT(percentile_sorted(s, kNaN), WithinAbs(1.0, 1e-12));
        CHECK(std::isnan(percentile_sorted(std::span<const double>{}, 0.5)));
    }
}

TEST_CASE("PERF-03: Ema starts at the first sample and converges geometrically to a steady input",
          "[perf][numeric]") {
    // 誤差は 1 サンプルごとに (1-alpha) 倍になる（統計ではなく等比数列なので seed は不要）。
    // alpha = 0.2、初期値 0、入力 1 なら 100 サンプル後の誤差は 0.8^100 = 2.0e-10 < 1e-9。
    SECTION("a constant input is reached to 1e-9 within 100 samples") {
        Ema e(0.2);
        CHECK(e.value() == 0.0);  // サンプル前は 0
        e.add(0.0);               // 最初のサンプルで初期化（平滑化しない）
        CHECK(e.value() == 0.0);
        for (int i = 0; i < 100; ++i) e.add(1.0);
        CHECK_THAT(e.value(), WithinAbs(1.0, 1e-9));
    }

    SECTION("the first sample initialises the state exactly") {
        Ema e(0.1);
        e.add(7.5);
        CHECK_THAT(e.value(), WithinAbs(7.5, 1e-12));
        e.add(7.5);
        CHECK_THAT(e.value(), WithinAbs(7.5, 1e-12));  // 定常入力は動かない
    }

    SECTION("an alternating input settles on a 2-cycle whose two values average to the mean") {
        // 交互入力 a,b の 2 周期不動点は v* = ((1-a)A + B)/(2-a)、v** = ((1-a)B + A)/(2-a) で、
        // v* + v** = A + B（厳密）。収束は 2 サンプルごとに (1-alpha)^2 倍。
        // 200 サンプル後の誤差 ~ 0.8^200 ≈ 4e-20 なので 1e-9 は十分に緩い。
        Ema    e(0.2);
        double after_a = 0.0;
        double after_b = 0.0;
        for (int i = 0; i < 100; ++i) {
            e.add(0.0);
            after_a = e.value();
            e.add(2.0);
            after_b = e.value();
        }
        CHECK_THAT(0.5 * (after_a + after_b), WithinAbs(1.0, 1e-9));
        CHECK(after_a < 1.0);
        CHECK(after_b > 1.0);
    }

    SECTION("NaN samples are ignored, before and after initialisation") {
        Ema e(0.5);
        e.add(kNaN);
        CHECK(e.value() == 0.0);  // NaN では初期化しない
        e.add(4.0);
        CHECK_THAT(e.value(), WithinAbs(4.0, 1e-12));
        e.add(kNaN);
        CHECK_THAT(e.value(), WithinAbs(4.0, 1e-12));
    }

    SECTION("alpha outside (0,1] falls back to 1 = no smoothing") {
        for (const double bad : {0.0, -0.5, 2.0, kNaN}) {
            Ema e(bad);
            CHECK(e.alpha() == 1.0);
            e.add(1.0);
            e.add(5.0);
            CHECK_THAT(e.value(), WithinAbs(5.0, 1e-12));  // 常に最新値
        }
        Ema ok(0.25);
        CHECK(ok.alpha() == 0.25);
    }
}

TEST_CASE("PERF-04: dropped_ratio is dropped / (dropped + received), with 0/0 defined as 0",
          "[perf][unit]") {
    CHECK(dropped_ratio(0, 0) == 0.0);
    CHECK(dropped_ratio(0, 10) == 0.0);
    CHECK(dropped_ratio(10, 0) == 1.0);
    CHECK_THAT(dropped_ratio(1, 3), WithinAbs(0.25, 1e-12));
    CHECK_THAT(dropped_ratio(3, 1), WithinAbs(0.75, 1e-12));
    CHECK_THAT(dropped_ratio(1, 1), WithinAbs(0.5, 1e-12));
    // 大きな累計でも 0..1 に収まる
    const std::uint64_t big = 4'000'000'000ULL;
    CHECK_THAT(dropped_ratio(big, big), WithinAbs(0.5, 1e-12));
    CHECK(dropped_ratio(1, big) >= 0.0);
    CHECK(dropped_ratio(1, big) <= 1.0);
}
