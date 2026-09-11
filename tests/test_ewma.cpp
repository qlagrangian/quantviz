// EWMA-xx — core/stats/ewma.hpp の仕様テスト
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "quantviz/core/stats/ewma.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::core::EwmaVariance;

TEST_CASE("EWMA-01: the first observation seeds the variance with r^2", "[ewma][unit]") {
    EwmaVariance e(0.94);
    CHECK(e.count() == 0);
    CHECK(e.variance() == 0.0);
    e.push(0.02);
    CHECK(e.count() == 1);
    CHECK_THAT(e.variance(), WithinRel(0.0004, 1e-15));
}

TEST_CASE("EWMA-02: a constant input keeps the variance at c^2", "[ewma][property]") {
    EwmaVariance e(0.9);
    for (int i = 0; i < 1000; ++i) e.push(0.01);
    CHECK_THAT(e.variance(), WithinRel(1e-4, 1e-12));
    CHECK_THAT(e.volatility(), WithinRel(0.01, 1e-12));
}

TEST_CASE("EWMA-03: the recursion equals the closed-form weighted sum", "[ewma][numeric]") {
    const double                     lambda = 0.94;
    std::mt19937_64                  rng(3);
    std::normal_distribution<double> nd(0.0, 0.01);
    std::vector<double>              rs(300);
    for (auto& r : rs) r = nd(rng);

    EwmaVariance e(lambda);
    for (double r : rs) e.push(r);

    const std::size_t n      = rs.size();
    double            closed = std::pow(lambda, static_cast<double>(n - 1)) * rs[0] * rs[0];
    for (std::size_t i = 1; i < n; ++i)
        closed += (1.0 - lambda) * std::pow(lambda, static_cast<double>(n - 1 - i)) * rs[i] * rs[i];

    CHECK_THAT(e.variance(), WithinRel(closed, 1e-10));
}

TEST_CASE("EWMA-04: after a shock, zero returns decay the variance geometrically by lambda", "[ewma][numeric]") {
    EwmaVariance e(0.9);
    e.push(0.05);  // var = 0.0025
    for (int i = 0; i < 20; ++i) e.push(0.0);
    CHECK_THAT(e.variance(), WithinRel(0.0025 * std::pow(0.9, 20), 1e-12));
}

TEST_CASE("EWMA-05: smaller lambda adapts faster to a regime shift", "[ewma][property]") {
    EwmaVariance slow(0.99), fast(0.90);
    for (int i = 0; i < 500; ++i) {
        slow.push(0.01);
        fast.push(0.01);
    }
    for (int i = 0; i < 20; ++i) {  // ボラが 3 倍になる
        slow.push(0.03);
        fast.push(0.03);
    }
    const double target = 0.03 * 0.03;
    CHECK(std::abs(fast.variance() - target) < std::abs(slow.variance() - target));
}

TEST_CASE("EWMA-06: set_lambda clamps to (0,1) and maps NaN to the default", "[ewma][unit]") {
    EwmaVariance e(0.5);
    e.set_lambda(1.5);
    CHECK(e.lambda() == EwmaVariance::kMaxLambda);
    e.set_lambda(-2.0);
    CHECK(e.lambda() == EwmaVariance::kMinLambda);
    e.set_lambda(std::numeric_limits<double>::quiet_NaN());
    CHECK(e.lambda() == 0.94);
}

TEST_CASE("EWMA-07: reset clears the state and the next push re-seeds", "[ewma][unit]") {
    EwmaVariance e(0.94);
    e.push(0.1);
    e.reset();
    CHECK(e.count() == 0);
    CHECK(e.variance() == 0.0);
    e.push(0.02);
    CHECK_THAT(e.variance(), WithinRel(0.0004, 1e-15));
}
