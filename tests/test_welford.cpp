// WEL-xx — core/stats/welford.hpp の仕様テスト
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <random>
#include <vector>

#include "quantviz/core/stats/welford.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::core::Welford;

TEST_CASE("WEL-01: an empty accumulator reports count 0, mean 0, variance 0", "[welford][unit]") {
    Welford w;
    CHECK(w.count() == 0);
    CHECK(w.mean() == 0.0);
    CHECK(w.variance() == 0.0);
    CHECK(w.population_variance() == 0.0);
}

TEST_CASE("WEL-02: a single value is its own mean with zero variance", "[welford][unit]") {
    Welford w;
    w.push(3.5);
    CHECK(w.count() == 1);
    CHECK(w.mean() == 3.5);
    CHECK(w.variance() == 0.0);
}

TEST_CASE("WEL-03: matches the two-pass mean and sample variance", "[welford][numeric]") {
    std::mt19937_64                  rng(7);
    std::normal_distribution<double> nd(1.0, 2.0);
    std::vector<double>              xs(5000);
    for (auto& x : xs) x = nd(rng);

    Welford w;
    double  sum = 0.0;
    for (double x : xs) {
        w.push(x);
        sum += x;
    }
    const double mean = sum / static_cast<double>(xs.size());
    double       ss   = 0.0;
    for (double x : xs) ss += (x - mean) * (x - mean);
    const double var = ss / static_cast<double>(xs.size() - 1);

    CHECK_THAT(w.mean(), WithinRel(mean, 1e-12));
    CHECK_THAT(w.variance(), WithinRel(var, 1e-10));
    CHECK_THAT(w.population_variance(), WithinRel(ss / static_cast<double>(xs.size()), 1e-10));
    CHECK_THAT(w.stddev(), WithinRel(std::sqrt(var), 1e-10));
}

TEST_CASE("WEL-04: stays accurate with a huge offset where the naive E[x^2]-E[x]^2 formula fails",
          "[welford][numeric]") {
    Welford      w;
    const double offset = 1e9;
    const double vals[] = {4.0, 7.0, 13.0, 16.0};  // mean 10, sample var 30
    for (double v : vals) w.push(offset + v);
    CHECK_THAT(w.mean(), WithinRel(offset + 10.0, 1e-15));
    CHECK_THAT(w.variance(), WithinRel(30.0, 1e-6));
}

TEST_CASE("WEL-05: reset returns to the empty state", "[welford][unit]") {
    Welford w;
    w.push(1.0);
    w.push(2.0);
    w.reset();
    CHECK(w.count() == 0);
    CHECK(w.mean() == 0.0);
    CHECK(w.variance() == 0.0);
}

TEST_CASE("WEL-06: the standard textbook example (2,4,4,4,5,5,7,9) gives mean 5, pop var 4",
          "[welford][unit]") {
    Welford w;
    for (double v : {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}) w.push(v);
    CHECK_THAT(w.mean(), WithinAbs(5.0, 1e-15));
    CHECK_THAT(w.population_variance(), WithinAbs(4.0, 1e-14));
    CHECK_THAT(w.variance(), WithinAbs(32.0 / 7.0, 1e-14));
}
