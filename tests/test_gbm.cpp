// GBM-xx — core/models/gbm.hpp の仕様テスト（決定性・厳密解・統計的性質）
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "quantviz/core/models/gbm.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::core::Gbm;
using quantviz::core::GbmParams;

namespace {
std::vector<double> path(Gbm& g, int n, double dt) {
    std::vector<double> p;
    p.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        g.step(dt);
        p.push_back(g.spot());
    }
    return p;
}
}  // namespace

TEST_CASE("GBM-01: initial state is (s0, t=0) with the given parameters", "[gbm][unit]") {
    Gbm g(GbmParams{120.0, 0.03, 0.25}, 7);
    CHECK(g.spot() == 120.0);
    CHECK(g.time() == 0.0);
    CHECK(g.params().mu == 0.03);
    CHECK(g.params().sigma == 0.25);
}

TEST_CASE("GBM-02: the same seed reproduces the identical path bit for bit", "[gbm][determinism]") {
    Gbm a(GbmParams{}, 2024), b(GbmParams{}, 2024);
    CHECK(path(a, 1000, 1.0 / 252) == path(b, 1000, 1.0 / 252));
}

TEST_CASE("GBM-03: different seeds give different paths", "[gbm][determinism]") {
    Gbm a(GbmParams{}, 1), b(GbmParams{}, 2);
    CHECK(path(a, 50, 1.0 / 252) != path(b, 50, 1.0 / 252));
}

TEST_CASE("GBM-04: with sigma = 0 the path is the deterministic exponential s0 * exp(mu t)", "[gbm][numeric]") {
    Gbm          g(GbmParams{100.0, 0.07, 0.0}, 1);
    const double dt = 1.0 / 252;
    for (int i = 0; i < 252; ++i) g.step(dt);
    CHECK_THAT(g.time(), WithinRel(1.0, 1e-12));
    CHECK_THAT(g.spot(), WithinRel(100.0 * std::exp(0.07), 1e-12));
}

TEST_CASE("GBM-05: the spot stays strictly positive even for extreme volatility", "[gbm][property]") {
    Gbm g(GbmParams{1.0, 0.0, 3.0}, 99);
    for (int i = 0; i < 20000; ++i) {
        g.step(1.0 / 252);
        REQUIRE(g.spot() > 0.0);
        REQUIRE(std::isfinite(g.spot()));
    }
}

TEST_CASE("GBM-06: step() returns log(S_new / S_old)", "[gbm][unit]") {
    Gbm g(GbmParams{}, 5);
    for (int i = 0; i < 100; ++i) {
        const double before = g.spot();
        const double r      = g.step(1.0 / 252);
        CHECK_THAT(r, WithinAbs(std::log(g.spot() / before), 1e-12));
    }
}

TEST_CASE("GBM-07: log returns have mean (mu - sigma^2/2) dt and variance sigma^2 dt (fixed seed, 4 SE)",
          "[gbm][statistical]") {
    const double mu = 0.05, sigma = 0.20, dt = 1.0 / 252;
    const int    n = 200000;
    Gbm          g(GbmParams{100.0, mu, sigma}, 31415);

    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < n; ++i) {
        const double r = g.step(dt);
        sum += r;
        sum2 += r * r;
    }
    const double mean = sum / n;
    const double var  = sum2 / n - mean * mean;

    const double expect_mean = (mu - 0.5 * sigma * sigma) * dt;
    const double expect_var  = sigma * sigma * dt;
    const double se_mean     = sigma * std::sqrt(dt) / std::sqrt(static_cast<double>(n));
    const double se_var      = expect_var * std::sqrt(2.0 / n);

    CHECK_THAT(mean, WithinAbs(expect_mean, 4.0 * se_mean));
    CHECK_THAT(var, WithinAbs(expect_var, 4.0 * se_var));
}

TEST_CASE("GBM-08: reset(seed) replays the path; parameters set via set_* survive the reset",
          "[gbm][unit]") {
    Gbm        g(GbmParams{}, 11);
    const auto first = path(g, 100, 1.0 / 252);
    g.set_mu(0.10);
    g.reset(11);
    CHECK(g.spot() == 100.0);
    CHECK(g.time() == 0.0);
    CHECK(g.params().mu == 0.10);        // 保持される
    const auto second = path(g, 100, 1.0 / 252);
    CHECK(first != second);              // mu が違うので別パス
    g.set_mu(0.05);
    g.reset(11);
    CHECK(path(g, 100, 1.0 / 252) == first);  // 元に戻せば同一
}

TEST_CASE("GBM-09: set_sigma clamps negative volatility to zero", "[gbm][unit]") {
    Gbm g;
    g.set_sigma(-1.0);
    CHECK(g.params().sigma == 0.0);
}

TEST_CASE("GBM-10: time accumulates n * dt", "[gbm][unit]") {
    Gbm g;
    for (int i = 0; i < 1000; ++i) g.step(0.001);
    CHECK_THAT(g.time(), WithinAbs(1.0, 1e-9));
}
