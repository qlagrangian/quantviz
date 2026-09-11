// STREAM-xx — scenes/streaming_model.hpp の契約テスト
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/core/models/gbm.hpp"
#include "quantviz/scenes/streaming_model.hpp"

using Catch::Matchers::WithinRel;
using quantviz::bridge::Command;
using quantviz::scenes::StreamingModel;
using quantviz::scenes::StreamingSnapshot;

namespace {
constexpr double kDt = 1.0 / (252.0 * 390.0);  // 1 分足
}

TEST_CASE("STREAM-01: satisfies the Model contract with a compact POD snapshot", "[stream][contract]") {
    STATIC_REQUIRE(quantviz::bridge::Model<StreamingModel>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<StreamingSnapshot>);
    STATIC_REQUIRE(sizeof(StreamingSnapshot) <= 128);  // 2 キャッシュライン以内
}

TEST_CASE("STREAM-02: a fresh model reports seq 0, spot s0 and the true parameters", "[stream][unit]") {
    StreamingModel m(StreamingModel::Config{{100.0, 0.05, 0.2}, 0.94, 42});
    const auto     s = m.snapshot();
    CHECK(s.seq == 0);
    CHECK(s.spot == 100.0);
    CHECK(s.t == 0.0);
    CHECK(s.mu_true == 0.05);
    CHECK(s.sigma_true == 0.2);
    CHECK(s.var_return == 0.0);
    CHECK(s.ewma_var == 0.0);
}

TEST_CASE("STREAM-03: stepping the scene equals stepping a bare Gbm with the same seed (dual-run)",
          "[stream][determinism]") {
    StreamingModel      m(StreamingModel::Config{{100.0, 0.05, 0.2}, 0.94, 777});
    quantviz::core::Gbm bare(quantviz::core::GbmParams{100.0, 0.05, 0.2}, 777);
    for (int i = 0; i < 1000; ++i) {
        m.step(kDt);
        const double r = bare.step(kDt);
        const auto   s = m.snapshot();
        REQUIRE(s.spot == bare.spot());
        REQUIRE(s.log_return == r);
        REQUIRE(s.seq == static_cast<std::uint64_t>(i + 1));
    }
    CHECK(m.welford().count() == 1000);
    CHECK(m.ewma().count() == 1000);
}

TEST_CASE("STREAM-04: SetParam changes mu/sigma/lambda for subsequent steps; unknown ids are ignored",
          "[stream][unit]") {
    StreamingModel m;
    m.apply(Command::set_param(StreamingModel::kSigma, 0.35));
    m.apply(Command::set_param(StreamingModel::kMu, -0.02));
    m.apply(Command::set_param(StreamingModel::kEwmaLambda, 0.8));
    m.apply(Command::set_param(999, 1.0));
    const auto s = m.snapshot();
    CHECK(s.sigma_true == 0.35);
    CHECK(s.mu_true == -0.02);
    CHECK(m.ewma().lambda() == 0.8);
    CHECK(s.seq == 0);  // パラメータ変更はステップを進めない
}

TEST_CASE("STREAM-05: Reset rewinds path and statistics but keeps the parameters", "[stream][unit]") {
    StreamingModel m;
    for (int i = 0; i < 100; ++i) m.step(kDt);
    m.apply(Command::set_param(StreamingModel::kSigma, 0.5));
    m.apply(Command::reset());
    const auto s = m.snapshot();
    CHECK(s.seq == 0);
    CHECK(s.spot == 100.0);
    CHECK(s.t == 0.0);
    CHECK(s.var_return == 0.0);
    CHECK(s.ewma_var == 0.0);
    CHECK(s.sigma_true == 0.5);
}

TEST_CASE("STREAM-06: Reset with an explicit seed replays that seed's path", "[stream][determinism]") {
    StreamingModel a(StreamingModel::Config{{}, 0.94, 1}), b(StreamingModel::Config{{}, 0.94, 2});
    b.apply(Command::reset(1));
    for (int i = 0; i < 200; ++i) {
        a.step(kDt);
        b.step(kDt);
        REQUIRE(a.snapshot().spot == b.snapshot().spot);
    }
}

TEST_CASE("STREAM-07: annualised realised and EWMA volatility converge to the true sigma",
          "[stream][statistical]") {
    const double   sigma = 0.25;
    StreamingModel m(StreamingModel::Config{{100.0, 0.0, sigma}, 0.999, 2718});
    for (int i = 0; i < 40000; ++i) m.step(kDt);
    const auto   s            = m.snapshot();
    const double realised_vol = std::sqrt(s.var_return / kDt);
    const double ewma_vol     = std::sqrt(s.ewma_var / kDt);
    CHECK_THAT(realised_vol, WithinRel(sigma, 0.02));  // SE ≈ sigma/sqrt(2N) ≈ 0.35 %
    CHECK_THAT(ewma_vol, WithinRel(sigma, 0.10));      // 実効窓 ≈ 1000 本 → SE ≈ 1.6 %
}
