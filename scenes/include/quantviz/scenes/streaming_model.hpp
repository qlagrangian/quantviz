#pragma once
// scenes/streaming_model.hpp — シーン 1「Streaming」: GBM + 逐次統計
//
// コア（Gbm, Welford, EwmaVariance）を 1 つの Model に束ね、描画が必要とする状態だけを
// POD の Snapshot として吐く。描画はこの Snapshot の純関数。

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/core/models/gbm.hpp"
#include "quantviz/core/stats/ewma.hpp"
#include "quantviz/core/stats/welford.hpp"

namespace quantviz::scenes {

struct StreamingSnapshot {
    double        t           = 0.0;  ///< sim 時刻（年）
    double        spot        = 0.0;
    double        log_return  = 0.0;  ///< 直近ステップの対数リターン
    double        mean_return = 0.0;  ///< Welford: 全履歴の平均（ステップ単位）
    double        var_return  = 0.0;  ///< Welford: 標本分散（ステップ単位）
    double        ewma_var    = 0.0;  ///< EWMA 分散（ステップ単位）
    double        mu_true     = 0.0;  ///< 現在のモデル mu（年率）— 参照線用
    double        sigma_true  = 0.0;  ///< 現在のモデル sigma（年率）— 参照線用
    std::uint64_t seq         = 0;    ///< ステップ通番（0 = 未ステップ）
};
static_assert(std::is_trivially_copyable_v<StreamingSnapshot>);

class StreamingModel {
public:
    using Snapshot = StreamingSnapshot;

    enum Param : std::uint32_t {
        kMu         = 1,
        kSigma      = 2,
        kEwmaLambda = 3,
    };

    struct Config {
        core::GbmParams gbm{};
        double          ewma_lambda = 0.94;
        std::uint64_t   seed        = 42;
    };

    StreamingModel() : StreamingModel(Config{}) {}
    explicit StreamingModel(Config cfg) : gbm_(cfg.gbm, cfg.seed), ewma_(cfg.ewma_lambda), seed_(cfg.seed) {}

    void step(double dt) {
        last_r_ = gbm_.step(dt);
        welford_.push(last_r_);
        ewma_.push(last_r_);
        ++seq_;
    }

    Snapshot snapshot() const noexcept {
        Snapshot s;
        s.t           = gbm_.time();
        s.spot        = gbm_.spot();
        s.log_return  = last_r_;
        s.mean_return = welford_.mean();
        s.var_return  = welford_.variance();
        s.ewma_var    = ewma_.variance();
        s.mu_true     = gbm_.params().mu;
        s.sigma_true  = gbm_.params().sigma;
        s.seq         = seq_;
        return s;
    }

    void apply(const bridge::Command& c) {
        switch (c.type) {
            case bridge::CommandType::SetParam:
                switch (c.param_id) {
                    case kMu:         gbm_.set_mu(c.value); break;
                    case kSigma:      gbm_.set_sigma(c.value); break;
                    case kEwmaLambda: ewma_.set_lambda(c.value); break;
                    default:          break;  // 未知の param_id は無視
                }
                break;
            case bridge::CommandType::Reset: reset(c.seed != 0 ? c.seed : seed_); break;
            default:                         break;  // 時計系は Runner が処理済み
        }
    }

    /// パスと統計を初期状態へ。mu / sigma / lambda は保持。
    void reset(std::uint64_t seed) {
        seed_ = seed;
        gbm_.reset(seed);
        welford_.reset();
        ewma_.reset();
        last_r_ = 0.0;
        seq_    = 0;
    }

    // テスト用の読み取りアクセサ
    const core::Gbm&          gbm() const noexcept { return gbm_; }
    const core::Welford&      welford() const noexcept { return welford_; }
    const core::EwmaVariance& ewma() const noexcept { return ewma_; }

private:
    core::Gbm          gbm_;
    core::Welford      welford_;
    core::EwmaVariance ewma_;
    std::uint64_t      seed_;
    double             last_r_ = 0.0;
    std::uint64_t      seq_    = 0;
};

static_assert(bridge::Model<StreamingModel>, "StreamingModel must satisfy the Model contract");

}  // namespace quantviz::scenes
