#pragma once
// viz/panels/streaming_panel.hpp — シーン 1「Streaming」の描画パネル群
//
// 責務: Runner から Snapshot を poll → History に蓄積 → ImPlot で描く → UI 操作を Command で返す。
// Model には一切触らない（Snapshot の純関数 + Command の生成のみ）。

#include <cstdint>
#include <memory>

#include "quantviz/bridge/runner.hpp"
#include "quantviz/scenes/streaming_model.hpp"
#include "quantviz/viz/clock_controls.hpp"
#include "quantviz/viz/history.hpp"
#include "quantviz/viz/rate_meter.hpp"
#include "quantviz/viz/scene_registry.hpp"

namespace quantviz::viz {

class StreamingPanel {
public:
    using Runner = bridge::Runner<scenes::StreamingModel>;

    StreamingPanel(double dt, const scenes::StreamingModel::Config& initial, double initial_speed);

    /// 毎フレーム 1 回。poll → 描画 → Command 送信。
    void draw(Runner& runner);

private:
    void ingest(Runner& runner);
    void draw_price();
    void draw_volatility();
    void draw_controls(Runner& runner);
    void clear_history();

    static constexpr std::size_t kHistory = 8192;

    double            dt_;
    History<kHistory> price_;
    History<kHistory> realised_vol_;
    History<kHistory> ewma_vol_;
    History<kHistory> sigma_true_;

    scenes::StreamingSnapshot last_{};
    std::uint64_t             prev_seq_ = 0;  ///< Reset をまたぐ古い Snapshot の検出用
    std::uint64_t             received_ = 0;
    RateMeter                 rate_;  ///< 受信 Snapshot/秒（表示用）

    // UI 状態（ImGui のスライダーは float）
    ClockControlState clock_{};  ///< speed / paused の共通 Control 状態
    float             mu_;
    float             sigma_;
    float             lambda_;
    float             window_days_ = 5.0f;
    bool              follow_      = true;
};

/// main.cpp / SceneRegistry 用のファクトリ: Streaming の Runner + Panel を 1 つのシーンに束ねる。
std::unique_ptr<Scene> make_streaming_scene();

}  // namespace quantviz::viz
