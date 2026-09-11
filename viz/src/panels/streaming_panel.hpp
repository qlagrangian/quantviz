#pragma once
// viz/panels/streaming_panel.hpp — シーン 1「Streaming」の描画パネル群
//
// 責務: Runner から Snapshot を poll → History に蓄積 → ImPlot で描く → UI 操作を Command で返す。
// Model には一切触らない（Snapshot の純関数 + Command の生成のみ）。

#include <cstdint>

#include "quantviz/bridge/runner.hpp"
#include "quantviz/scenes/streaming_model.hpp"
#include "quantviz/viz/history.hpp"

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
    std::uint64_t             received_    = 0;
    double                    rate_ema_    = 0.0;  ///< 受信 Snapshot/秒（表示用）
    double                    last_wall_   = 0.0;
    std::uint64_t             last_count_  = 0;

    // UI 状態（ImGui のスライダーは float）
    float mu_;
    float sigma_;
    float lambda_;
    float speed_;
    float window_days_ = 5.0f;
    bool  paused_      = false;
    bool  follow_      = true;
};

}  // namespace quantviz::viz
