#pragma once
// viz/panels/garch_panel.hpp — シーン 3「GARCH」の描画パネル群
//
// 責務: Runner から Snapshot を poll → σ_t の 3 本を History に蓄積 → ImPlot で描く
// （Volatility / 尤度面 heatmap + 最適化軌跡 / Control）。Model には一切触らない。

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "quantviz/bridge/runner.hpp"
#include "quantviz/scenes/garch_model.hpp"
#include "quantviz/viz/clock_controls.hpp"
#include "quantviz/viz/history.hpp"
#include "quantviz/viz/scene_registry.hpp"

namespace quantviz::viz {

class GarchPanel {
public:
    /// Snapshot が ~10 KB あるのでリングは 256 枚（計画書「Snapshot のサイズ方針」）。
    static constexpr std::size_t kSnapshotCapacity = 256;
    using Runner = bridge::Runner<scenes::GarchModel, kSnapshotCapacity>;

    GarchPanel(double dt, const scenes::GarchModel::Config& initial, double initial_speed);

    /// 毎フレーム 1 回。poll → 描画 → Command 送信。
    void draw(Runner& runner);

private:
    void ingest(Runner& runner);
    void draw_volatility();
    void draw_surface();
    void draw_controls(Runner& runner);
    void clear_history();

    static constexpr std::size_t kHistory = 8192;  ///< 100 steps/s で約 80 秒ぶん
    static constexpr std::size_t kGrid    = scenes::GarchSnapshot::kGrid;

    double            dt_;
    History<kHistory> sigma_true_;      ///< 真の σ_t（年率 %）
    History<kHistory> sigma_filtered_;  ///< 真値パラメータで窓をフィルタした σ_t
    History<kHistory> sigma_est_;       ///< 推定パラメータで窓をフィルタした σ_t

    scenes::GarchSnapshot last_{};
    std::uint64_t         received_   = 0;
    double                rate_ema_   = 0.0;  ///< 受信 Snapshot/秒（表示用）
    double                last_wall_  = 0.0;
    std::uint64_t         last_count_ = 0;

    /// heatmap 用に行（β 軸）を反転してクランプした格子。ImPlot は行 0 を上端に描くため。
    std::array<double, kGrid * kGrid> heat_{};

    // UI 状態（ImGui のスライダーは float / int）
    ClockControlState clock_{};
    float             omega_;
    float             alpha_;
    float             beta_;
    int               optimizer_;
    int               window_;
    float             window_days_ = 250.0f;
    bool              follow_      = true;
};

/// main.cpp / SceneRegistry 用のファクトリ: GARCH の Runner + Panel を 1 つのシーンに束ねる。
std::unique_ptr<Scene> make_garch_scene();

}  // namespace quantviz::viz
