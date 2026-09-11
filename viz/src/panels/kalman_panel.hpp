#pragma once
// viz/panels/kalman_panel.hpp — シーン 4「Kalman pair」の描画パネル群
//
// 責務: Runner から Snapshot を poll → History に蓄積 → ImPlot で描く → UI 操作を Command で返す。
// Model には一切触らない（Snapshot の純関数 + Command の生成のみ）。
// ウィンドウは Prices / Hedge ratio / Spread / Control の 4 枚。

#include <cstddef>
#include <cstdint>
#include <memory>

#include "quantviz/bridge/runner.hpp"
#include "quantviz/scenes/kalman_pair_model.hpp"
#include "quantviz/viz/clock_controls.hpp"
#include "quantviz/viz/history.hpp"
#include "quantviz/viz/scene_registry.hpp"

namespace quantviz::viz {

class KalmanPanel {
public:
    using Runner = bridge::Runner<scenes::KalmanPairModel>;

    KalmanPanel(const scenes::KalmanPairModel::Config& initial, double initial_speed);

    /// 毎フレーム 1 回。poll → 描画 → Command 送信。
    void draw(Runner& runner);

private:
    void ingest(Runner& runner);
    void draw_prices();
    void draw_hedge_ratio();
    void draw_spread();
    void draw_controls(Runner& runner);
    void clear_history();

    static constexpr std::size_t kHistory = 4096;  // 4096 取引日 ≈ 16 年（window スライダーの上限は 4000 日）

    History<kHistory> px_x_;        ///< x（説明変数側の価格）
    History<kHistory> px_y_;        ///< y（被説明変数側の価格）
    History<kHistory> beta_true_;   ///< 真の β_t（参照線）
    History<kHistory> beta_hat_;    ///< カルマン推定 β̂_t
    History<kHistory> beta_lo_;     ///< β̂ − 2√P（帯の下端）
    History<kHistory> beta_hi_;     ///< β̂ + 2√P（帯の上端）
    History<kHistory> spread_;      ///< ヘッジ後スプレッド y − β̂ x

    scenes::KalmanPairSnapshot last_{};
    std::uint64_t              prev_seq_   = 0;  ///< Reset をまたぐ古い Snapshot の検出用
    std::uint64_t              received_   = 0;
    double                     rate_ema_   = 0.0;  ///< 受信 Snapshot/秒（表示用）
    double                     last_wall_  = 0.0;
    std::uint64_t              last_count_ = 0;

    // UI 状態（ImGui のスライダーは float）
    ClockControlState clock_{};  ///< speed / paused の共通 Control 状態
    float             obs_noise_;
    float             state_noise_;
    float             beta_center_;
    float             window_days_ = 500.0f;
    bool              follow_      = true;
};

/// main.cpp / SceneRegistry 用のファクトリ: Kalman pair の Runner + Panel を 1 つのシーンに束ねる。
std::unique_ptr<Scene> make_kalman_scene();

}  // namespace quantviz::viz
