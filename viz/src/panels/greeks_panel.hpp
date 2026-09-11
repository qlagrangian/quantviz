#pragma once
// viz/panels/greeks_panel.hpp — シーン 2「Greeks」の描画パネル群
//
// 責務: Runner から Snapshot を poll → 最新 1 枚だけを保持 → ImPlot で描く → UI 操作を Command で返す。
// Model には一切触らない（Snapshot の純関数 + Command の生成のみ）。
// このシーンは時系列ではなく「今の断面」（K 軸の Greeks ストリップと Γ(S,T) 格子）を見るので、
// Streaming と違って History は持たない。Snapshot が大きい（16 KB 強）ので Runner の SnapCap は 256。

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "quantviz/bridge/runner.hpp"
#include "quantviz/scenes/greeks_model.hpp"
#include "quantviz/viz/clock_controls.hpp"
#include "quantviz/viz/scene_registry.hpp"

namespace quantviz::viz {

class GreeksPanel {
public:
    /// SnapCap は make_greeks_scene() の RunnerScene と一致させること（Snapshot が大きいため 256）。
    using Runner = bridge::Runner<scenes::GreeksModel, 256, 256>;

    GreeksPanel(const scenes::GreeksModel::Config& initial, double initial_speed);

    /// 毎フレーム 1 回。poll → 描画 → Command 送信。
    void draw(Runner& runner);

private:
    using Snap = scenes::GreeksSnapshot;

    void ingest(Runner& runner);
    void draw_strip();
    void draw_surface();
    void draw_controls(Runner& runner);

    /// spot に最も近いストライクの index（軸は等間隔なので丸めで求まる）。
    std::size_t atm_index() const noexcept;

    Snap          last_{};
    std::uint64_t received_   = 0;
    double        rate_ema_   = 0.0;  ///< 受信 Snapshot/秒（表示用）
    double        last_wall_  = 0.0;
    std::uint64_t last_count_ = 0;

    /// ImPlot のヒートマップは row-major で「row 0 を上端」に描く。Snapshot は [iS][iT] なので、
    /// 表示用に [iT 降順][iS 昇順] へ転置したコピーを毎フレーム作る（48x32 = 1536 要素）。
    std::array<double, Snap::kGridS * Snap::kGridT> heat_{};

    /// 実務単位へ換算した表示用の列（Snapshot は「1.0 あたり」で持つ。black_scholes.hpp の
    /// 「1 日 / 1bp / 1% への換算は呼び出し側の責務」に従い、換算はここで行う）。
    std::array<double, Snap::kStrikes> vega_disp_{};   ///< ボラ 1 ポイント（= 0.01）あたり
    std::array<double, Snap::kStrikes> rho_disp_{};    ///< 金利 1 % あたり
    std::array<double, Snap::kStrikes> theta_disp_{};  ///< 1 日あたり（1/365 年）

    // UI 状態（ImGui のスライダーは float）
    ClockControlState clock_{};  ///< speed / paused の共通 Control 状態
    float             r_;
    float             sigma_;
    float             maturity_;
    float             strike_span_;
};

/// main.cpp / SceneRegistry 用のファクトリ: Greeks の Runner + Panel を 1 つのシーンに束ねる。
std::unique_ptr<Scene> make_greeks_scene();

}  // namespace quantviz::viz
