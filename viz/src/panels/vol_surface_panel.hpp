#pragma once
// viz/panels/vol_surface_panel.hpp — シーン「Vol surface」: SSVI インプライド・ボラ面の描画パネル群
//
// 責務: Runner から Snapshot（パラメータ・時刻）と Surface（64×32 の IV 格子）を取り、
// 3D（`SurfaceView`）と 2D のスマイル断面（ImPlot）で描き、UI 操作を Command で返す。
// Model には触らない（Snapshot / Surface の純関数 + Command の生成のみ）。
//
// 面は Snapshot リングではなく `Runner::poll_surface`（TripleBuffer）で来る。新しい面が届いた
// フレームだけメッシュを組み直し、VBO への upload も実際に描いたフレームだけ行う。
//
// GL が使えない環境（`gl_load()` 失敗）でも viewer は落とさない: 3D ウィンドウは
// 「OpenGL functions unavailable」のテキストに落ちるだけで、スマイル・時計・Telemetry は動き続ける。

#include <cstddef>
#include <cstdint>
#include <memory>

#include "gl/surface_renderer.hpp"
#include "gl/surface_view.hpp"
#include "quantviz/bridge/runner.hpp"
#include "quantviz/scenes/vol_surface_model.hpp"
#include "quantviz/viz/clock_controls.hpp"
#include "quantviz/viz/gl/surface_mesh.hpp"
#include "quantviz/viz/rate_meter.hpp"
#include "quantviz/viz/scene_registry.hpp"

namespace quantviz::viz {

class VolSurfacePanel {
public:
    /// SnapCap は make_vol_surface_scene() の RunnerScene と一致させること。
    using Runner = bridge::Runner<scenes::VolSurfaceModel, 256, 256>;

    VolSurfacePanel(const scenes::VolSurfaceModel::Config& initial, double initial_speed);

    /// 毎フレーム 1 回。poll → 描画 → Command 送信。
    void draw(Runner& runner);

private:
    using Snap = scenes::VolSurfaceSnapshot;
    using Surf = scenes::VolSurfaceSurface;

    void ingest(Runner& runner);
    void draw_surface();
    void draw_smile();
    void draw_controls(Runner& runner);

    /// 面の最小・最大から色と等高線のスケールを作る（データ由来 + 少しの余白）。
    void refresh_z_range() noexcept;

    Snap          last_{};      ///< 最新 Snapshot（パラメータ・時刻）
    Surf          surface_{};   ///< 最新の面（TripleBuffer から取り込んだコピー）
    std::uint64_t received_ = 0;  ///< 受け取った Snapshot の総数
    std::uint64_t surfaces_ = 0;  ///< 受け取った面の総数（publish 数とは違う: 古い面は捨てられる）
    RateMeter     rate_;
    // 他パネルの `prev_seq_`（Reset をまたぐ古い Snapshot の検出）は要らない: このシーンは
    // History を持たず毎フレーム最新の 1 枚だけを描くので、巻き戻りで捨てる蓄積状態が無い。

    gl::SurfaceMesh mesh_{Surf::kK, Surf::kT};  ///< x = k（kK 点）, y = T（kT 点）
    SurfaceRenderer renderer_;
    SurfaceView     view_;

    bool  axes_set_   = false;  ///< 軸は Config で固定なので 1 回だけ流し込む
    bool  mesh_dirty_ = false;  ///< 未 upload の頂点があるか（実際に描けたフレームだけ下ろす）
    bool  init_tried_ = false;  ///< レンダラの初期化を 1 回だけ試す
    float z_min_      = 0.0f;   ///< 色・等高線のスケール（面の実測 ± 余白）
    float z_max_      = 1.0f;

    int smile_index_ = 0;  ///< 「Smile @ T」で描く T 軸の index（0 〜 kT−1）

    // UI 状態（ImGui のスライダーは float）
    ClockControlState clock_{};
    float             sigma_atm_;
    float             rho_;
    float             eta_;
    float             gamma_;
};

/// main.cpp / SceneRegistry 用のファクトリ。
std::unique_ptr<Scene> make_vol_surface_scene();

}  // namespace quantviz::viz
