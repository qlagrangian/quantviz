#pragma once
// viz/panels/greeks_panel.hpp — シーン 2「Greeks」の描画パネル群
//
// 責務: Runner から Snapshot を poll → 最新 1 枚だけを保持 → ImPlot で描く → UI 操作を Command で返す。
// Model には一切触らない（Snapshot の純関数 + Command の生成のみ）。
// このシーンは時系列ではなく「今の断面」（K 軸の Greeks ストリップと Γ(S,T) 格子）を見るので、
// Streaming と違って History は持たない。Snapshot が大きい（16 KB 強）ので Runner の SnapCap は 256。
//
// Γ 面は 1 つのウィンドウの中でタブに分けて 2 通りに見せる（M2 Task 8）:
//   * 「3D」  : `gl::SurfaceMesh` + `SurfaceView`（x = S, y = T, z = Γ）。尾根の形と広がりが読める
//         （高さは z 範囲に正規化されるので、絶対値は Telemetry の gamma z range を見る）。
//   * 「heatmap」: 既存の ImPlot ヒートマップ。値を面で読むならこちらが速い。
// ウィンドウを増やさずタブにしたのは、左側の縦 760 px にストリップ・面・（右の）Control が
// すでに収まっており、3 段に割ると 3D の描画領域が 200 px を切って尾根が潰れるため。
//
// 面の GPU 転送は「パラメータが変わったフレームだけ」。Snapshot は 100 steps/s で流れてくるが、
// Γ(S,T) は (r, sigma, T) にしか依存しない（`GreeksModel::commit_pending` が同じ条件で面を
// 張り直す）ので、最後に上げた 3 つと比べて変化したときだけメッシュを作り直して VBO に上げる。

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "gl/surface_renderer.hpp"
#include "gl/surface_view.hpp"
#include "quantviz/bridge/runner.hpp"
#include "quantviz/scenes/greeks_model.hpp"
#include "quantviz/viz/clock_controls.hpp"
#include "quantviz/viz/gl/surface_mesh.hpp"
#include "quantviz/viz/rate_meter.hpp"
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
    void draw_surface();          ///< タブ 1 枚のウィンドウ（3D / heatmap）
    void draw_surface_3d();       ///< 「3D」タブの中身
    void draw_surface_heatmap();  ///< 「heatmap」タブの中身
    void draw_controls(Runner& runner);

    /// Snapshot の Γ 格子を `mesh_` に載せ直す（転置 + float 化 + 法線 + z 範囲）。
    /// 呼ぶのは (r, sigma, T) が変わったフレームだけ。`gpu_dirty_` を立てて VBO 更新を予約する。
    void rebuild_mesh();

    /// spot に最も近いストライクの index（軸は等間隔なので丸めで求まる）。
    std::size_t atm_index() const noexcept;

    Snap          last_{};
    std::uint64_t received_ = 0;
    RateMeter     rate_;  ///< 受信 Snapshot/秒（表示用）
    // 他の 3 パネルが持つ `prev_seq_`（Reset をまたぐ古い Snapshot の検出）はここには要らない:
    // このシーンは History を持たず、毎フレーム最新の 1 枚だけを描くので、巻き戻りで捨てる
    // 蓄積状態がそもそも無い（古い Snapshot が 1 枚描かれた次のフレームには新しい 1 枚に入れ替わる）。

    /// ImPlot のヒートマップは row-major で「row 0 を上端」に描く。Snapshot は [iS][iT] なので、
    /// 表示用に [iT 降順][iS 昇順] へ転置したコピーを毎フレーム作る（48x32 = 1536 要素）。
    std::array<double, Snap::kGridS * Snap::kGridT> heat_{};

    // ---- 3D タブ（x = S = grid_s, y = T = grid_t, z = Γ）
    /// `SurfaceMesh` の走査順は y メジャー（index = iy * n_x + ix）。Snapshot は [iS][iT] なので
    /// `rebuild_mesh()` で転置して float 化する。確保はここ（コンストラクタ）の 1 回きり。
    gl::SurfaceMesh mesh_{Snap::kGridS, Snap::kGridT};
    SurfaceRenderer renderer_;
    SurfaceView     view_;

    std::array<float, Snap::kGridS>                mesh_xs_{};  ///< S 軸（float 化した grid_s）
    std::array<float, Snap::kGridT>                mesh_ys_{};  ///< T 軸（float 化した grid_t）
    std::array<float, Snap::kGridS * Snap::kGridT> mesh_z_{};   ///< Γ を [iT][iS] に転置したもの

    float         z_min_      = 0.f;    ///< 色・高さの下限（データから）
    float         z_max_      = 1.f;    ///< 同上限
    double        surf_r_     = 0.0;    ///< 最後にメッシュを作った Snapshot の r
    double        surf_sigma_ = 0.0;    ///< 同 sigma
    double        surf_T_     = 0.0;    ///< 同 T
    bool          surf_valid_ = false;  ///< 一度でもメッシュを作ったか（初回は必ず作る）
    bool          gpu_dirty_  = false;  ///< CPU 側のメッシュが VBO より新しい（次の描画で上げる）
    bool          init_tried_ = false;  ///< レンダラ初期化は 1 回だけ試す（失敗を毎フレーム繰り返さない）
    std::uint64_t uploads_    = 0;      ///< VBO へ上げた回数（Telemetry「surface uploads」）

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
