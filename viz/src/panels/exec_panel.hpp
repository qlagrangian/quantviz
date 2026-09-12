#pragma once
// viz/panels/exec_panel.hpp — シーン「Optimal execution」（Almgren–Chriss）の描画パネル群
//
// 責務: Runner から `ExecSnapshot`（リング）と `ExecSurface`（TripleBuffer）を受け取り、
// 4 つのウィンドウに描く。Model には触らない（Snapshot / Surface の純関数 + Command の生成だけ）。
//
//   1. "Trajectories"                 — λ の梯子 8 本 + 現在の λ（太線）。再生ヘッド t に縦線とマーカー
//   2. "Frontier"                     — 効率的フロンティア E[C] vs V[C]（λ 固定格子 32 点）と現在の λ
//   3. "Inventory surface x(t, lambda)" — 面を `gl::SurfaceMesh` に載せて `SurfaceView` で回す
//   4. "Control"                      — λ / η / γ / σ / T・視点・時計・テレメトリ
//
// 【λ 軸は対数】メッシュの y 軸には λ ではなく **log10(λ)** を渡す: λ は 3 桁にわたるので線形の
// まま渡すと格子の 9 割が潰れて、面が「端の 1 本だけ立っている壁」に見える。2D のフロンティアも
// 同じ理由で V[C] 軸を Log10 にする（λ を 6 桁振ると分散も 6 桁動く）。
//
// 【履歴を持たない】このシーンは時系列を描かない（軌道は時刻 t の関数ではなくプログラムの形）。
// 毎フレーム最新の Snapshot 1 枚と最新の面 1 枚だけを描くので、`History` も、他パネルが持つ
// `prev_seq_` の巻き戻しガードも要らない（Reset で捨てるべき蓄積状態がそもそも無い）。
//
// 【一時停止中の操作】スライダーを動かすとモデルはその場で全部を引き直す。bridge の R10 により
// ステップが走らない tick でも Snapshot と面が 1 組 publish されるので、一時停止中でも次のフレームで
// 絵が入れ替わる。待機表示が出るのは起動直後だけ。
//
// GL が使えない環境でも viewer は落とさない（3D ウィンドウがテキスト表示に落ちるだけ）。

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "gl/surface_renderer.hpp"
#include "gl/surface_view.hpp"
#include "quantviz/bridge/runner.hpp"
#include "quantviz/core/exec/almgren_chriss.hpp"
#include "quantviz/scenes/exec_model.hpp"
#include "quantviz/viz/clock_controls.hpp"
#include "quantviz/viz/gl/surface_mesh.hpp"
#include "quantviz/viz/rate_meter.hpp"
#include "quantviz/viz/scene_registry.hpp"

namespace quantviz::viz {

class ExecPanel {
public:
    /// SnapCap は make_exec_scene() の RunnerScene と一致させること（Snapshot が ≈ 5.6 KB）。
    using Runner = bridge::Runner<scenes::ExecModel, 64, 256>;

    ExecPanel(const core::AcParams& initial, double initial_speed);

    /// 毎フレーム 1 回。poll → 描画 → Command 送信。
    void draw(Runner& runner);

private:
    using Snap = scenes::ExecSnapshot;
    using Surf = scenes::ExecSurface;

    void ingest(Runner& runner);
    void rebuild_curves();  ///< t 軸とフロンティアの (x, y) を最新 Snapshot から張り直す
    void rebuild_mesh();    ///< 最新の面をメッシュへ（軸 → z → 法線）
    void draw_trajectories();
    void draw_frontier();
    void draw_surface();
    void draw_controls(Runner& runner);

    Snap          last_{};
    Surf          surface_{};
    /// 受信した Snapshot の累計。`RateMeter` に渡すので**単調増加**であること（巻き戻すとレートが
    /// 符号なしで一周する）。0 は「まだ 1 枚も受け取っていない」= 待機表示の条件でもある。
    std::uint64_t received_ = 0;
    std::uint64_t surfaces_ = 0;
    RateMeter     rate_;

    bool have_surface_ = false;
    bool mesh_dirty_   = false;  ///< VBO 未反映の更新があるか（upload できたフレームだけ下ろす）
    bool init_tried_   = false;  ///< レンダラ初期化は 1 回だけ試す

    gl::SurfaceMesh mesh_{Surf::kT, Surf::kL};  ///< x = t（kT 点）, y = log10 λ（kL 点）
    SurfaceRenderer renderer_;
    SurfaceView     view_;

    std::array<float, Surf::kL>           mesh_y_{};      ///< メッシュの y 軸（= log10 λ、昇順）
    std::array<double, Snap::kN>          traj_t_{};      ///< 軌道の t 軸（0 … T）
    std::array<double, Snap::kFrontier>   frontier_v_{};  ///< フロンティアの V[C]
    std::array<double, Snap::kFrontier>   frontier_e_{};  ///< フロンティアの E[C]
    std::array<double, Snap::kL>          ladder_v_{};    ///< 梯子 8 本の V[C]（フロンティア上の点）
    std::array<double, Snap::kL>          ladder_e_{};    ///< 梯子 8 本の E[C]

    // UI 状態（ImGui のスライダーは float）
    //
    // 【λ と η は log10 で持つ】ImGui の対数スライダー（ImGuiSliderFlags_Logarithmic）は、書式の
    // 小数桁から作る logarithmic_zero_epsilon = 0.1^precision より**小さい端**を、その epsilon に
    // 丸めてしまう（imgui_widgets.cpp の ScaleRatioFromValueT）。λ ∈ [1e-9, 1e-3] は両端とも
    // epsilon（"%.2e" なら 0.01）未満なので、両端が同じ値に潰れてスライダーが死ぬ。そこで
    // **指数そのもの**を線形スライダーで持ち、Command を作るときに 10^x へ戻す。
    ClockControlState clock_{};
    float             lambda_log10_;
    float             eta_log10_;
    float             gamma_;
    float             sigma_;
    float             horizon_;  ///< T（"T" は Param の名前と紛れるので UI 側はこの名前）
};

/// main.cpp / SceneRegistry 用のファクトリ。
std::unique_ptr<Scene> make_exec_scene();

}  // namespace quantviz::viz
