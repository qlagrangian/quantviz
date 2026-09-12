#pragma once
// viz/panels/fdm_panel.hpp — シーン「FDM American」の描画パネル群
//
// 責務: Runner から `FdmSceneSnapshot`（リング）と `FdmSceneSurface`（TripleBuffer）を受け取り、
// 4 つのウィンドウに描く。Model には触らない（Snapshot / Surface の純関数 + Command の生成だけ）。
//
//   1. "V(S, t) surface (3D)"     — 面を `gl::SurfaceMesh` に載せて `SurfaceView` で回す
//   2. "V(S) now"                 — 現在の時刻断面。V(S)・本源的価値・European 参照（BS）・S* と S0 の縦線
//   3. "Exercise boundary S*(t)"  — 反復ごとの S* を τ（満期までの残存期間）軸で。3D 側に線を引けない
//                                   （レンダラは三角形しか描かない）ので、2D の相方として独立させる
//   4. "Control"                  — K / r / σ / q・American/European・Call/Put・ω・時計・テレメトリ
//
// 【時間の向き】Snapshot の `t_remaining` と Surface の `times` は「掃引が今日に届くまでの残り年数 t」で、
// 満期が T、解き終わりが 0（`fdm_american_model.hpp` 参照）。満期までの残存期間は τ = T − t。
// 3D のメッシュ y 軸には **−t** を渡す: 昇順になり（`SurfaceMesh` の前提）、かつ t = 0（今日）が
// 手前に来る。2D の行使境界は教科書と同じ τ 軸で描く（τ → 0 で S* → K に寄るのが見える）。
//
// 【未計算行】モデルは未計算の行をゼロで埋めて寄越す。パネルはそれをそのまま描く: ゼロ平面から
// 楔形に伸びる形が「後ろ向き反復がどこまで来たか」を一目で見せる（最後の行を前へ押し出すと、
// まだ解いていない領域に解があるように見えてしまう）。
//
// 【一時停止中の操作】Reset / パラメータ変更は掃引を満期へ巻き戻す。bridge の R10 により、
// ステップが走らない tick でも Snapshot と面が 1 組 publish されるので、一時停止中でも次のフレームで
// 新しい状態に入れ替わる。待機表示が出るのは「まだ 1 枚も受け取っていない」起動直後だけ。
//
// GL が使えない環境でも viewer は落とさない（3D ウィンドウがテキスト表示に落ちるだけ）。

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "gl/surface_renderer.hpp"
#include "gl/surface_view.hpp"
#include "quantviz/bridge/runner.hpp"
#include "quantviz/scenes/fdm_american_model.hpp"
#include "quantviz/viz/clock_controls.hpp"
#include "quantviz/viz/gl/surface_mesh.hpp"
#include "quantviz/viz/rate_meter.hpp"
#include "quantviz/viz/scene_registry.hpp"

namespace quantviz::viz {

class FdmPanel {
public:
    /// SnapCap は make_fdm_scene() の RunnerScene と一致させること（Snapshot が ≈ 6 KB）。
    using Runner = bridge::Runner<scenes::FdmAmericanModel, 64, 256>;

    FdmPanel(const scenes::FdmAmericanModel::Config& initial, double initial_speed);

    /// 毎フレーム 1 回。poll → 描画 → Command 送信。
    void draw(Runner& runner);

private:
    using Snap = scenes::FdmSceneSnapshot;
    using Surf = scenes::FdmSceneSurface;

    void ingest(Runner& runner);
    void rebuild_curves();   ///< 本源的価値と European 参照を今の Snapshot のパラメータで張り直す
    void rebuild_mesh();     ///< 最新の面をメッシュへ（軸 → z → 法線）
    void draw_surface();
    void draw_curve();
    void draw_boundary();
    void draw_controls(Runner& runner);

    /// 手元の面を捨てる（新しい Snapshot が古い面と食い違ったとき）。
    void invalidate_surface() noexcept;

    Snap          last_{};
    Surf          surface_{};
    /// 受信した Snapshot の累計。`RateMeter` に渡すだけなので**単調増加**でなければならない
    /// （巻き戻すと Δ が符号なしで一周して、レートが 1e19 件/秒になる）。0 は「まだ 1 枚も
    /// 受け取っていない」＝待機表示の唯一の条件でもある。
    std::uint64_t received_  = 0;
    std::uint64_t surfaces_  = 0;
    std::uint32_t prev_iter_ = 0;  ///< 巻き戻し検出用（iteration が減ったら面を捨てる）
    RateMeter     rate_;

    bool have_surface_ = false;  ///< 有効な面を持っているか（掃引が巻き戻った直後だけ false）
    bool mesh_dirty_   = false;  ///< VBO 未反映の更新があるか（upload できたフレームだけ下ろす）
    bool init_tried_   = false;  ///< レンダラ初期化は 1 回だけ試す

    gl::SurfaceMesh mesh_{Surf::kS, Surf::kT};
    SurfaceRenderer renderer_;
    SurfaceView     view_;

    std::array<float, Surf::kT>  mesh_y_{};      ///< メッシュの y 軸（= −t、昇順）
    std::array<double, Snap::kCurve> intrinsic_{};   ///< 本源的価値
    std::array<double, Snap::kCurve> european_{};    ///< 同じパラメータの BS European 価格
    std::array<double, Snap::kCurve> boundary_x_{};  ///< 行使境界の τ 軸

    // UI 状態（ImGui のスライダーは float）
    ClockControlState clock_{};
    float             strike_;
    float             r_;
    float             sigma_;
    float             q_;
    float             omega_;
    bool              american_;
    int               type_index_;  ///< 0 = Call, 1 = Put（core::OptionType の並び）
};

/// main.cpp / SceneRegistry 用のファクトリ。
std::unique_ptr<Scene> make_fdm_scene();

}  // namespace quantviz::viz
