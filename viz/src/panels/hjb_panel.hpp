#pragma once
// viz/panels/hjb_panel.hpp — シーン「Merton HJB」の描画パネル群
//
// 責務: Runner から `HjbSnapshot`（リング）と `HjbSurface`（TripleBuffer）を受け取り、4 つのウィンドウに
// 描く。Model には触らない（Snapshot / Surface の純関数 + Command の生成だけ）。
//
//   1. "V(w, t) surface (3D)" — 面を `gl::SurfaceMesh` に載せて `SurfaceView` で回す
//   2. "V(w) now"             — 現在の時刻断面の V(w)。数値解と閉形式 `merton_value` の重ね描き
//   3. "pi*(w)"               — 数値解 π*(w) と解析定数 (μ−r)/(γσ²)（制約域に丸めたもの）の重ね描き
//   4. "Control"              — μ / r / σ / γ・時計・テレメトリ
//
// 【時間の向き】Snapshot の `t_remaining` と Surface の `times` は「掃引が今日に届くまでの残り年数 t」で、
// 満期が T、解き終わりが 0（`hjb_model.hpp` 参照）。残存期間は τ = T − t。3D のメッシュ y 軸には **−t**
// を渡す: 昇順になり（`SurfaceMesh` の前提）、かつ t = 0（今日）が手前に来る。FDM シーンと同じ規約。
//
// 【w 軸は対数】富の格子は対数等間隔なので、メッシュの x 軸には **ln w** を渡す（列が等間隔に並ぶ）。
// 2D の 2 枚も X 軸を対数スケールにする。線形軸だと格子の 8 割が右端に潰れて、w が小さい側で何が
// 起きているか（V の発散、π* の境界誤差）がまったく読めない。
//
// 【V のスケール】V は γ > 1 で負（U = w^{1−γ}/(1−γ) < 0）、しかも w_min 側で桁違いに大きい
// （既定の γ = 3 では U(0.2) = −12.5 に対し U(5) = −0.02 で 600 倍）。そこで 2 段構えにする:
//   1. レンジは実測 min/max ではなく、**閉形式の四隅**（w ∈ {w_min, w_max} × t ∈ {0, T}）と 0 から作った
//      固定値にする。面が伸びるたびに色の対応が変わると、どこが高いのか読めなくなる。V は w にも τ にも
//      単調なので四隅が実際の上下限になり、0 を含めるのは未計算行（ゼロ）を範囲内に収めるため。
//      レンジはパラメータが変わったフレームだけ張り直す（= 再 init のたびに auto-fit）。
//   2. 高さと色には既定で**符号付き対数** h(V) = sign(V)·ln(1 + |V|) を掛ける。線形のままだと 600 倍の
//      崖が全レンジを食い、残り 99 % の w では「解けた領域」と「未計算のゼロ平面」が同じ色に潰れて、
//      掃引がどこまで来たのかまったく見えない（これがこのシーンで一番見せたいものなのに）。h は狭義単調
//      なので大小関係も等高線の順序も保たれる。線形が見たいときは Control の "log height (3D)" を外す。
//
// 【未計算行】モデルは未計算の行をゼロで埋めて寄越す（FDM シーンと同じ規則）が、V < 0 のときその
// ゼロ平面は計算済みの領域の**上**に来て、斜め上から見ると解を覆い隠してしまう。そこで描画では
// 未計算行をスケールの下端（最も暗い色）に落とす。「まだ無い」ことが一目で分かり、かつデータを
// 隠さない。前方へ押し出す（extrude）案を採らないのは FDM シーンと同じ理由: まだ解いていない
// 領域に解があるように見えてしまうから。
//
// 【メッシュ作り直しの省略】掃引が終わったあとも Runner は同じ面を 20 枚/秒で publish し続ける
// （step は seq を進めるので面も毎回出る）。面の中身は「行数（filled_rows）が増える」か「パラメータが
// 変わる（= SetParam が必ず init をやり直し、行数もレンジも動く）」以外では変わらない — 一度埋まった行
// が載せる時間レベルは固定で、あとから書き換わらないからである。そこで `filled_rows` と `ScaleKey` が
// 前回と同じなら 40,000 頂点ぶんの符号付き対数 + 法線計算をまるごと省く（VBO も送り直さない）。
// 高さの写像を切り替えたときだけは、面が届くのを待たずにその場で作り直す。
//
// 【一時停止中の操作】Reset / パラメータ変更は掃引を満期へ巻き戻す。bridge の R10 により、ステップが
// 走らない tick でも Snapshot と面が 1 組 publish されるので、一時停止中でも次のフレームで新しい状態に
// 入れ替わる。待機表示が出るのは「まだ 1 枚も受け取っていない」起動直後だけ。
//
// GL が使えない環境でも viewer は落とさない（3D ウィンドウがテキスト表示に落ちるだけ）。

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "gl/surface_renderer.hpp"
#include "gl/surface_view.hpp"
#include "quantviz/bridge/runner.hpp"
#include "quantviz/scenes/hjb_model.hpp"
#include "quantviz/viz/clock_controls.hpp"
#include "quantviz/viz/gl/surface_mesh.hpp"
#include "quantviz/viz/rate_meter.hpp"
#include "quantviz/viz/scene_registry.hpp"

namespace quantviz::viz {

class HjbPanel {
public:
    /// SnapCap は make_hjb_scene() の RunnerScene と一致させること（Snapshot が ≈ 10 KB）。
    using Runner = bridge::Runner<scenes::HjbModel, 64, 256>;

    HjbPanel(const scenes::HjbModel::Config& initial, double initial_speed);

    /// 毎フレーム 1 回。poll → 描画 → Command 送信。
    void draw(Runner& runner);

private:
    using Snap = scenes::HjbSnapshot;
    using Surf = scenes::HjbSurface;

    /// 3D の色スケールと 2D の X 範囲を決める「パラメータの組」。これが動いたら掃引はやり直しなので、
    /// レンジも張り直す（= auto-fit のタイミング）。
    struct ScaleKey {
        double mu    = 0.0;
        double r     = 0.0;
        double sigma = 0.0;
        double gamma = 0.0;
        double w_min = 0.0;
        double w_max = 0.0;

        friend bool operator==(const ScaleKey&, const ScaleKey&) = default;
    };

    void ingest(Runner& runner);
    void rebuild_scale();  ///< 閉形式の四隅から 3D の z レンジを張り直す
    void rebuild_mesh();   ///< 最新の面をメッシュへ（軸 → z → 法線）

    /// 3D の高さ・色に使う値。`log_height_` が立っていれば符号付き対数（ヘッダ「V のスケール」）。
    [[nodiscard]] float height_of(float v) const noexcept;
    void draw_surface();
    void draw_value();
    void draw_policy();
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
    std::uint64_t prev_seq_  = 0;  ///< 巻き戻し検出（seq が**厳密に**減ったら面を捨てる。R10 の同 seq 再送は除く）
    std::uint32_t prev_iter_ = 0;
    RateMeter     rate_;

    bool have_surface_ = false;  ///< 有効な面を持っているか（掃引が巻き戻った直後だけ false）
    bool mesh_dirty_   = false;  ///< VBO 未反映の更新があるか（upload できたフレームだけ下ろす）
    bool init_tried_   = false;  ///< レンダラ初期化は 1 回だけ試す
    bool mesh_valid_   = false;  ///< メッシュが今の面を写しているか（下の「作り直しの省略」）

    gl::SurfaceMesh mesh_{Surf::kW, Surf::kT};
    SurfaceRenderer renderer_;
    SurfaceView     view_;

    std::array<float, Surf::kT>      mesh_y_{};  ///< メッシュの y 軸（= −t、昇順）
    std::array<float, Surf::kW>      mesh_x_{};  ///< メッシュの x 軸（= ln w、昇順）
    /// 高さに渡す値（生の V か符号付き対数。未計算行はスケール下端）。
    std::array<float, Surf::kW * Surf::kT> mesh_z_{};

    ScaleKey scale_key_{};       ///< 現在のレンジを作ったときのパラメータ
    bool     have_scale_ = false;
    /// 現在のメッシュを作った時点の面の状態（下の「作り直しの省略」）。
    ScaleKey      mesh_key_{};
    std::uint32_t mesh_rows_ = 0;
    float    z_lo_       = 0.f;  ///< 3D の色 / 高さスケール
    float    z_hi_       = 1.f;

    // UI 状態（ImGui のスライダーは float）
    ClockControlState clock_{};
    float             mu_;
    float             r_;
    float             sigma_;
    float             gamma_;
    bool              pi_zoom_    = false;  ///< π* の Y 軸を数値解のばらつきに合わせる（既定は固定範囲）
    bool              log_height_ = true;   ///< 3D の高さ・色を符号付き対数にする（ヘッダ「V のスケール」）
};

/// main.cpp / SceneRegistry 用のファクトリ。
std::unique_ptr<Scene> make_hjb_scene();

}  // namespace quantviz::viz
