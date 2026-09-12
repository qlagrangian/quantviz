#pragma once
// viz/src/gl/surface_renderer.hpp — `gl::SurfaceMesh` を FBO の中に描く OpenGL レンダラ（viz 層）
//
// 責務: 設計書 §9.3 の右半分。頂点（位置 + 法線）を VBO へ、三角形を EBO へ上げ、
// 「高さ色（viridis 風）× Lambert + 等高線」のシェーダで **オフスクリーンの FBO** に描く。
// 描いた結果はテクスチャなので、パネルは `ImGui::Image(texture(), ...)` で普通のウィジェットの
// ように置ける（ImGui のドローリストへ直接 GL を差し込まないので、描画順やクリップで悩まない）。
//
// 契約:
//   * `init(n_x, n_y)` は VAO/VBO/EBO/シェーダ/FBO を 1 回だけ作る。以後 GL オブジェクトは
//     `resize()` が張り直すカラーテクスチャ・深度バッファ以外は作り直さない。
//   * `upload(mesh)` は `glBufferSubData` で頂点を上書きするだけ（再確保なし）。インデックスは
//     格子サイズだけで決まるので最初の 1 回で確定する。
//   * `render()` は自分の FBO に描いたあと、**ImGui のバックエンドが前提にしている状態へ戻す**
//     （既定 FBO とビューポート、深度テストとポリゴンモード）。GL 状態を丸ごと保存・復元するの
//     ではなく、ImGui が当てにしているものだけを確実に元へ戻す、という約束であることに注意。
//   * ここは GUI スレッド（= GL コンテキストが current なスレッド）専用。計算スレッドは触らない。
//   * 寿命: GL オブジェクトを持つのでコピーもムーブもできない。デストラクタは GL を呼ぶので、
//     **GL コンテキストが生きているうちに破棄されなければならない**（main.cpp が SceneRegistry を
//     GUI シャットダウンより前のスコープで畳むことでそれを保証している）。
//
// 座標系: メッシュは (x, y, z=高さ)。描画ワールドは y が上なので、モデル行列が
// (x, y, z) → (x', z', y') と軸を入れ替えつつ、x/y は [−1, 1]、高さは [−kHeight, kHeight] に
// 正規化する（軸の実スケールが何桁違っても同じ画になる）。

#include <cstddef>
#include <string>

#include "quantviz/viz/gl/camera.hpp"
#include "quantviz/viz/gl/surface_mesh.hpp"

namespace quantviz::viz {

class SurfaceRenderer {
public:
    /// 正規化後の高さの半幅（ワールド単位）。x/y は ±1 なので、0.5 くらいが見やすい。
    static constexpr float kHeight = 0.5f;

    SurfaceRenderer() = default;
    ~SurfaceRenderer();

    // GL オブジェクト（の名前）を所有するのでコピーもムーブもしない。
    // ムーブを許すと「破棄された側がまだ生きている名前を消す」事故が入り込む。
    SurfaceRenderer(const SurfaceRenderer&)            = delete;
    SurfaceRenderer& operator=(const SurfaceRenderer&) = delete;
    SurfaceRenderer(SurfaceRenderer&&)                 = delete;
    SurfaceRenderer& operator=(SurfaceRenderer&&)      = delete;

    /// GL オブジェクトを作る。`gl_load()` 成功後・GL コンテキストが current なときに 1 回だけ。
    /// 失敗したら false（理由は `last_error()`）。すでに初期化済みなら何もせず true。
    /// `glsl_version` が nullptr なら `glapi::glsl_version()`（= main.cpp がコンテキスト生成時に
    /// 選んだ版）を使う。
    bool init(std::size_t n_x, std::size_t n_y, const char* glsl_version = nullptr);

    [[nodiscard]] bool ready() const noexcept { return program_ != 0; }

    /// 頂点（位置 + 法線）を上書きする。格子サイズが `init` と違うメッシュは無視する。
    void upload(const gl::SurfaceMesh& mesh);

    /// 描画先のサイズを変える（カラーテクスチャと深度バッファを作り直す）。同じサイズなら何もしない。
    void resize(int w, int h);

    /// FBO に 1 枚描く。`contour_lines <= 0` で等高線なし、`wire` でワイヤフレーム。
    void render(const gl::OrbitCamera& cam, float z_min, float z_max, int contour_lines,
                bool wire = false);

    /// `ImGui::Image` に渡すカラーテクスチャ（未初期化なら 0）。
    [[nodiscard]] unsigned int texture() const noexcept { return color_tex_; }

    [[nodiscard]] int width() const noexcept { return fb_w_; }
    [[nodiscard]] int height() const noexcept { return fb_h_; }

    /// 直近の失敗理由（シェーダのコンパイル／リンクログなど）。成功していれば空。
    [[nodiscard]] const std::string& last_error() const noexcept { return error_; }

private:
    bool build_program(const char* glsl_version);
    void destroy();

    // 格子サイズ（init で固定）
    std::size_t n_x_ = 0;
    std::size_t n_y_ = 0;

    // GL オブジェクト（GLuint = unsigned int。ここでは GL ヘッダを持ち込まない）
    unsigned int vao_         = 0;
    unsigned int vbo_         = 0;
    unsigned int ebo_         = 0;
    unsigned int program_     = 0;
    unsigned int fbo_         = 0;
    unsigned int color_tex_   = 0;
    unsigned int depth_rb_    = 0;

    int fb_w_ = 0;
    int fb_h_ = 0;

    // uniform location（-1 = そのシェーダが使っていない。GL 的には黙って無視される）
    int u_mvp_     = -1;
    int u_model_   = -1;
    int u_light_   = -1;
    int u_z_min_   = -1;
    int u_z_max_   = -1;
    int u_contour_ = -1;

    int  index_count_  = 0;      ///< EBO に入っている index の個数（0 = まだ上げていない）
    bool fbo_ok_       = false;  ///< 直近の resize() で FBO が complete になったか
    bool error_logged_ = false;  ///< glGetError の報告は 1 回だけ（毎フレーム出さない）

    // upload() が見たメッシュの x/y 範囲（モデル行列の正規化に使う）
    float x_min_ = 0.f, x_max_ = 1.f;
    float y_min_ = 0.f, y_max_ = 1.f;

    std::string error_;
};

}  // namespace quantviz::viz
