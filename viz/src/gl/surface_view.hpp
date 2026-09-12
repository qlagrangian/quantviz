#pragma once
// viz/src/gl/surface_view.hpp — `SurfaceRenderer` の絵を ImGui のウィジェットとして置く（viz 層）
//
// 責務: 「FBO のテクスチャを `ImGui::Image` で貼る」＋「その矩形の上のマウス操作を軌道カメラの
// 回転・ズーム・パンに翻訳する」だけ。カメラの状態（角度・距離・注視点）と表示オプション
// （等高線の本数・ワイヤフレーム）はこの構造体が持ち、パネルは 1 行呼ぶだけでよい。
//
// 入力の受け方: 画像を貼ったあと、同じ矩形の上に `InvisibleButton` を重ねる。ImGui は
// 「id を持つアイテムが hover / active でないとき」にウィンドウ移動を始めるので、id を持たない
// `Image` だけだとドラッグがウィンドウ移動に化ける。透明ボタンが id を持つことでそれを止め、
// ついでにドラッグ中はポインタが矩形の外に出ても active が続く（= 回転が途切れない）。
//
// 操作: 左ドラッグ = 回転（掴んで回す向き）、右ドラッグ = 注視点の平行移動（視野平面内）、
// ホイール = ズーム、R = 視点リセット。

#include <cmath>
#include <cstdint>
#include <type_traits>

#include <imgui.h>

#include "gl/surface_renderer.hpp"
#include "quantviz/viz/gl/camera.hpp"
#include "quantviz/viz/gl/surface_mesh.hpp"

namespace quantviz::viz {

struct SurfaceView {
    /// 回転感度（rad / px）。1 ドラッグ 200 px で約 90 度。
    static constexpr float kRotatePerPixel = 0.008f;
    /// ホイール 1 ノッチあたりの距離倍率（上回し = 近づく）。
    static constexpr float kZoomPerNotch = 0.88f;
    /// これより小さい描画領域は描かない（FBO を作る意味がない）。
    static constexpr int kMinSize = 16;

    gl::OrbitCamera camera{};        ///< 現在の視点
    gl::OrbitCamera home{};          ///< R キーで戻る視点（パネルが好みの初期視点を入れておく）
    int             contour_lines = 12;
    bool            wire          = false;

    /// 1 フレーム分: リサイズ → 頂点更新 → FBO へ描画 → Image + 入力。
    /// `size` は描画領域（ふつうは `ImGui::GetContentRegionAvail()`）で、単位は **ポイント**。
    /// `mesh_dirty` が false なら VBO の更新を省く（前フレームと同じ面を回しているだけのとき）。
    ///
    /// 返値: このフレームで実際に `upload()` したか。呼び手はこれが true になったときだけ
    /// 自分の dirty フラグを下ろす（描かなかったフレームで下ろすと、その更新が永久に失われる）。
    [[nodiscard]] bool draw(SurfaceRenderer& renderer, const gl::SurfaceMesh& mesh, ImVec2 size,
                            float z_min, float z_max, bool mesh_dirty = true) {
        if (!renderer.ready()) {
            ImGui::TextDisabled("3D renderer is not initialised");
            return false;
        }
        const int w_pt = static_cast<int>(size.x);
        const int h_pt = static_cast<int>(size.y);
        if (w_pt < kMinSize || h_pt < kMinSize) {
            ImGui::TextDisabled("window too small for the 3D view");
            return false;
        }

        // HiDPI: FBO は物理ピクセルで確保し（そうしないと Retina 等でぼやける）、
        // `ImGui::Image` に渡す大きさとマウス座標はポイントのまま扱う。
        const ImGuiIO& io = ImGui::GetIO();
        const float    sx = io.DisplayFramebufferScale.x > 0.f ? io.DisplayFramebufferScale.x : 1.f;
        const float    sy = io.DisplayFramebufferScale.y > 0.f ? io.DisplayFramebufferScale.y : 1.f;
        renderer.resize(static_cast<int>(size.x * sx), static_cast<int>(size.y * sy));

        const bool uploaded = mesh_dirty;
        if (uploaded) renderer.upload(mesh);
        renderer.render(camera, z_min, z_max, contour_lines, wire);

        const ImVec2 p0 = ImGui::GetCursorScreenPos();
        const ImVec2 sz(static_cast<float>(w_pt), static_cast<float>(h_pt));
        // uv を上下反転する: GL のテクスチャは原点が左下、ImGui は左上。
        ImGui::Image(to_texture_id(renderer.texture()), sz, ImVec2(0.f, 1.f), ImVec2(1.f, 0.f));

        // 画像と同じ矩形に透明ボタンを重ねて入力を取る（レイアウトは進めない）。
        ImGui::SetCursorScreenPos(p0);
        ImGui::InvisibleButton("##surface_input", sz,
                               ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
        handle_input(ImGui::IsItemActive(), ImGui::IsItemHovered(), h_pt);

        ImGui::GetWindowDrawList()->AddText(ImVec2(p0.x + 8.f, p0.y + 6.f),
                                            IM_COL32(205, 205, 212, 190),
                                            "drag: rotate \xc2\xb7 wheel: zoom \xc2\xb7 R: reset");
        return uploaded;
    }

private:
    /// GL のテクスチャ名（GLuint）を `ImTextureID` に詰める。ImGui のバージョンで実体が
    /// `void*`（〜1.91 系）だったり整数（1.92 以降の `ImU64`）だったりするので、両方に対応する。
    /// テンプレートにしてあるのは `if constexpr` の捨てられる枝を実際に捨てさせるため。
    template <class Id = ImTextureID>
    [[nodiscard]] static Id to_texture_id(unsigned int tex) noexcept {
        if constexpr (std::is_pointer_v<Id>)
            return reinterpret_cast<Id>(static_cast<std::uintptr_t>(tex));
        else
            return static_cast<Id>(tex);
    }

    void handle_input(bool active, bool hovered, int viewport_h) {
        const ImGuiIO& io = ImGui::GetIO();

        if (active && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            const ImVec2 d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.f);
            if (d.x != 0.f || d.y != 0.f) {
                // 「面を掴んで回す」向き: 右へ引けば手前が右へ = カメラは左へ（yaw 減）。
                camera.rotate(-d.x * kRotatePerPixel, d.y * kRotatePerPixel);
                ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
            }
        }

        if (active && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            const ImVec2 d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right, 0.f);
            if (d.x != 0.f || d.y != 0.f) {
                pan(d.x, d.y, viewport_h);
                ImGui::ResetMouseDragDelta(ImGuiMouseButton_Right);
            }
        }

        if (hovered && io.MouseWheel != 0.f) camera.zoom(std::pow(kZoomPerNotch, io.MouseWheel));
        // R はテキスト入力中に拾わない: スライダーを Ctrl+クリックして "r" を打った瞬間に
        // 視点が飛ぶ、という事故を防ぐ（ImGui は入力欄が活きている間 WantTextInput を立てる）。
        if (hovered && !io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_R, false)) camera = home;
    }

    /// 注視点を視野平面内で動かす。1 ポイント = 画面高さから逆算したワールド距離なので、
    /// ズームしても「掴んだ点がポインタに付いてくる」感じが保たれる。
    /// `viewport_h` はマウスの移動量と同じ単位（ポイント）で渡すこと。
    void pan(float dx, float dy, int viewport_h) noexcept {
        const gl::Vec3 forward = gl::normalize(camera.target - camera.eye());
        const gl::Vec3 right   = gl::normalize(gl::cross(forward, gl::OrbitCamera::kUp));
        const gl::Vec3 up      = gl::cross(right, forward);
        const float    h       = static_cast<float>(viewport_h > 0 ? viewport_h : 1);
        // fovy は projection() と同じ安全化を通す（生の fovy が壊れていても平行移動が飛ばない）。
        const float    scale   = 2.f * camera.distance * std::tan(camera.safe_fovy() * 0.5f) / h;
        camera.target = camera.target + right * (-dx * scale) + up * (dy * scale);
    }
};

}  // namespace quantviz::viz
