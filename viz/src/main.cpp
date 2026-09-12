// viz/main.cpp — quantviz viewer entry point
//
//   描画スレッド（このファイル）: GLFW + ImGui + ImPlot, 60 fps
//   計算スレッド（Runner）      : 選択中のシーンの Model を steps_per_second で回す
//   両者は Snapshot リングと Command リングだけで繋がる。
//
// シーンは SceneRegistry に登録し、メニューバーの "Scene" で切り替える。同時に走るシーンは 1 つ。
// 各シーンのウィンドウはメニューバー（高さ約 22 px）の下、y >= 32 に置く（各パネルの FirstUseEver 位置）。

#include <cstddef>
#include <cstdio>

#if defined(__APPLE__)
#define GL_SILENCE_DEPRECATION
#endif
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

#include "gl/gl_loader.hpp"
#include "panels/garch_panel.hpp"
#include "panels/greeks_panel.hpp"
#include "panels/kalman_panel.hpp"
#include "panels/streaming_panel.hpp"
#include "panels/vol_surface_panel.hpp"
#include "quantviz/viz/scene_registry.hpp"

namespace {

void glfw_error_callback(int error, const char* description) {
    std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

/// メニューバー: シーン選択 + 現在のシーン名 + フレームレート。
void draw_menu_bar(quantviz::viz::SceneRegistry& registry) {
    using quantviz::viz::SceneRegistry;
    if (!ImGui::BeginMainMenuBar()) return;
    const std::size_t cur = registry.current_index();
    if (ImGui::BeginMenu("Scene")) {
        for (std::size_t i = 0; i < registry.size(); ++i) {
            const bool selected = (i == cur);
            if (ImGui::MenuItem(registry.names()[i].c_str(), nullptr, selected) && !selected) {
                if (!registry.select(i)) std::fprintf(stderr, "scene %zu failed to start\n", i);
            }
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Restart current scene") && cur != SceneRegistry::kNone) {
            if (!registry.select(cur)) std::fprintf(stderr, "scene %zu failed to restart\n", cur);
        }
        ImGui::EndMenu();
    }
    if (cur != SceneRegistry::kNone) ImGui::TextDisabled("| %s", registry.names()[cur].c_str());
    ImGui::TextDisabled("| %.1f fps", static_cast<double>(ImGui::GetIO().Framerate));
    ImGui::EndMainMenuBar();
}

}  // namespace

int main() {
    // ------------------------------------------------------------------ window / GL context
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) return 1;

#if defined(__APPLE__)
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif

    GLFWwindow* window = glfwCreateWindow(1220, 760, "quantviz", nullptr, nullptr);
    if (window == nullptr) {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // vsync: 描画は 60 fps、計算はそれと独立

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // 3D 用の GL 関数を自前ロードする（設計書 §12.1）。失敗しても viewer は動かし続ける:
    // 3D パネルは `glapi::gl_available()` を見て「OpenGL functions unavailable」表示に落ちる。
    if (!quantviz::viz::glapi::gl_load())
        std::fprintf(stderr, "gl_load() failed: 3D scenes will fall back to a text notice\n");
    // 上で GL コンテキストに合わせて選んだ版を 3D レンダラのシェーダにも使わせる（出所は 1 つ）。
    quantviz::viz::glapi::set_glsl_version(glsl_version);

    // registry はこのスコープで閉じる: シーン（= GL オブジェクトを持つパネル）を、下の
    // GUI シャットダウン（GL コンテキスト破棄）より前に必ず破棄するため。
    {
        // -------------------------------------------------------------- scenes
        quantviz::viz::SceneRegistry registry;
        registry.add("Streaming", quantviz::viz::make_streaming_scene);
        registry.add("Greeks", quantviz::viz::make_greeks_scene);
        registry.add("GARCH", quantviz::viz::make_garch_scene);
        registry.add("Kalman pair", quantviz::viz::make_kalman_scene);
        registry.add("Vol surface", quantviz::viz::make_vol_surface_scene);
        if (!registry.select(0)) {
            std::fprintf(stderr, "initial scene failed to start\n");
            return 1;
        }

        // -------------------------------------------------------------- frame loop
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
                ImGui_ImplGlfw_Sleep(10);
                continue;
            }

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            draw_menu_bar(registry);
            if (auto* scene = registry.current()) scene->draw();

            ImGui::Render();
            int display_w = 0, display_h = 0;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
        }

        if (auto* scene = registry.current()) scene->stop();  // 計算スレッドを join してから畳む
    }

    // ------------------------------------------------------------------ shutdown
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
