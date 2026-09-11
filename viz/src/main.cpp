// viz/main.cpp — quantviz viewer entry point (M0: Streaming scene)
//
//   描画スレッド（このファイル）: GLFW + ImGui + ImPlot, 60 fps
//   計算スレッド（Runner）      : StreamingModel を steps_per_second で回す
//   両者は Snapshot リングと Command リングだけで繋がる。

#include <cstdio>

#if defined(__APPLE__)
#define GL_SILENCE_DEPRECATION
#endif
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

#include "panels/streaming_panel.hpp"
#include "quantviz/bridge/runner.hpp"
#include "quantviz/scenes/streaming_model.hpp"

namespace {

void glfw_error_callback(int error, const char* description) {
    std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
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

    GLFWwindow* window = glfwCreateWindow(1220, 720, "quantviz — M0 Streaming (GBM + online stats)", nullptr, nullptr);
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

    // ------------------------------------------------------------------ core + bridge
    constexpr double kDt = 1.0 / (252.0 * 390.0);  // 1 分足（年単位）

    quantviz::scenes::StreamingModel::Config model_cfg;
    model_cfg.gbm         = {100.0, 0.05, 0.20};
    model_cfg.ewma_lambda = 0.94;
    model_cfg.seed        = 42;

    quantviz::bridge::RunnerConfig run_cfg;
    run_cfg.dt                     = kDt;
    run_cfg.clock.steps_per_second = 500.0;  // speed 1x: 1 取引日(390 本) ≈ 0.8 壁秒
    run_cfg.clock.speed            = 1.0;
    run_cfg.publish_every          = 1;

    quantviz::bridge::Runner<quantviz::scenes::StreamingModel> runner(
        quantviz::scenes::StreamingModel{model_cfg}, run_cfg);
    quantviz::viz::StreamingPanel panel(kDt, model_cfg, run_cfg.clock.speed);
    runner.start();

    // ------------------------------------------------------------------ frame loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        panel.draw(runner);

        ImGui::Render();
        int display_w = 0, display_h = 0;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // ------------------------------------------------------------------ shutdown
    runner.stop();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
