#include "panels/exec_panel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <span>
#include <utility>

#include <imgui.h>
#include <implot.h>

#include "gl/gl_loader.hpp"
#include "panels/clock_panel.hpp"
#include "panels/panel_common.hpp"

namespace quantviz::viz {

namespace {

constexpr int kTrajCount     = static_cast<int>(scenes::ExecSnapshot::kN);
constexpr int kFrontierCount = static_cast<int>(scenes::ExecSnapshot::kFrontier);
constexpr int kLadderCount   = static_cast<int>(scenes::ExecSnapshot::kL);

// 起動直後（まだ 1 枚も受け取っていない）の待機表示。パラメータ変更のあとは bridge の R10 が
// 一時停止中でも 1 組を publish するので、ここには戻らない。
constexpr const char* kWaitingSnapshot = "waiting for the first snapshot ...";
constexpr const char* kWaitingSurface  = "waiting for the surface ...";

/// 現在の λ の線・マーカーの色（梯子の細線とはっきり区別する）。
const ImVec4 kCurrentColor(1.00f, 0.62f, 0.22f, 1.0f);
const ImVec4 kMarkerColor(0.95f, 0.95f, 0.98f, 0.9f);
/// 梯子 8 点の色。ImPlot の自動色に任せると現在の λ と同じオレンジが当たって見分けが付かない。
const ImVec4 kLadderColor(0.55f, 0.60f, 0.70f, 0.9f);

}  // namespace

ExecPanel::ExecPanel(const core::AcParams& initial, double initial_speed)
    : clock_{static_cast<float>(initial_speed), false},
      lambda_log10_(static_cast<float>(std::log10(initial.lambda))),
      eta_log10_(static_cast<float>(std::log10(initial.eta))),
      gamma_(static_cast<float>(initial.gamma)),
      sigma_(static_cast<float>(initial.sigma)),
      horizon_(static_cast<float>(initial.T)) {
    // 初期視点: t 軸（執行の進み）が手前に来る斜め上から。R キーはここへ戻る。
    view_.camera.distance = 3.0f;
    view_.camera.yaw      = 0.85f;
    view_.camera.pitch    = 0.55f;
    view_.home            = view_.camera;
}

void ExecPanel::draw(Runner& runner) {
    ingest(runner);
    draw_surface();
    draw_trajectories();
    draw_frontier();
    draw_controls(runner);
}

// ---------------------------------------------------------------- Snapshot / Surface 取り込み
void ExecPanel::ingest(Runner& runner) {
    bool got = false;
    Snap s;
    while (runner.poll(s)) {  // リングを空にして最新 1 枚だけを使う（このシーンに履歴は無い）
        last_ = s;
        ++received_;
        got = true;
    }
    rate_.sample(received_, now_seconds());
    if (got) rebuild_curves();

    if (runner.poll_surface(surface_)) {  // 常に最新 1 枚（古い面は TripleBuffer が捨てている）
        ++surfaces_;
        have_surface_ = true;
        // GL が無い環境では 3D はテキスト表示に落ちるので、2048 頂点ぶんの法線計算は回さない。
        if (glapi::gl_available()) {
            rebuild_mesh();
            mesh_dirty_ = true;  // 下ろすのは実際に upload できたフレーム（draw_surface）
        }
    }
}

/// 軌道の t 軸とフロンティアの (x, y) を最新 Snapshot から張り直す。どれもパラメータ真値だけで決まる。
void ExecPanel::rebuild_curves() {
    const double tau = last_.params.T / static_cast<double>(scenes::ExecModel::kSteps);
    for (std::size_t i = 0; i < Snap::kN; ++i) traj_t_[i] = static_cast<double>(i) * tau;

    for (std::size_t i = 0; i < Snap::kFrontier; ++i) {
        frontier_v_[i] = last_.frontier[i].variance;
        frontier_e_[i] = last_.frontier[i].expected;
    }
    for (std::size_t l = 0; l < Snap::kL; ++l) {
        ladder_v_[l] = last_.costs[l].variance;
        ladder_e_[l] = last_.costs[l].expected;
    }
}

void ExecPanel::rebuild_mesh() {
    // y 軸は log10(λ)（昇順）。λ は必ず正（モデルのクランプ域の下端でも 3e-11）。
    // λ ≤ 0 は起こらない（クランプ域の下端でも 3e-11）が、起きても軸が「単位の違う数」に化けない
    // ように下限で押さえる: log10 のまま、単に −30 に張り付く。
    for (std::size_t l = 0; l < Surf::kL; ++l)
        mesh_y_[l] = std::log10(std::max(surface_.lambdas[l], 1e-30f));
    mesh_.set_axes(std::span<const float>(surface_.ts), std::span<const float>(mesh_y_));
    mesh_.set_z(std::span<const float>(surface_.x));  // row-major [iL][iT] = mesh の [iy][ix]
    mesh_.update_normals();
}

// ---------------------------------------------------------------- 3D: x(t, lambda)
void ExecPanel::draw_surface() {
    ImGui::SetNextWindowSize(ImVec2(760, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, kPanelTop), ImGuiCond_FirstUseEver);
    // ホイールは 3D のズームに使うので、ウィンドウ側のスクロールには渡さない。
    ImGui::Begin("Inventory surface x(t, lambda)", nullptr,
                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    if (!glapi::gl_available()) {
        ImGui::TextDisabled("OpenGL functions unavailable - the 3D view is disabled");
        ImGui::TextDisabled("(gl_load() failed at startup; see stderr for the missing entry point)");
        ImGui::TextDisabled("The 2D trajectories and the frontier below still work.");
        ImGui::End();
        return;
    }

    if (!renderer_.ready() && !init_tried_) {
        init_tried_ = true;
        if (!renderer_.init(Surf::kT, Surf::kL))
            std::fprintf(stderr, "SurfaceRenderer::init failed: %s\n", renderer_.last_error().c_str());
    }
    if (!renderer_.ready()) {
        ImGui::TextDisabled("3D renderer unavailable:");
        ImGui::TextWrapped("%s", renderer_.last_error().c_str());
        ImGui::End();
        return;
    }

    if (!have_surface_ || received_ == 0) {
        ImGui::TextDisabled("%s", kWaitingSurface);
        ImGui::End();
        return;
    }

    ImGui::TextDisabled("x = t (days)   y = log10(lambda)   z = shares left");
    // 色のスケールは実測 min/max ではなく [0, X] に固定する: λ を動かすたびに色の対応が
    // 変わると「どこがまだ在庫を抱えているか」が読めなくなる。
    if (view_.draw(renderer_, mesh_, ImGui::GetContentRegionAvail(), 0.f,
                   static_cast<float>(last_.params.X), mesh_dirty_))
        mesh_dirty_ = false;  // 実際に VBO へ送れたフレームだけ下ろす

    ImGui::End();
}

// ---------------------------------------------------------------- 2D: 軌道の束
void ExecPanel::draw_trajectories() {
    ImGui::SetNextWindowSize(ImVec2(375, 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, 442), ImGuiCond_FirstUseEver);
    ImGui::Begin("Trajectories");

    if (received_ == 0) {
        ImGui::TextDisabled("%s", kWaitingSnapshot);
        ImGui::End();
        return;
    }

    if (ImPlot::BeginPlot("##traj", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("t (days)", "shares left", 0, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, last_.params.T, ImPlotCond_Always);
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);

        char label[24];
        for (std::size_t l = 0; l < Snap::kL; ++l) {
            std::snprintf(label, sizeof(label), "%.1e", last_.lambdas[l]);
            ImPlot::PlotLine(label, traj_t_.data(), last_.trajectories.data() + l * Snap::kN,
                             kTrajCount);
        }
        // 現在の λ（スライダーの値）は太線で。梯子はこの周り ±1.5 桁に張ってある。
        ImPlot::SetNextLineStyle(kCurrentColor, 2.5f);
        ImPlot::PlotLine("current", traj_t_.data(), last_.trajectory.data(), kTrajCount);

        // 再生ヘッド: 縦線 + 軌道上のマーカー（step が動かすのはこれだけ）。
        const double t = last_.t;
        // 再生ヘッドの 2 本は凡例から外す（"##" 始まり）。8 本の梯子 + current で既に 9 項目あり、
        // 11 項目だとプロットの 1/3 が凡例で埋まる。
        ImPlot::SetNextLineStyle(kMarkerColor, 1.0f);
        ImPlot::PlotInfLines("##t", &t, 1);
        const double x_now = last_.x_now;
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0f, kCurrentColor, 1.0f, kCurrentColor);
        ImPlot::PlotScatter("##x(t)", &t, &x_now, 1);
        ImPlot::EndPlot();
    }

    ImGui::End();
}

// ---------------------------------------------------------------- 2D: 効率的フロンティア
void ExecPanel::draw_frontier() {
    ImGui::SetNextWindowSize(ImVec2(375, 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(395, 442), ImGuiCond_FirstUseEver);
    ImGui::Begin("Frontier");

    if (received_ == 0) {
        ImGui::TextDisabled("%s", kWaitingSnapshot);
        ImGui::End();
        return;
    }

    ImGui::TextDisabled("lambda %.0e ... %.0e (fixed grid)", last_.frontier_lambdas.front(),
                        last_.frontier_lambdas.back());
    if (ImPlot::BeginPlot("##frontier", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("V[C] (variance of shortfall)", "E[C] (expected cost)",
                          ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        // 分散は λ を 6 桁振ると 6 桁動く。線形軸だと左端に張り付いて曲線が読めない。
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);

        ImPlot::PlotLine("E[C] vs V[C]", frontier_v_.data(), frontier_e_.data(), kFrontierCount);
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Square, 3.0f, kLadderColor, 1.0f, kLadderColor);
        ImPlot::PlotScatter("ladder", ladder_v_.data(), ladder_e_.data(), kLadderCount);

        const double v = last_.cost.variance;
        const double e = last_.cost.expected;
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0f, kCurrentColor, 1.0f, kCurrentColor);
        ImPlot::PlotScatter("current lambda", &v, &e, 1);
        ImPlot::EndPlot();
    }

    ImGui::End();
}

// ---------------------------------------------------------------- Controls → Command
void ExecPanel::draw_controls(Runner& runner) {
    using bridge::Command;
    using scenes::ExecModel;

    ImGui::SetNextWindowSize(ImVec2(420, 722), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(780, kPanelTop), ImGuiCond_FirstUseEver);
    ImGui::Begin("Control");

    // どのスライダーもモデル側でその場で全部が引き直される（掃引ではないので seq は巻き戻らない）。
    // Ctrl+クリックのテキスト入力は AlwaysClamp でも念のためウィジェットの後にもう一度落とす。
    ImGui::SeparatorText("Almgren-Chriss parameters (applied immediately)");
    ImGui::PushItemWidth(-118.0f);  // ラベルぶんの幅を残す（既定だと右端で切れる）
    // λ と η は指数を動かす（ヘッダの「λ と η は log10 で持つ」）。表示は "1e-5.70" の形。
    const float lo_lambda = static_cast<float>(std::log10(ExecModel::kMinLambda));
    const float hi_lambda = static_cast<float>(std::log10(ExecModel::kMaxLambda));
    if (ImGui::SliderFloat("lambda", &lambda_log10_, lo_lambda, hi_lambda, "10^%+.2f",
                           ImGuiSliderFlags_AlwaysClamp)) {
        lambda_log10_ = std::clamp(lambda_log10_, lo_lambda, hi_lambda);
        runner.send(Command::set_param(ExecModel::kLambda,
                                       std::pow(10.0, static_cast<double>(lambda_log10_))));
    }
    const float lo_eta = static_cast<float>(std::log10(ExecModel::kMinEta));
    const float hi_eta = static_cast<float>(std::log10(ExecModel::kMaxEta));
    if (ImGui::SliderFloat("eta (temporary)", &eta_log10_, lo_eta, hi_eta, "10^%+.2f",
                           ImGuiSliderFlags_AlwaysClamp)) {
        eta_log10_ = std::clamp(eta_log10_, lo_eta, hi_eta);
        runner.send(
            Command::set_param(ExecModel::kEta, std::pow(10.0, static_cast<double>(eta_log10_))));
    }
    if (ImGui::SliderFloat("gamma (permanent)", &gamma_, static_cast<float>(ExecModel::kMinGamma),
                           static_cast<float>(ExecModel::kMaxGamma), "%.2e",
                           ImGuiSliderFlags_AlwaysClamp)) {
        gamma_ = std::clamp(gamma_, static_cast<float>(ExecModel::kMinGamma),
                            static_cast<float>(ExecModel::kMaxGamma));
        runner.send(Command::set_param(ExecModel::kGamma, static_cast<double>(gamma_)));
    }
    if (ImGui::SliderFloat("sigma (vol)", &sigma_, static_cast<float>(ExecModel::kMinSigma),
                           static_cast<float>(ExecModel::kMaxSigma), "%.3f",
                           ImGuiSliderFlags_AlwaysClamp)) {
        sigma_ = std::clamp(sigma_, static_cast<float>(ExecModel::kMinSigma),
                            static_cast<float>(ExecModel::kMaxSigma));
        runner.send(Command::set_param(ExecModel::kSigma, static_cast<double>(sigma_)));
    }
    if (ImGui::SliderFloat("T (days)", &horizon_, static_cast<float>(ExecModel::kMinT),
                           static_cast<float>(ExecModel::kMaxT), "%.2f",
                           ImGuiSliderFlags_AlwaysClamp)) {
        horizon_ = std::clamp(horizon_, static_cast<float>(ExecModel::kMinT),
                              static_cast<float>(ExecModel::kMaxT));
        runner.send(Command::set_param(ExecModel::kT, static_cast<double>(horizon_)));
    }
    ImGui::PopItemWidth();
    // λ と η のスライダーは指数を表示するので、線形の値もそのまま出す（コアが使っている真値。
    // まだ Snapshot が来ていない起動直後だけスライダーの値から引く）。
    if (received_ > 0)
        ImGui::TextDisabled("lambda = %.3e   eta = %.3e", last_.params.lambda, last_.params.eta);
    else
        ImGui::TextDisabled("lambda = %.3e   eta = %.3e",
                            std::pow(10.0, static_cast<double>(lambda_log10_)),
                            std::pow(10.0, static_cast<double>(eta_log10_)));
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("x(t) = X sinh(k(T-t))/sinh(kT), k ~ sqrt(lambda sigma^2/eta), N = 63 slices");
    ImGui::TextDisabled("lambda -> 0 is TWAP (a straight line); lambda up front-loads the program");
    ImGui::PopTextWrapPos();

    ImGui::SeparatorText("View");
    ImGui::SliderInt("contour lines", &view_.contour_lines, 0, 40, "%d", ImGuiSliderFlags_AlwaysClamp);
    ImGui::Checkbox("wireframe", &view_.wire);
    if (ImGui::Button("Reset view", ImVec2(100, 0))) view_.camera = view_.home;
    ImGui::SameLine();
    ImGui::TextDisabled("drag / wheel / R on the 3D image");

    // Reset は再生ヘッドと通番を 0 に戻すだけ。描画側に捨てる履歴は無い。
    draw_clock_controls(clock_, runner, [] {});

    ImGui::SeparatorText("Telemetry");
    draw_runner_telemetry(runner, last_.seq, rate_.per_second());
    if (received_ == 0) {
        ImGui::TextDisabled("%s", kWaitingSnapshot);
    } else {
        const double sd = std::sqrt(std::max(0.0, last_.cost.variance));
        ImGui::Text("t (playhead)   %.3f / %.2f d", last_.t, last_.params.T);
        ImGui::Text("x(t)           %.0f / %.0f shares", last_.x_now, last_.params.X);
        ImGui::Text("E[C]           %.1f  (%.4f / share)", last_.cost.expected,
                    last_.cost.expected / last_.params.X);
        ImGui::Text("sqrt(V[C])     %.1f", sd);
        ImGui::Text("kappa          %.4f   (kappa T = %.3f)", last_.kappa, last_.kappa * last_.params.T);
        ImGui::Text("half-life      %.3f d", last_.kappa > 0.0 ? std::log(2.0) / last_.kappa : INFINITY);
        ImGui::Text("surfaces       %llu received / %llu published",
                    static_cast<unsigned long long>(surfaces_),
                    static_cast<unsigned long long>(runner.surfaces_published()));
        ImGui::Text("grid           %zu x %zu  (%zu triangles)", Surf::kT, Surf::kL,
                    mesh_.triangle_count());
        ImGui::Text("fps            %.1f", static_cast<double>(ImGui::GetIO().Framerate));
    }

    ImGui::End();
}

// ---------------------------------------------------------------- シーン生成
std::unique_ptr<Scene> make_exec_scene() {
    // Snapshot が ≈ 5.6 KB なのでリングは小さく。面（3 枚 ≈ 25 KB）は TripleBuffer。
    constexpr std::size_t kSnapCap = 64;

    scenes::ExecModel::Config model_cfg;  // Almgren-Chriss (2000) §3 の数値例

    bridge::RunnerConfig run_cfg;
    // 1 ステップ = 0.02 日。既定の T = 5 日を 250 ステップ（60 steps/s で約 4 秒）で 1 周する。
    run_cfg.dt                     = 0.02;
    run_cfg.clock.steps_per_second = 60.0;
    run_cfg.clock.speed            = 1.0;
    run_cfg.publish_every          = 1;
    // 面はパラメータの関数なので毎ステップ同じものが出る。それでも surface_every = 1 なのは、
    // スライダーを動かしてから 3D が変わるまでの遅れを 1 ステップに抑えるため（Vol surface と同じ）。
    run_cfg.surface_every = 1;

    // パネルのスライダーはモデルが実際に使う値を映す（Model は Config をクランプして持つ）。
    scenes::ExecModel    model{model_cfg};
    const core::AcParams panel_params = model.params();

    // Panel は in-place 構築（面 + メッシュで 100 KB 超をスタックに積まない）。
    return std::make_unique<RunnerScene<scenes::ExecModel, ExecPanel, kSnapCap>>(
        std::move(model), run_cfg, std::in_place, panel_params, run_cfg.clock.speed);
}

}  // namespace quantviz::viz
