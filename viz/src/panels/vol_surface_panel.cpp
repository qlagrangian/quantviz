#include "panels/vol_surface_panel.hpp"

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
#include "quantviz/core/pricing/vol_surface.hpp"

namespace quantviz::viz {

namespace {

constexpr int kSmileCount = static_cast<int>(scenes::VolSurfaceSurface::kK);

/// 色・等高線のスケールに載せる余白（レンジの割合）。端の面が真っ黒／真っ白に張り付かない程度。
constexpr float kZMargin = 0.05f;

const ImVec4 kOkColor(0.35f, 0.85f, 0.45f, 1.0f);
const ImVec4 kNgColor(0.94f, 0.42f, 0.38f, 1.0f);

}  // namespace

VolSurfacePanel::VolSurfacePanel(const scenes::VolSurfaceModel::Config& initial, double initial_speed)
    : clock_{static_cast<float>(initial_speed), false},
      sigma_atm_(static_cast<float>(initial.params.sigma_atm)),
      rho_(static_cast<float>(initial.params.rho)),
      eta_(static_cast<float>(initial.params.eta)),
      gamma_(static_cast<float>(initial.params.gamma)) {
    // 初期視点: 斜め上から、k 軸が手前に来る向き。R キーはここへ戻る。
    view_.camera.distance = 3.0f;
    view_.camera.yaw      = 0.9f;
    view_.camera.pitch    = 0.50f;
    view_.home            = view_.camera;
    // T 軸の真ん中のスマイルを既定で見せる（端の行は形が読みにくい）。
    smile_index_ = static_cast<int>(Surf::kT / 2);
}

void VolSurfacePanel::draw(Runner& runner) {
    ingest(runner);
    draw_surface();
    draw_smile();
    draw_controls(runner);
}

// ---------------------------------------------------------------- Snapshot / Surface 取り込み
void VolSurfacePanel::ingest(Runner& runner) {
    Snap s;
    while (runner.poll(s)) {  // リングを空にして最新 1 枚だけを使う（時系列は持たない）
        last_ = s;
        ++received_;
    }
    rate_.sample(received_, now_seconds());

    // 面は TripleBuffer なので 1 回の poll で「最新 1 枚」が来る（無ければ false）。
    if (!runner.poll_surface(surface_)) return;
    ++surfaces_;

    if (!axes_set_) {  // k・T 軸は Config で固定（Param では動かない）
        mesh_.set_axes(std::span<const float>(surface_.ks), std::span<const float>(surface_.ts));
        axes_set_ = true;
    }
    mesh_.set_z(std::span<const float>(surface_.iv));  // row-major [iT][iK] = mesh の [iy][ix]
    mesh_.update_normals();
    refresh_z_range();
    // 立てるのはここ、下ろすのは実際に upload できたとき（draw_surface）。
    mesh_dirty_ = true;
}

void VolSurfacePanel::refresh_z_range() noexcept {
    const auto [lo_it, hi_it] = std::minmax_element(surface_.iv.begin(), surface_.iv.end());
    const float lo            = *lo_it;
    const float hi            = *hi_it;
    float       span          = hi - lo;
    // 平らな面（σ_atm が小さく η も小さいとき）でも色スケールを潰さない。
    if (!(span > 0.0f)) span = std::max(std::fabs(hi) * 0.1f, 1e-3f);
    z_min_ = lo - kZMargin * span;
    z_max_ = hi + kZMargin * span;
}

// ---------------------------------------------------------------- 3D ウィンドウ
void VolSurfacePanel::draw_surface() {
    ImGui::SetNextWindowSize(ImVec2(760, 430), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, kPanelTop), ImGuiCond_FirstUseEver);
    // ホイールは 3D のズームに使うので、ウィンドウ側のスクロールには渡さない。
    ImGui::Begin("IV surface (3D)", nullptr,
                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    if (!glapi::gl_available()) {
        ImGui::TextDisabled("OpenGL functions unavailable - the 3D view is disabled");
        ImGui::TextDisabled("(gl_load() failed at startup; see stderr for the missing entry point)");
        ImGui::TextDisabled("The smile cross-section below still works.");
        ImGui::End();
        return;
    }

    if (!renderer_.ready() && !init_tried_) {
        init_tried_ = true;
        if (!renderer_.init(Surf::kK, Surf::kT))
            std::fprintf(stderr, "SurfaceRenderer::init failed: %s\n", renderer_.last_error().c_str());
    }
    if (!renderer_.ready()) {
        ImGui::TextDisabled("3D renderer unavailable:");
        ImGui::TextWrapped("%s", renderer_.last_error().c_str());
        ImGui::End();
        return;
    }

    if (surfaces_ == 0) {
        ImGui::TextDisabled("waiting for the first surface ...");
        ImGui::End();
        return;
    }

    ImGui::TextDisabled("x = k (log-moneyness)   y = T (years)   z = implied vol");
    if (view_.draw(renderer_, mesh_, ImGui::GetContentRegionAvail(), z_min_, z_max_, mesh_dirty_))
        mesh_dirty_ = false;  // 実際に VBO へ送れたフレームだけ下ろす
    ImGui::End();
}

// ---------------------------------------------------------------- スマイル断面（2D）
void VolSurfacePanel::draw_smile() {
    ImGui::SetNextWindowSize(ImVec2(760, 282), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, 470), ImGuiCond_FirstUseEver);
    ImGui::Begin("Smile @ T");

    if (surfaces_ == 0) {
        ImGui::TextDisabled("waiting for the first surface ...");
        ImGui::End();
        return;
    }

    // この値は配列の添字になる。ImGui のスライダーは Ctrl+Click のテキスト入力を既定では
    // 範囲に丸めない（AlwaysClamp が要る）ので、フラグを渡したうえでウィジェットの**後**に
    // もう一度落とす。先に落とすだけだと、入力されたそのフレームに範囲外の添字で読んでしまう。
    ImGui::SetNextItemWidth(280.0f);
    ImGui::SliderInt("T index", &smile_index_, 0, static_cast<int>(Surf::kT) - 1, "%d",
                     ImGuiSliderFlags_AlwaysClamp);
    smile_index_          = std::clamp(smile_index_, 0, static_cast<int>(Surf::kT) - 1);
    const std::size_t row = static_cast<std::size_t>(smile_index_);
    ImGui::SameLine();
    ImGui::Text("T = %.3f y", static_cast<double>(surface_.ts[row]));

    // 3D の面と同じ配列から描く（row-major [iT][iK] なので 1 行が連続している）。
    if (ImPlot::BeginPlot("##smile", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("k = log(K/F)", "implied vol", ImPlotAxisFlags_AutoFit,
                          ImPlotAxisFlags_AutoFit);
        ImPlot::SetupLegend(ImPlotLocation_North, ImPlotLegendFlags_Horizontal | ImPlotLegendFlags_Outside);
        ImPlot::PlotLine("iv(k)", surface_.ks.data(), &surface_.iv[row * Surf::kK], kSmileCount);
        // ATM（k = 0）: SSVI では w(0, T) = σ_atm² T なので iv(0, T) = σ_atm ちょうど。
        const double atm = 0.0;
        ImPlot::SetNextLineStyle(ImVec4(0.75f, 0.75f, 0.80f, 0.7f), 1.0f);
        ImPlot::PlotInfLines("k = 0", &atm, 1);
        const double sigma_atm = last_.params.sigma_atm;
        ImPlot::SetNextLineStyle(ImVec4(0.75f, 0.75f, 0.80f, 0.7f), 1.0f);
        ImPlot::PlotInfLines("sigma_atm", &sigma_atm, 1, ImPlotInfLinesFlags_Horizontal);
        ImPlot::EndPlot();
    }

    ImGui::End();
}

// ---------------------------------------------------------------- Controls → Command
void VolSurfacePanel::draw_controls(Runner& runner) {
    using bridge::Command;
    using scenes::VolSurfaceModel;

    ImGui::SetNextWindowSize(ImVec2(420, 720), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(780, kPanelTop), ImGuiCond_FirstUseEver);
    ImGui::Begin("Control");

    ImGui::SeparatorText("SSVI parameters (applied from the next step)");
    if (ImGui::SliderFloat("sigma_atm", &sigma_atm_, 0.05f, 1.00f, "%.3f"))
        runner.send(Command::set_param(VolSurfaceModel::kSigmaAtm, static_cast<double>(sigma_atm_)));
    if (ImGui::SliderFloat("rho (skew)", &rho_, -0.95f, 0.95f, "%.3f"))
        runner.send(Command::set_param(VolSurfaceModel::kRho, static_cast<double>(rho_)));
    if (ImGui::SliderFloat("eta (wings)", &eta_, 0.05f, 3.00f, "%.3f"))
        runner.send(Command::set_param(VolSurfaceModel::kEta, static_cast<double>(eta_)));
    if (ImGui::SliderFloat("gamma (decay)", &gamma_, 0.00f, 1.00f, "%.3f"))
        runner.send(Command::set_param(VolSurfaceModel::kGamma, static_cast<double>(gamma_)));
    // 式と判定は Control の幅で折り返す（ウィンドウを細くしても切れないように）。
    ImGui::PushTextWrapPos(0.0f);
    // 面が publish されるのはステップしたときだけなので、一時停止中はスライダーを動かしても
    // Step を押すまで絵が変わらない（Snapshot も同様）。それを言っておく。
    if (clock_.paused) ImGui::TextDisabled("paused: press Step to redraw the surface");
    ImGui::TextDisabled("w(k,T) = th/2 (1 + rho x + sqrt((x + rho)^2 + 1 - rho^2)),  x = phi k");
    ImGui::TextDisabled("th = sigma_atm^2 T,  phi = eta / th^gamma");

    // 判定はコアの述語（Gatheral-Jacquier Thm 4.1 + 翼のバンド）。真値である Snapshot の
    // パラメータで評価する（スライダーの値ではなく、コアが実際に使っている値を映す）。
    if (received_ == 0) {
        ImGui::TextDisabled("arbitrage check: waiting for the first snapshot ...");
    } else if (core::ssvi_calendar_arbitrage_free(last_.params)) {
        ImGui::TextColored(kOkColor, "calendar arbitrage-free, wings within eta(1+|rho|) <= 2");
    } else {
        ImGui::TextColored(kNgColor, "outside the safe band: eta(1+|rho|) = %.2f > 2",
                           last_.params.eta * (1.0 + std::fabs(last_.params.rho)));
    }
    ImGui::PopTextWrapPos();

    ImGui::SeparatorText("View");
    ImGui::SliderInt("contour lines", &view_.contour_lines, 0, 40, "%d", ImGuiSliderFlags_AlwaysClamp);
    ImGui::Checkbox("wireframe", &view_.wire);
    if (ImGui::Button("Reset view", ImVec2(100, 0))) view_.camera = view_.home;
    ImGui::SameLine();
    ImGui::TextDisabled("drag / wheel / R on the 3D image");

    // 3D も 2D も最新の 1 枚しか描かないので、Reset で消す描画側の履歴は無い。
    draw_clock_controls(clock_, runner, [] {});

    ImGui::SeparatorText("Telemetry");
    if (received_ == 0) {
        ImGui::TextDisabled("waiting for the first snapshot ...");
    } else {
        ImGui::Text("t              %.3f y  (%.2f trading days)", last_.t, last_.t * kTradingDays);
        ImGui::Text("sigma / rho    %.3f / %+.3f", last_.params.sigma_atm, last_.params.rho);
        ImGui::Text("eta / gamma    %.3f / %.3f", last_.params.eta, last_.params.gamma);
        ImGui::Text("k / T range    [%.2f, %.2f] / [%.2f, %.2f]", last_.k_min, last_.k_max, last_.t_min,
                    last_.t_max);
    }
    ImGui::Text("surfaces       %llu received / %llu published",
                static_cast<unsigned long long>(surfaces_),
                static_cast<unsigned long long>(runner.surfaces_published()));
    ImGui::Text("grid           %zu x %zu  (%zu triangles)", Surf::kK, Surf::kT, mesh_.triangle_count());
    ImGui::Text("color scale    [%.4f, %.4f]  (iv +/- %.0f%% margin)", static_cast<double>(z_min_),
                static_cast<double>(z_max_), static_cast<double>(kZMargin) * 100.0);
    ImGui::Text("fbo            %d x %d", renderer_.width(), renderer_.height());
    draw_runner_telemetry(runner, last_.seq, rate_.per_second());
    ImGui::Text("fps            %.1f", static_cast<double>(ImGui::GetIO().Framerate));

    ImGui::End();
}

// ---------------------------------------------------------------- シーン生成（main.cpp / SceneRegistry 用）
std::unique_ptr<Scene> make_vol_surface_scene() {
    constexpr std::size_t kSnapCap = 256;  // Snapshot は小さいが、面と同じく「最新だけ」見るシーン

    scenes::VolSurfaceModel::Config model_cfg;  // 既定 SSVI、k ∈ [−1, 1]、T ∈ [0.05, 3]

    bridge::RunnerConfig run_cfg;
    run_cfg.dt                     = 1.0 / 252.0;  // 1 ステップ = 1 取引日
    run_cfg.clock.speed            = 1.0;
    run_cfg.publish_every          = 1;
    // 面は (k, T) の関数で t に依らないので、毎ステップ publish されるのは「同じ面」である。
    // それでも steps_per_second = 60 / surface_every = 1 にしてあるのは意図的で、
    // スライダーを動かしてから絵が変わるまでの遅れを 1 ステップ（≒ 16 ms）に抑えるため。
    // 裏返すと、一時停止中は Step を押したときにしか面が更新されない（パネルがその旨を表示する）。
    run_cfg.clock.steps_per_second = 60.0;
    run_cfg.surface_every          = 1;

    // パネルのスライダーはモデルが実際に使う値を映す（Model は Config をクランプして持つ）。
    scenes::VolSurfaceModel::Config panel_cfg = model_cfg;
    panel_cfg.params                          = core::ssvi_clamp(model_cfg.params);

    // Panel は in-place 構築（面 + メッシュで 30 KB 超をスタックに積まない）。
    return std::make_unique<RunnerScene<scenes::VolSurfaceModel, VolSurfacePanel, kSnapCap>>(
        scenes::VolSurfaceModel{model_cfg}, run_cfg, std::in_place, panel_cfg, run_cfg.clock.speed);
}

}  // namespace quantviz::viz
