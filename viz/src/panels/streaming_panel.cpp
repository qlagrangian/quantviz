#include "panels/streaming_panel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <utility>

#include <imgui.h>
#include <implot.h>

#include "panels/clock_panel.hpp"

namespace quantviz::viz {

namespace {

constexpr double kTradingDays = 252.0;

double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

StreamingPanel::StreamingPanel(double dt, const scenes::StreamingModel::Config& initial, double initial_speed)
    : dt_(dt),
      clock_{static_cast<float>(initial_speed), false},
      mu_(static_cast<float>(initial.gbm.mu)),
      sigma_(static_cast<float>(initial.gbm.sigma)),
      lambda_(static_cast<float>(initial.ewma_lambda)) {}

void StreamingPanel::draw(Runner& runner) {
    ingest(runner);
    draw_price();
    draw_volatility();
    draw_controls(runner);
}

// ---------------------------------------------------------------- Snapshot → History
void StreamingPanel::ingest(Runner& runner) {
    scenes::StreamingSnapshot s;
    while (runner.poll(s)) {
        last_ = s;
        ++received_;
        if (s.seq == 0) continue;  // Reset 直後の空スナップショットは描かない
        const double days = s.t * kTradingDays;
        price_.push(days, s.spot);
        realised_vol_.push(days, s.var_return > 0.0 ? std::sqrt(s.var_return / dt_) : 0.0);
        ewma_vol_.push(days, s.ewma_var > 0.0 ? std::sqrt(s.ewma_var / dt_) : 0.0);
        sigma_true_.push(days, s.sigma_true);
    }

    // 受信レート（1 秒 EMA）
    const double t = now_seconds();
    if (last_wall_ == 0.0) {
        last_wall_  = t;
        last_count_ = received_;
    } else if (t - last_wall_ >= 0.25) {
        const double inst = static_cast<double>(received_ - last_count_) / (t - last_wall_);
        rate_ema_         = rate_ema_ == 0.0 ? inst : 0.8 * rate_ema_ + 0.2 * inst;
        last_wall_        = t;
        last_count_       = received_;
    }
}

// ---------------------------------------------------------------- Spot
void StreamingPanel::draw_price() {
    ImGui::SetNextWindowSize(ImVec2(760, 360), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, 32), ImGuiCond_FirstUseEver);
    ImGui::Begin("Spot - GBM (exact discretisation)");
    if (ImPlot::BeginPlot("##spot", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("t (trading days)", "S", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
        if (follow_ && !price_.empty()) {
            const double x1 = price_.latest_x();
            ImPlot::SetupAxisLimits(ImAxis_X1, x1 - window_days_, x1, ImPlotCond_Always);
        }
        ImPlot::PlotLine("S", price_.xs(), price_.ys(), price_.count(), 0, price_.offset());
        ImPlot::EndPlot();
    }
    ImGui::End();
}

// ---------------------------------------------------------------- Volatility
void StreamingPanel::draw_volatility() {
    ImGui::SetNextWindowSize(ImVec2(760, 320), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, 402), ImGuiCond_FirstUseEver);
    ImGui::Begin("Volatility - annualised: sqrt(var_step / dt)");
    if (ImPlot::BeginPlot("##vol", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("t (trading days)", "sigma", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
        if (follow_ && !price_.empty()) {
            const double x1 = price_.latest_x();
            ImPlot::SetupAxisLimits(ImAxis_X1, x1 - window_days_, x1, ImPlotCond_Always);
        }
        ImPlot::PlotLine("true sigma", sigma_true_.xs(), sigma_true_.ys(), sigma_true_.count(), 0,
                         sigma_true_.offset());
        ImPlot::PlotLine("EWMA (lambda)", ewma_vol_.xs(), ewma_vol_.ys(), ewma_vol_.count(), 0,
                         ewma_vol_.offset());
        ImPlot::PlotLine("realised (Welford, all history)", realised_vol_.xs(), realised_vol_.ys(),
                         realised_vol_.count(), 0, realised_vol_.offset());
        ImPlot::EndPlot();
    }
    ImGui::End();
}

// ---------------------------------------------------------------- Controls → Command
void StreamingPanel::draw_controls(Runner& runner) {
    using bridge::Command;
    using scenes::StreamingModel;

    ImGui::SetNextWindowSize(ImVec2(420, 690), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(780, 32), ImGuiCond_FirstUseEver);
    ImGui::Begin("Control");

    ImGui::SeparatorText("Model parameters (applied from the next step)");
    if (ImGui::SliderFloat("mu (annual drift)", &mu_, -0.5f, 0.5f, "%.3f"))
        runner.send(Command::set_param(StreamingModel::kMu, mu_));
    if (ImGui::SliderFloat("sigma (annual vol)", &sigma_, 0.0f, 1.0f, "%.3f"))
        runner.send(Command::set_param(StreamingModel::kSigma, sigma_));
    if (ImGui::SliderFloat("EWMA lambda", &lambda_, 0.80f, 0.999f, "%.4f", ImGuiSliderFlags_Logarithmic))
        runner.send(Command::set_param(StreamingModel::kEwmaLambda, lambda_));
    ImGui::TextDisabled("effective window ~ 1/(1-lambda) = %.0f steps", 1.0 / (1.0 - static_cast<double>(lambda_)));

    draw_clock_controls(clock_, runner, [this] { clear_history(); });

    ImGui::SeparatorText("View");
    ImGui::Checkbox("follow latest", &follow_);
    ImGui::SliderFloat("window (days)", &window_days_, 0.5f, 60.0f, "%.1f");

    ImGui::SeparatorText("Telemetry");
    // 共通の行（seq / snapshots per sec / ring / steps / frame）は clock_panel.hpp、この下はシーン固有。
    draw_runner_telemetry(runner, last_.seq, rate_ema_);
    ImGui::Text("t              %.3f y  (%.2f days)", last_.t, last_.t * kTradingDays);
    ImGui::Text("spot           %.4f", last_.spot);
    ImGui::Text("last log ret   %+.6f", last_.log_return);

    ImGui::End();
}

// ---------------------------------------------------------------- シーン生成（main.cpp / SceneRegistry 用）
std::unique_ptr<Scene> make_streaming_scene() {
    constexpr double kDt = 1.0 / (252.0 * 390.0);  // 1 分足（年単位）

    scenes::StreamingModel::Config model_cfg;
    model_cfg.gbm         = {100.0, 0.05, 0.20};  // s0, mu, sigma
    model_cfg.ewma_lambda = 0.94;
    model_cfg.seed        = 42;

    bridge::RunnerConfig run_cfg;
    run_cfg.dt                     = kDt;
    run_cfg.clock.steps_per_second = 500.0;  // speed 1x: 1 取引日(390 本) ≈ 0.8 壁秒
    run_cfg.clock.speed            = 1.0;
    run_cfg.publish_every          = 1;

    // Panel は in-place 構築（History を抱えた大きな値をスタックに積まない）。
    return std::make_unique<RunnerScene<scenes::StreamingModel, StreamingPanel>>(
        scenes::StreamingModel{model_cfg}, run_cfg, std::in_place, kDt, model_cfg, run_cfg.clock.speed);
}

void StreamingPanel::clear_history() {
    price_.clear();
    realised_vol_.clear();
    ewma_vol_.clear();
    sigma_true_.clear();
}

}  // namespace quantviz::viz
