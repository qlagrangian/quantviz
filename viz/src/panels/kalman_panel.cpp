#include "panels/kalman_panel.hpp"

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
constexpr double kDt          = 1.0 / kTradingDays;  // 1 ステップ = 1 取引日

double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

KalmanPanel::KalmanPanel(const scenes::KalmanPairModel::Config& initial, double initial_speed)
    : clock_{static_cast<float>(initial_speed), false},
      obs_noise_(static_cast<float>(initial.obs_noise)),
      state_noise_(static_cast<float>(initial.state_noise)),
      beta_center_(static_cast<float>(initial.beta_center)) {}

void KalmanPanel::draw(Runner& runner) {
    ingest(runner);
    draw_prices();
    draw_hedge_ratio();
    draw_spread();
    draw_controls(runner);
}

// ---------------------------------------------------------------- Snapshot → History
void KalmanPanel::ingest(Runner& runner) {
    scenes::KalmanPairSnapshot s;
    while (runner.poll(s)) {
        last_ = s;
        ++received_;
        if (s.seq == 0) continue;  // Reset 直後の空スナップショットは描かない
        // Reset の時点でリングに積まれていた Snapshot は「古い seq」を持ったまま到着し、
        // 巻き戻った 0 日目の点より先に描かれてしまう。seq が進まなかった＝巻き戻ったので捨てる。
        // （Reset ボタンの clear_history() は prev_seq_ に触らない。触ると古い方が残ってしまう）
        if (s.seq <= prev_seq_) clear_history();
        prev_seq_ = s.seq;

        const double days = s.t * kTradingDays;
        const double sd   = s.beta_var > 0.0 ? std::sqrt(s.beta_var) : 0.0;
        px_x_.push(days, s.x);
        px_y_.push(days, s.y);
        beta_true_.push(days, s.beta_true);
        beta_hat_.push(days, s.beta_hat);
        beta_lo_.push(days, s.beta_hat - 2.0 * sd);
        beta_hi_.push(days, s.beta_hat + 2.0 * sd);
        spread_.push(days, s.spread);
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

// ---------------------------------------------------------------- Prices
void KalmanPanel::draw_prices() {
    ImGui::SetNextWindowSize(ImVec2(760, 230), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, 32), ImGuiCond_FirstUseEver);
    ImGui::Begin("Prices - cointegrated pair y = beta_t x + eps");
    if (ImPlot::BeginPlot("##prices", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("t (trading days)", "price", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
        if (follow_ && !px_x_.empty()) {
            const double x1 = px_x_.latest_x();
            ImPlot::SetupAxisLimits(ImAxis_X1, x1 - window_days_, x1, ImPlotCond_Always);
        }
        ImPlot::PlotLine("x", px_x_.xs(), px_x_.ys(), px_x_.count(), 0, px_x_.offset());
        ImPlot::PlotLine("y", px_y_.xs(), px_y_.ys(), px_y_.count(), 0, px_y_.offset());
        ImPlot::EndPlot();
    }
    ImGui::End();
}

// ---------------------------------------------------------------- Hedge ratio
void KalmanPanel::draw_hedge_ratio() {
    ImGui::SetNextWindowSize(ImVec2(760, 240), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, 272), ImGuiCond_FirstUseEver);
    ImGui::Begin("Hedge ratio - Kalman beta with a +/-2 sigma band");
    if (ImPlot::BeginPlot("##beta", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("t (trading days)", "beta", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
        if (follow_ && !beta_hat_.empty()) {
            const double x1 = beta_hat_.latest_x();
            ImPlot::SetupAxisLimits(ImAxis_X1, x1 - window_days_, x1, ImPlotCond_Always);
        }
        // 帯は lo/hi を同じ時刻で push しているので、xs/count/offset は lo のものを共用できる。
        // ImPlot 0.16 の PlotShaded は PlotLine と違って count <= 1 を弾かない
        // （RendererShaded の Prims = count − 1 が unsigned に落ちて巨大な予約になり落ちる）。
        // count > 1 はこちらで保証する。
        if (beta_lo_.count() > 1) {
            ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.25f);
            ImPlot::PlotShaded("beta hat +/- 2 sigma", beta_lo_.xs(), beta_lo_.ys(), beta_hi_.ys(),
                               beta_lo_.count(), 0, beta_lo_.offset());
        }
        ImPlot::PlotLine("beta true", beta_true_.xs(), beta_true_.ys(), beta_true_.count(), 0,
                         beta_true_.offset());
        ImPlot::PlotLine("beta hat", beta_hat_.xs(), beta_hat_.ys(), beta_hat_.count(), 0,
                         beta_hat_.offset());
        ImPlot::EndPlot();
    }
    ImGui::End();
}

// ---------------------------------------------------------------- Spread
void KalmanPanel::draw_spread() {
    ImGui::SetNextWindowSize(ImVec2(760, 200), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, 522), ImGuiCond_FirstUseEver);
    ImGui::Begin("Spread - y - beta_hat x (stationary if the pair is cointegrated)");
    if (ImPlot::BeginPlot("##spread", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("t (trading days)", "spread", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
        if (follow_ && !spread_.empty()) {
            const double x1 = spread_.latest_x();
            ImPlot::SetupAxisLimits(ImAxis_X1, x1 - window_days_, x1, ImPlotCond_Always);
        }
        ImPlot::PlotLine("spread", spread_.xs(), spread_.ys(), spread_.count(), 0, spread_.offset());
        const double zero = 0.0;
        ImPlot::PlotInfLines("##zero", &zero, 1, ImPlotInfLinesFlags_Horizontal);
        ImPlot::EndPlot();
    }
    ImGui::End();
}

// ---------------------------------------------------------------- Controls → Command
void KalmanPanel::draw_controls(Runner& runner) {
    using bridge::Command;
    using scenes::KalmanPairModel;

    ImGui::SetNextWindowSize(ImVec2(420, 690), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(780, 32), ImGuiCond_FirstUseEver);
    ImGui::Begin("Control");

    ImGui::SeparatorText("Model parameters (applied from the next step)");
    // ImGui のスライダーは float、Command は double。境界では明示的に広げる（clock_controls.hpp と同じ）。
    if (ImGui::SliderFloat("obs noise", &obs_noise_, 0.01f, 20.0f, "%.3f", ImGuiSliderFlags_Logarithmic))
        runner.send(Command::set_param(KalmanPairModel::kObsNoise, static_cast<double>(obs_noise_)));
    if (ImGui::SliderFloat("state noise", &state_noise_, 0.0f, 0.02f, "%.4f"))
        runner.send(Command::set_param(KalmanPairModel::kStateNoise, static_cast<double>(state_noise_)));
    if (ImGui::SliderFloat("true beta", &beta_center_, 0.2f, 3.0f, "%.3f"))
        runner.send(Command::set_param(KalmanPairModel::kBetaTrue, static_cast<double>(beta_center_)));
    ImGui::TextDisabled("filter: F = 1, H = [x_t]");
    ImGui::TextDisabled("Q = state noise^2, R = obs noise^2");

    draw_clock_controls(clock_, runner, [this] { clear_history(); });

    ImGui::SeparatorText("View");
    ImGui::Checkbox("follow latest", &follow_);
    ImGui::SliderFloat("window (days)", &window_days_, 20.0f, 4000.0f, "%.0f");

    ImGui::SeparatorText("Telemetry");
    // シーン固有の行はここ、共通の行（seq / snapshots per sec / ring / steps / frame）は clock_panel.hpp。
    const double sd = last_.beta_var > 0.0 ? std::sqrt(last_.beta_var) : 0.0;
    ImGui::Text("t              %.3f y  (%.0f days)", last_.t, last_.t * kTradingDays);
    ImGui::Text("x / y          %.4f / %.4f", last_.x, last_.y);
    ImGui::Text("beta true      %.5f", last_.beta_true);
    ImGui::Text("beta hat       %.5f  +/- %.5f (2 sigma)", last_.beta_hat, 2.0 * sd);
    ImGui::Text("beta error     %+.5f", last_.beta_hat - last_.beta_true);
    ImGui::Text("spread         %+.5f", last_.spread);
    ImGui::Text("innovation     %+.5f", last_.innovation);
    ImGui::Text("skipped upd.   %u", last_.skipped);  // 0 でなければ R か x が壊れている
    draw_runner_telemetry(runner, last_.seq, rate_ema_);

    ImGui::End();
}

void KalmanPanel::clear_history() {
    px_x_.clear();
    px_y_.clear();
    beta_true_.clear();
    beta_hat_.clear();
    beta_lo_.clear();
    beta_hi_.clear();
    spread_.clear();
}

// ---------------------------------------------------------------- シーン生成（main.cpp / SceneRegistry 用）
std::unique_ptr<Scene> make_kalman_scene() {
    scenes::KalmanPairModel::Config model_cfg;  // 既定値（x0 100, beta 1.2, sigma_eps 2.0, ...）
    model_cfg.seed = 42;

    bridge::RunnerConfig run_cfg;
    run_cfg.dt                     = kDt;
    run_cfg.clock.steps_per_second = 250.0;  // speed 1x: 1 取引年 ≈ 1 壁秒
    run_cfg.clock.speed            = 1.0;
    run_cfg.publish_every          = 1;

    // Panel は in-place 構築（History を抱えた大きな値をスタックに積まない）。
    return std::make_unique<RunnerScene<scenes::KalmanPairModel, KalmanPanel>>(
        scenes::KalmanPairModel{model_cfg}, run_cfg, std::in_place, model_cfg, run_cfg.clock.speed);
}

}  // namespace quantviz::viz
