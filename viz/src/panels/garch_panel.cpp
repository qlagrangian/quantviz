#include "panels/garch_panel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include <imgui.h>
#include <implot.h>

#include "panels/clock_panel.hpp"
#include "panels/panel_common.hpp"

namespace quantviz::viz {

namespace {

/// heatmap の色域。最大値からこれだけ下までを色で塗り分け、それ以下は下端色に潰す。
/// 対数尤度は非定常側で数千下がるので、そのまま色域にすると尾根が 1 色に潰れて見えない。
constexpr double kLogLikSpan = 50.0;

/// 分散（ステップ = 1 営業日）→ 年率ボラ（%）。
double annualised_pct(double sigma2, double dt) {
    return sigma2 > 0.0 ? 100.0 * std::sqrt(sigma2 / dt) : 0.0;
}

/// 表示用の半減期の上限（日）。α+β が変換の上限（1−1e−8）に張り付くと ln(0.5)/ln(α+β) は 1e7 日
/// 規模になり桁が読めないので、これを超えたら丸めて "+" を付ける。
constexpr double kMaxHalfLifeDays = 10000.0;

/// 表示用の半減期（日）と、上限で丸めたことを示す接尾辞。
struct HalfLife {
    double      days;
    const char* suffix;
};

HalfLife half_life_display(core::GarchParams p) {
    const double h = core::garch_half_life(p);
    return h > kMaxHalfLifeDays ? HalfLife{kMaxHalfLifeDays, "+"} : HalfLife{h, ""};
}

}  // namespace

GarchPanel::GarchPanel(double dt, const scenes::GarchModel::Config& initial, double initial_speed)
    : dt_(dt),
      clock_{static_cast<float>(initial_speed), false},
      omega_(static_cast<float>(initial.truth.omega)),
      alpha_(static_cast<float>(initial.truth.alpha)),
      beta_(static_cast<float>(initial.truth.beta)),
      optimizer_(initial.use_bfgs ? 1 : 0),
      window_(static_cast<int>(initial.window)) {
    // 最初の Snapshot が届く前のフレームでも Control が意味のある値を出せるように種を入れておく
    // （空の Snapshot のままだと alpha+beta = 0 / half-life 0 / unconditional sigma inf が出る）。
    last_.true_params = initial.truth;
    last_.est_params  = scenes::GarchModel::kInitialEstimate;
    last_.window      = static_cast<std::uint32_t>(initial.window);
    last_.optimizer   = static_cast<std::uint8_t>(initial.use_bfgs ? 1 : 0);
}

void GarchPanel::draw(Runner& runner) {
    ingest(runner);
    draw_volatility();
    draw_surface();
    draw_controls(runner);
}

// ---------------------------------------------------------------- Snapshot → History
void GarchPanel::ingest(Runner& runner) {
    scenes::GarchSnapshot s;
    while (runner.poll(s)) {
        // seq が**厳密に**減ったら巻き戻り（Reset。その時点でリングに残っていた古い Snapshot も
        // ここで捕まる）。それまでに溜めた History を捨てる。Reset の seq 0 もこの条件に入る。
        // 比較が `<=` ではなく `<` なのは bridge の R10（一時停止中の SetParam は seq を据え置いた
        // まま Snapshot を 1 枚出し直す）を巻き戻しと取り違えないため: 同じ seq の再送では
        // History を消さず、新しいパラメータの点を足すだけにする（真値の参照線が「段」になる）。
        // clear_history() は prev_seq_ に触らない（触ると古い方が残る）。last_ の差し替え前に呼ぶ。
        if (s.seq < prev_seq_) clear_history();
        last_     = s;
        prev_seq_ = s.seq;  // 巻き戻りを認める（Reset 直後は 0 に戻る）
        ++received_;
        if (s.seq == 0) continue;  // まだ 1 歩も進んでいない点は History に積まない（偽の線分になる）
        const double days = s.t * kTradingDays;
        sigma_true_.push(days, annualised_pct(s.sigma2_true, dt_));
        // フィルタ 2 本は窓が埋まるまで 0（＝まだ推定していない）。0 は描かない。
        if (s.sigma2_filtered > 0.0) sigma_filtered_.push(days, annualised_pct(s.sigma2_filtered, dt_));
        if (s.sigma2_est > 0.0) sigma_est_.push(days, annualised_pct(s.sigma2_est, dt_));
    }

    rate_.sample(received_, now_seconds());
}

// ---------------------------------------------------------------- Volatility
void GarchPanel::draw_volatility() {
    ImGui::SetNextWindowSize(ImVec2(760, 360), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, kPanelTop), ImGuiCond_FirstUseEver);
    ImGui::Begin("Volatility - annualised sigma_t = sqrt(252 * sigma2_t)");
    if (ImPlot::BeginPlot("##vol", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("t (trading days)", "sigma (% p.a.)", ImPlotAxisFlags_None,
                          ImPlotAxisFlags_AutoFit);
        setup_follow_axis(sigma_true_, follow_, window_days_);
        if (sigma_true_.count() > 1)
            ImPlot::PlotLine("true sigma", sigma_true_.xs(), sigma_true_.ys(), sigma_true_.count(), 0,
                             sigma_true_.offset());
        if (sigma_filtered_.count() > 1)
            ImPlot::PlotLine("filtered @ true params", sigma_filtered_.xs(), sigma_filtered_.ys(),
                             sigma_filtered_.count(), 0, sigma_filtered_.offset());
        if (sigma_est_.count() > 1)
            ImPlot::PlotLine("filtered @ estimate", sigma_est_.xs(), sigma_est_.ys(), sigma_est_.count(),
                             0, sigma_est_.offset());
        ImPlot::EndPlot();
    }
    ImGui::End();
}

// ---------------------------------------------------------------- 尤度面 L(alpha, beta)
void GarchPanel::draw_surface() {
    ImGui::SetNextWindowSize(ImVec2(760, 328), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, 402), ImGuiCond_FirstUseEver);
    ImGui::Begin("Likelihood surface L(alpha, beta)");

    // 今の窓に対する当てはめがまだ無い（起動直後 / Reset 直後 / 窓を伸ばした直後）
    if (last_.path_len == 0) {
        ImGui::TextDisabled("filling the rolling window (%u days) before the next fit...", last_.window);
        ImGui::End();
        return;
    }

    // 色域: 最大値から kLogLikSpan だけ下まで。格子は行 = β 昇順なので、ImPlot が行 0 を上端に
    // 描く分だけ行を反転して heat_ に写す。
    double hi = -std::numeric_limits<double>::infinity();
    for (const double v : last_.loglik_grid) hi = std::max(hi, v);
    const double lo = hi - kLogLikSpan;
    for (std::size_t r = 0; r < kGrid; ++r)
        for (std::size_t c = 0; c < kGrid; ++c)
            heat_[r * kGrid + c] = std::clamp(last_.loglik_grid[(kGrid - 1 - r) * kGrid + c], lo, hi);

    // セル中心が格子点になるよう、境界は半セルぶん外へ広げる。
    const double a0 = last_.grid_alpha.front(), a1 = last_.grid_alpha.back();
    const double b0 = last_.grid_beta.front(), b1 = last_.grid_beta.back();
    const double da = (a1 - a0) / static_cast<double>(kGrid - 1);
    const double db = (b1 - b0) / static_cast<double>(kGrid - 1);

    ImPlot::PushColormap(ImPlotColormap_Viridis);
    ImPlot::ColormapScale("log-lik", lo, hi, ImVec2(70, -1));
    ImGui::SameLine();
    if (ImPlot::BeginPlot("##loglik", ImVec2(-1, -1))) {
        constexpr ImPlotAxisFlags kAxis = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines;
        ImPlot::SetupAxes("alpha", "beta", kAxis, kAxis);
        ImPlot::SetupAxesLimits(a0 - 0.5 * da, a1 + 0.5 * da, b0 - 0.5 * db, b1 + 0.5 * db,
                                ImPlotCond_Always);
        // 尾根は左上→右下に走るので、凡例は空いている右上へ（既定の左上だと真値マーカーを隠す）。
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);
        ImPlot::PlotHeatmap("L", heat_.data(), static_cast<int>(kGrid), static_cast<int>(kGrid), lo, hi,
                            nullptr, ImPlotPoint(a0 - 0.5 * da, b0 - 0.5 * db),
                            ImPlotPoint(a1 + 0.5 * da, b1 + 0.5 * db));

        // viridis の上で見えるよう、軌跡とマーカーの色は明示する。
        if (last_.path_len > 1) {
            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, 0.85f), 1.5f);
            ImPlot::PlotLine("optimiser path", last_.path_alpha.data(), last_.path_beta.data(),
                             static_cast<int>(last_.path_len));
        }

        const double est_a = last_.est_params.alpha, est_b = last_.est_params.beta;
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0f, ImVec4(1.0f, 1.0f, 1.0f, 1.0f), 1.5f,
                                   ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        ImPlot::PlotScatter("estimate", &est_a, &est_b, 1);

        const double true_a = last_.true_params.alpha, true_b = last_.true_params.beta;
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 10.0f, IMPLOT_AUTO_COL, 2.5f,
                                   ImVec4(1.0f, 0.25f, 0.25f, 1.0f));
        ImPlot::PlotScatter("true", &true_a, &true_b, 1);
        ImPlot::EndPlot();
    }
    ImPlot::PopColormap();
    ImGui::End();
}

// ---------------------------------------------------------------- Controls → Command
void GarchPanel::draw_controls(Runner& runner) {
    using bridge::Command;
    using scenes::GarchModel;

    ImGui::SetNextWindowSize(ImVec2(420, 698), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(780, kPanelTop), ImGuiCond_FirstUseEver);
    ImGui::Begin("Control");

    ImGui::SeparatorText("True parameters (applied immediately)");
    // ImGui のスライダーは float、Command は double（UI → Command 規約: 明示的に広げる）
    if (ImGui::SliderFloat("omega", &omega_, 1e-7f, 1e-4f, "%.2e", ImGuiSliderFlags_Logarithmic))
        runner.send(Command::set_param(GarchModel::kOmega, static_cast<double>(omega_)));
    if (ImGui::SliderFloat("alpha", &alpha_, 0.0f, 0.4f, "%.3f"))
        runner.send(Command::set_param(GarchModel::kAlpha, static_cast<double>(alpha_)));
    if (ImGui::SliderFloat("beta", &beta_, 0.5f, 0.999f, "%.4f"))
        runner.send(Command::set_param(GarchModel::kBeta, static_cast<double>(beta_)));
    // モデルは α+β < 1 を保つために (α, β) を比例縮小することがあるので、効いている値を出す。
    const core::GarchParams truth = last_.true_params;
    ImGui::TextDisabled("in effect: alpha %.4f  beta %.4f  (alpha+beta = %.4f)", truth.alpha, truth.beta,
                        truth.alpha + truth.beta);
    const HalfLife truth_hl = half_life_display(truth);
    ImGui::TextDisabled("half-life %.1f days%s, unconditional sigma %.2f %% p.a.", truth_hl.days,
                        truth_hl.suffix, annualised_pct(core::garch_unconditional_variance(truth), dt_));

    ImGui::SeparatorText("Estimation (rolling-window MLE)");
    if (ImGui::Combo("optimizer", &optimizer_, "Nelder-Mead\0BFGS\0"))
        runner.send(Command::set_param(GarchModel::kOptimizer, static_cast<double>(optimizer_)));
    if (ImGui::SliderInt("window (days)", &window_, static_cast<int>(GarchModel::kMinWindow),
                         static_cast<int>(GarchModel::kMaxWindow)))
        runner.send(Command::set_param(GarchModel::kWindow, static_cast<double>(window_)));
    // 軌跡は先頭から間引いて最大 kPath 点に落としてある（降下の全体が見えるように）。
    ImGui::TextDisabled("path = %u of %u optimiser path points (subsampled)", last_.path_len,
                        last_.path_iters);

    draw_clock_controls(clock_, runner, [this] { clear_history(); });

    ImGui::SeparatorText("View");
    ImGui::Checkbox("follow latest", &follow_);
    ImGui::SliderFloat("plot window (days)", &window_days_, 20.0f, 2000.0f, "%.0f",
                       ImGuiSliderFlags_Logarithmic);

    ImGui::SeparatorText("Telemetry");
    // 共通の行（seq / snapshots per sec / ring / steps / frame）は clock_panel.hpp、この下はシーン固有。
    draw_runner_telemetry(runner, last_.seq, rate_.per_second());
    ImGui::Text("t              %.2f y  (%.0f days)", last_.t, last_.t * kTradingDays);
    ImGui::Text("r_last         %+.5f", last_.r_last);
    ImGui::Text("sigma_t true   %.2f %% p.a.", annualised_pct(last_.sigma2_true, dt_));
    ImGui::Text("sigma_t est    %.2f %% p.a.", annualised_pct(last_.sigma2_est, dt_));
    ImGui::Text("est omega      %.3e", last_.est_params.omega);
    ImGui::Text("est alpha      %.4f", last_.est_params.alpha);
    ImGui::Text("est beta       %.4f", last_.est_params.beta);
    const HalfLife est_hl = half_life_display(last_.est_params);
    ImGui::Text("est half-life  %.1f days%s", est_hl.days, est_hl.suffix);
    ImGui::Text("log-lik        %.2f", last_.log_lik);
    ImGui::Text("path points    %u", last_.path_len);

    ImGui::End();
}

// ---------------------------------------------------------------- シーン生成（main.cpp / SceneRegistry 用）
std::unique_ptr<Scene> make_garch_scene() {
    constexpr double kDt = 1.0 / 252.0;  // 1 ステップ = 1 営業日

    scenes::GarchModel::Config model_cfg;
    model_cfg.truth       = {1e-6, 0.08, 0.90};  // ω, α, β（持続性 0.98、無条件ボラ ≈ 11 % p.a.）
    model_cfg.window      = 500;
    model_cfg.refit_every = 10;
    model_cfg.seed        = 20240912;

    bridge::RunnerConfig run_cfg;
    run_cfg.dt                     = kDt;
    run_cfg.clock.steps_per_second = 100.0;  // speed 1x: 窓 500 日が 5 秒で埋まる
    run_cfg.clock.speed            = 1.0;
    run_cfg.publish_every          = 1;

    // Snapshot が ~10 KB なのでリングは 256 枚。Panel は in-place 構築。
    return std::make_unique<RunnerScene<scenes::GarchModel, GarchPanel, GarchPanel::kSnapshotCapacity>>(
        scenes::GarchModel{model_cfg}, run_cfg, std::in_place, kDt, model_cfg, run_cfg.clock.speed);
}

void GarchPanel::clear_history() {
    sigma_true_.clear();
    sigma_filtered_.clear();
    sigma_est_.clear();
    // 尤度面・軌跡・推定値は捨てる（次の当てはめまで描かない）。Model が Reset で保持するもの
    // （真値・窓長・最適化器）はそのまま残す: 全体をゼロにすると次の Snapshot が届くまで
    // 「α+β = 0、無条件分散 = inf」を表示してしまう。
    last_.seq             = 0;
    last_.t               = 0.0;
    last_.r_last          = 0.0;
    last_.sigma2_true     = 0.0;
    last_.sigma2_filtered = 0.0;
    last_.sigma2_est      = 0.0;
    last_.est_params      = scenes::GarchModel::kInitialEstimate;
    last_.log_lik         = 0.0;
    last_.path_len        = 0;
    last_.path_iters      = 0;  // これを残すと Reset 直後の 1 フレームだけ "0 of 114" と出る
}

}  // namespace quantviz::viz
