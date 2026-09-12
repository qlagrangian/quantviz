#include "panels/fdm_panel.hpp"

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
#include "quantviz/core/pricing/black_scholes.hpp"

namespace quantviz::viz {

namespace {

constexpr int kCurveCount = static_cast<int>(scenes::FdmSceneSnapshot::kCurve);

/// V(S) プロットの X 範囲は [0, kSpotWindow · K]。S_max = 4K 全部を映すと、見たい K 周りが潰れる。
constexpr double kSpotWindow = 2.5;

// 待機表示。Reset / パラメータ変更のあと（一時停止中なら Step / Resume まで）全ウィンドウがこれを出す。
constexpr const char* kWaitingSnapshot = "waiting for the next snapshot (Step or Resume) ...";
constexpr const char* kWaitingSurface  = "waiting for the next surface (Step or Resume) ...";

/// 配当利回り q を持つ BS 価格。`bs_price` は q を取らないので S を e^{−qτ} 倍して渡す
/// （d1 の中の log(S/K) が −qτ ぶん動き、前進項も S e^{−qτ} になるので厳密に一致する）。
double bs_price_q(double s, double k, double tau, double r, double q, double sigma,
                  core::OptionType type) noexcept {
    return core::bs_price(s * std::exp(-q * tau), k, tau, r, sigma, type);
}

}  // namespace

FdmPanel::FdmPanel(const scenes::FdmAmericanModel::Config& initial, double initial_speed)
    : clock_{static_cast<float>(initial_speed), false},
      strike_(static_cast<float>(initial.K)),
      r_(static_cast<float>(initial.r)),
      sigma_(static_cast<float>(initial.sigma)),
      q_(static_cast<float>(initial.q)),
      omega_(static_cast<float>(initial.omega)),
      american_(initial.american),
      type_index_(initial.type == core::OptionType::Put ? 1 : 0) {
    // 初期視点: 満期（奥）から今日（手前）へ向かって面が伸びるのを斜め上から見る。R キーはここへ戻る。
    view_.camera.distance = 3.1f;
    view_.camera.yaw      = 0.60f;
    view_.camera.pitch    = 0.55f;
    view_.home            = view_.camera;
}

void FdmPanel::draw(Runner& runner) {
    ingest(runner);
    draw_surface();
    draw_curve();
    draw_boundary();
    draw_controls(runner);
}

// ---------------------------------------------------------------- Snapshot / Surface 取り込み
void FdmPanel::ingest(Runner& runner) {
    bool got = false;
    Snap s;
    while (runner.poll(s)) {  // リングを空にして最新 1 枚だけを使う（このシーンに履歴は無い）
        last_          = s;
        have_snapshot_ = true;
        ++received_;
        got = true;
    }
    rate_.sample(received_, now_seconds());

    if (got) {
        // 反復が巻き戻っていたら（Reset / パラメータ変更）手元の面は別物なので捨てる。
        if (last_.iteration < prev_iter_) invalidate_surface();
        prev_iter_ = last_.iteration;
        rebuild_curves();
    }

    if (runner.poll_surface(surface_)) {  // 常に最新 1 枚（古い面は TripleBuffer が捨てている）
        ++surfaces_;
        have_surface_ = true;
        // GL が無い環境では 3D ウィンドウがテキスト表示に落ちるので、40k 頂点ぶんの法線計算は回さない
        // （面自体は poll しておく: テレメトリの filled_rows と面のドロップ挙動は GL の有無に依らない）。
        if (glapi::gl_available()) {
            rebuild_mesh();
            mesh_dirty_ = true;
        }
    }
}

void FdmPanel::invalidate_surface() noexcept {
    have_surface_        = false;
    surface_.filled_rows = 0;
}

/// Reset / パラメータ変更を送った直後に呼ぶ。モデルは満期へ巻き戻るが、Runner が次の Snapshot を出すのは
/// **次のステップ**なので、一時停止中は手元の Snapshot も面も古いまま残る。片方（面）だけ捨てると
/// 「3D は waiting、2D とテレメトリは前回の掃引の値」という嘘の組み合わせになるので、両方まとめて捨てて
/// 全ウィンドウを "waiting" に揃える。次の Step / Resume で新しい 1 枚が来たら通常表示に戻る。
void FdmPanel::invalidate_view() noexcept {
    invalidate_surface();
    last_          = Snap{};
    have_snapshot_ = false;
    prev_iter_     = 0;
    // received_ には触らない: RateMeter に渡す累計なので単調増加でなければならない（hpp のコメント）。
}

/// 本源的価値と European 参照。どちらも Snapshot が載せているパラメータ真値だけから決まる。
void FdmPanel::rebuild_curves() {
    const double tau = std::max(0.0, last_.maturity - last_.t_remaining);  // 満期までの残存期間
    for (std::size_t i = 0; i < Snap::kCurve; ++i) {
        const double s  = last_.spots[i];
        intrinsic_[i]   = last_.type == core::OptionType::Call ? std::max(s - last_.K, 0.0)
                                                               : std::max(last_.K - s, 0.0);
        european_[i]    = bs_price_q(s, last_.K, tau, last_.r, last_.q, last_.sigma, last_.type);
    }
    // 行使境界の τ 軸: index i は i + 1 回目の反復の直後 = τ = (i + 1) · T / M。
    const double dtau = last_.n_time > 0 ? last_.maturity / static_cast<double>(last_.n_time) : 0.0;
    for (std::size_t i = 0; i < Snap::kCurve; ++i)
        boundary_x_[i] = static_cast<double>(i + 1) * dtau;
}

void FdmPanel::rebuild_mesh() {
    // メッシュの y 軸は −t（昇順、t = 0 の「今日」が最大 = 手前）。モデルの時間レベルは間引きで
    // 重複しうるので、狭義単調でなければ [−T, 0] の等間隔軸へ落とす（法線の差分商が 0 除算になる）。
    bool strict = true;
    for (std::size_t i = 1; i < Surf::kT && strict; ++i) strict = surface_.times[i] < surface_.times[i - 1];
    if (strict) {
        for (std::size_t i = 0; i < Surf::kT; ++i) mesh_y_[i] = -surface_.times[i];
    } else {
        const float t0 = surface_.times[0];
        for (std::size_t i = 0; i < Surf::kT; ++i)
            mesh_y_[i] = -t0 * (1.f - static_cast<float>(i) / static_cast<float>(Surf::kT - 1));
    }
    mesh_.set_axes(std::span<const float>(surface_.spots), std::span<const float>(mesh_y_));
    mesh_.set_z(std::span<const float>(surface_.values));
    mesh_.update_normals();
}

// ---------------------------------------------------------------- V(S, t) surface (3D)
void FdmPanel::draw_surface() {
    ImGui::SetNextWindowSize(ImVec2(760, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, kPanelTop), ImGuiCond_FirstUseEver);
    // ホイールは 3D のズームに使うので、ウィンドウ側のスクロールには渡さない。
    ImGui::Begin("V(S, t) surface (3D)", nullptr,
                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    if (!glapi::gl_available()) {
        ImGui::TextDisabled("OpenGL functions unavailable - the 3D view is disabled");
        ImGui::TextDisabled("(gl_load() failed at startup; see stderr for the missing entry point)");
        ImGui::End();
        return;
    }

    if (!renderer_.ready() && !init_tried_) {
        init_tried_ = true;
        if (!renderer_.init(Surf::kS, Surf::kT))
            std::fprintf(stderr, "SurfaceRenderer::init failed: %s\n", renderer_.last_error().c_str());
    }
    if (!renderer_.ready()) {
        ImGui::TextDisabled("3D renderer unavailable:");
        ImGui::TextWrapped("%s", renderer_.last_error().c_str());
        ImGui::End();
        return;
    }

    if (!have_surface_ || !have_snapshot_) {
        ImGui::TextDisabled("%s", kWaitingSurface);
        ImGui::End();
        return;
    }

    // 色と等高線のスケールは実測 min/max ではなく理論上の上限で固定する: 面が伸びるたびに色の
    // 対応が変わると、どこが高いのか読めなくなる。put は V ≤ K、call は V ≤ S_max − K。
    const double s_max = static_cast<double>(surface_.spots[Surf::kS - 1]);
    const float  z_max = static_cast<float>(last_.type == core::OptionType::Put ? last_.K
                                                                                : s_max - last_.K);
    if (view_.draw(renderer_, mesh_, ImGui::GetContentRegionAvail(), 0.f, z_max, mesh_dirty_))
        mesh_dirty_ = false;  // 実際に VBO へ送れたフレームだけ下ろす

    ImGui::End();
}

// ---------------------------------------------------------------- V(S) now
void FdmPanel::draw_curve() {
    ImGui::SetNextWindowSize(ImVec2(375, 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, 442), ImGuiCond_FirstUseEver);
    ImGui::Begin("V(S) now");

    if (!have_snapshot_) {
        ImGui::TextDisabled("%s", kWaitingSnapshot);
        ImGui::End();
        return;
    }

    if (ImPlot::BeginPlot("##vs", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("S (spot)", "V", 0, ImPlotAxisFlags_AutoFit);
        // K を動かすと見たい帯も動くので、X 範囲は毎フレーム K から張り直す。
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, kSpotWindow * last_.K, ImPlotCond_Always);
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);

        ImPlot::PlotLine("V(S) FDM", last_.spots.data(), last_.values.data(), kCurveCount);
        ImPlot::PlotLine("intrinsic", last_.spots.data(), intrinsic_.data(), kCurveCount);
        ImPlot::PlotLine("European (BS)", last_.spots.data(), european_.data(), kCurveCount);

        const double s0 = last_.s0;
        ImPlot::SetNextLineStyle(ImVec4(0.85f, 0.85f, 0.90f, 0.9f), 1.0f);
        ImPlot::PlotInfLines("S0", &s0, 1);
        if (std::isfinite(last_.exercise_boundary_now)) {  // European / 該当なしは NaN
            const double sb = last_.exercise_boundary_now;
            ImPlot::SetNextLineStyle(ImVec4(1.00f, 0.60f, 0.20f, 0.9f), 1.5f);
            ImPlot::PlotInfLines("S*", &sb, 1);
        }
        ImPlot::EndPlot();
    }
    ImGui::End();
}

// ---------------------------------------------------------------- Exercise boundary
void FdmPanel::draw_boundary() {
    ImGui::SetNextWindowSize(ImVec2(375, 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(395, 442), ImGuiCond_FirstUseEver);
    ImGui::Begin("Exercise boundary S*(t)");

    if (!have_snapshot_) {
        ImGui::TextDisabled("%s", kWaitingSnapshot);
        ImGui::End();
        return;
    }
    if (!last_.american) {
        ImGui::TextDisabled("European: no early exercise, hence no boundary");
    } else if (last_.boundary_len == 0) {
        ImGui::TextDisabled("step the solver to trace the boundary");
    } else {
        ImGui::TextDisabled("S* -> K as tau -> 0 (put: exercise below the line)");
    }

    if (ImPlot::BeginPlot("##boundary", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("tau (years to expiry)", "S*", 0, ImPlotAxisFlags_AutoFit);
        // τ 軸は常に [0, T]: 反復が進むほど線が右へ伸びる様子を見せたいので、AutoFit で
        // 毎フレーム縮尺が変わらないようにする（点が 1 個のときに軸が暴れるのも防げる）。
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, last_.maturity, ImPlotCond_Always);
        ImPlot::SetupLegend(ImPlotLocation_SouthEast);
        if (last_.american && last_.boundary_len > 0) {
            // NaN（その反復では境界節点が見つからなかった）は ImPlot が切れ目として描く。
            ImPlot::PlotLine("S*(tau)", boundary_x_.data(), last_.boundary_t.data(),
                             static_cast<int>(last_.boundary_len));
        }
        const double k = last_.K;
        ImPlot::SetNextLineStyle(ImVec4(0.85f, 0.85f, 0.90f, 0.7f), 1.0f);
        ImPlot::PlotInfLines("K", &k, 1, ImPlotInfLinesFlags_Horizontal);
        ImPlot::EndPlot();
    }
    ImGui::End();
}

// ---------------------------------------------------------------- Controls → Command
void FdmPanel::draw_controls(Runner& runner) {
    using bridge::Command;
    using scenes::FdmAmericanModel;

    ImGui::SetNextWindowSize(ImVec2(420, 690), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(780, kPanelTop), ImGuiCond_FirstUseEver);
    ImGui::Begin("Control");

    // パラメータを変えると掃引は満期からやり直しになる（PDE の途中で係数は差し替えられない）。
    // 手元の面も別物になるので、送った時点で捨てる。
    ImGui::SeparatorText("Contract (changing any of these restarts the sweep)");
    if (ImGui::SliderFloat("K (strike)", &strike_, 50.0f, 200.0f, "%.1f")) {
        runner.send(Command::set_param(FdmAmericanModel::kStrike, static_cast<double>(strike_)));
        invalidate_view();
    }
    if (ImGui::SliderFloat("r (rate)", &r_, -0.05f, 0.20f, "%.4f")) {
        runner.send(Command::set_param(FdmAmericanModel::kRate, static_cast<double>(r_)));
        invalidate_view();
    }
    if (ImGui::SliderFloat("sigma", &sigma_, 0.01f, 1.00f, "%.3f")) {
        runner.send(Command::set_param(FdmAmericanModel::kSigma, static_cast<double>(sigma_)));
        invalidate_view();
    }
    if (ImGui::SliderFloat("q (dividend)", &q_, 0.0f, 0.20f, "%.4f")) {
        runner.send(Command::set_param(FdmAmericanModel::kDividend, static_cast<double>(q_)));
        invalidate_view();
    }
    if (ImGui::Checkbox("American (early exercise)", &american_)) {
        runner.send(Command::set_param(FdmAmericanModel::kAmerican, american_ ? 1.0 : 0.0));
        invalidate_view();
    }
    static const char* kTypeNames[] = {"Call", "Put"};
    if (ImGui::Combo("option type", &type_index_, kTypeNames, 2)) {
        runner.send(Command::set_param(FdmAmericanModel::kOptionType,
                                       static_cast<double>(type_index_)));
        invalidate_view();
    }
    ImGui::BeginDisabled(!american_);  // ω は PSOR（American）でしか効かない
    if (ImGui::SliderFloat("omega (PSOR)", &omega_, 1.0f, 1.99f, "%.2f")) {
        runner.send(Command::set_param(FdmAmericanModel::kOmega, static_cast<double>(omega_)));
        invalidate_view();
    }
    ImGui::EndDisabled();
    // ここは Snapshot を読まない（Reset 直後は last_ が空なので「0 iterations」と嘘をつく）。
    // 反復数は下の Telemetry の "iteration i / M" が出す。
    ImGui::TextDisabled("S_max = 4K;  1 Step = 1 backward iteration");
    ImGui::TextDisabled("surface %u x %u samples (the PDE grid is finer)",
                        static_cast<unsigned>(Surf::kS), static_cast<unsigned>(Surf::kT));

    ImGui::SeparatorText("View");
    ImGui::SliderInt("contour lines", &view_.contour_lines, 0, 40);
    ImGui::Checkbox("wireframe", &view_.wire);
    if (ImGui::Button("Reset view", ImVec2(100, 0))) view_.camera = view_.home;
    ImGui::SameLine();
    ImGui::TextDisabled("drag / wheel / R on the 3D image");

    // Reset は掃引を満期へ巻き戻す。描画側の履歴は無いが、手元の面は次の面が来るまで無効。
    draw_clock_controls(clock_, runner, [this] { invalidate_view(); });

    ImGui::SeparatorText("Telemetry");
    draw_runner_telemetry(runner, last_.seq, rate_.per_second());
    if (!have_snapshot_) {
        ImGui::TextDisabled("%s", kWaitingSnapshot);
    } else {
        const double tau = std::max(0.0, last_.maturity - last_.t_remaining);
        ImGui::Text("iteration      %u / %u   (remaining %u)", last_.iteration, last_.n_time,
                    last_.remaining);
        ImGui::Text("t (sweep left) %.4f y", last_.t_remaining);
        ImGui::Text("tau to expiry  %.4f y  (%.1f trading days)", tau, tau * kTradingDays);
        ImGui::Text("V(S0 = %.1f)   %.6f", last_.s0, last_.v_at_s0);
        ImGui::Text("delta(S0)      %.6f", last_.delta_at_s0);
        if (std::isfinite(last_.exercise_boundary_now))
            ImGui::Text("S*             %.4f", last_.exercise_boundary_now);
        else
            ImGui::Text("S*             -");
        ImGui::Text("PSOR sweeps    %u", last_.psor_iterations);
        if (last_.status == Snap::kStatusOk) {
            ImGui::Text("status         ok");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1.0f),
                               "status         PSOR did not converge");
        }
        ImGui::Text("surfaces       %llu  (rows %u / %u)",
                    static_cast<unsigned long long>(surfaces_), surface_.filled_rows,
                    static_cast<unsigned>(Surf::kT));
        ImGui::Text("fps            %.1f", static_cast<double>(ImGui::GetIO().Framerate));
    }

    ImGui::End();
}

// ---------------------------------------------------------------- シーン生成
std::unique_ptr<Scene> make_fdm_scene() {
    // Snapshot が ≈ 6 KB なのでリングは小さく。面は TripleBuffer（3 枚 ≈ 480 KB）なので、
    // RunnerScene ごと make_unique でヒープに置かれることが前提（`scene_registry.hpp`）。
    constexpr std::size_t kSnapCap = 64;

    scenes::FdmAmericanModel::Config model_cfg;  // 200x200 格子、K=100, T=1, r=5%, σ=20%, American put

    bridge::RunnerConfig run_cfg;
    run_cfg.dt                     = 1.0 / 252.0;  // モデルは使わない（刻みは時間格子が決める）
    run_cfg.clock.steps_per_second = 20.0;         // 1 ステップ = 1 反復。目で追える速さ
    run_cfg.clock.speed            = 1.0;
    run_cfg.publish_every          = 1;
    run_cfg.surface_every          = 1;

    // Panel は in-place 構築（面 + メッシュで 2 MB 超をスタックに積まない）。
    return std::make_unique<RunnerScene<scenes::FdmAmericanModel, FdmPanel, kSnapCap>>(
        scenes::FdmAmericanModel{model_cfg}, run_cfg, std::in_place, model_cfg, run_cfg.clock.speed);
}

}  // namespace quantviz::viz
