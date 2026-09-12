#include "panels/hjb_panel.hpp"

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
#include "quantviz/core/exec/hjb_merton.hpp"

namespace quantviz::viz {

namespace {

constexpr int kNodeCount = static_cast<int>(scenes::HjbSnapshot::kNodes);

// 起動直後（まだ 1 枚も受け取っていない）の待機表示。Reset / パラメータ変更のあとは bridge の R10 が
// 一時停止中でも 1 組を publish するので、ここには戻らない。
constexpr const char* kWaitingSnapshot = "waiting for the first snapshot ...";
constexpr const char* kWaitingSurface  = "waiting for the surface ...";

/// π* の Y 軸の固定上限（解析値の 1.6 倍か、解析値が 0 でも軸が潰れない最小幅）。
constexpr double kPiAxisFloor = 0.05;

/// 誤差を測る内側 90 %（両端 5 %）。モデルの `max_abs_pi_error` と同じ窓
/// （`hjb_model.hpp`「π* の誤差」: 境界節点は片側差分ぶんずれる）。
constexpr std::size_t kEdge = scenes::HjbSnapshot::kNodes * 5 / 100;

}  // namespace

HjbPanel::HjbPanel(const scenes::HjbModel::Config& initial, double initial_speed)
    : clock_{static_cast<float>(initial_speed), false},
      mu_(static_cast<float>(initial.mu)),
      r_(static_cast<float>(initial.r)),
      sigma_(static_cast<float>(initial.sigma)),
      gamma_(static_cast<float>(initial.gamma)) {
    // 初期視点: 満期（奥）から今日（手前）へ向かって面が伸びるのを斜め上から見る。R キーはここへ戻る。
    view_.camera.distance = 3.1f;
    view_.camera.yaw      = 0.60f;
    view_.camera.pitch    = 0.55f;
    view_.home            = view_.camera;
}

void HjbPanel::draw(Runner& runner) {
    ingest(runner);
    draw_surface();
    draw_value();
    draw_policy();
    draw_controls(runner);
}

// ---------------------------------------------------------------- Snapshot / Surface 取り込み
void HjbPanel::ingest(Runner& runner) {
    bool got = false;
    Snap s;
    while (runner.poll(s)) {  // リングを空にして最新 1 枚だけを使う（このシーンに履歴は無い）
        last_ = s;
        ++received_;
        got = true;
    }
    rate_.sample(received_, now_seconds());

    if (got) {
        // 掃引が巻き戻っていたら（Reset / パラメータ変更）手元の面は別物なので捨てる。
        // seq は**厳密に**減ったときだけ（R10 の同 seq 再送は巻き戻しではない）。
        if (last_.seq < prev_seq_ || last_.iteration < prev_iter_) invalidate_surface();
        prev_seq_  = last_.seq;
        prev_iter_ = last_.iteration;
        rebuild_scale();
    }

    if (runner.poll_surface(surface_)) {  // 常に最新 1 枚（古い面は TripleBuffer が捨てている）
        ++surfaces_;
        have_surface_ = true;
        // GL が無い環境では 3D ウィンドウがテキスト表示に落ちるので、40k 頂点ぶんの法線計算は回さない
        // （面自体は poll しておく: テレメトリの filled_rows と面のドロップ挙動は GL の有無に依らない）。
        // 中身が前回と同じ面（掃引が終わったあとは毎秒 20 枚届く）もここで弾く
        // （ヘッダ「メッシュ作り直しの省略」）。
        const bool changed = !mesh_valid_ || surface_.filled_rows != mesh_rows_ || scale_key_ != mesh_key_;
        if (glapi::gl_available() && changed) {
            rebuild_mesh();
            mesh_dirty_ = true;
        }
    }
}

void HjbPanel::invalidate_surface() noexcept {
    have_surface_        = false;
    mesh_valid_          = false;  // 次の面は中身が同じに見えても必ず作り直す
    surface_.filled_rows = 0;
}

/// 3D の色 / 高さレンジ。閉形式の四隅（w ∈ {w_min, w_max} × t ∈ {0, T}）と 0 を含む範囲に固定する
/// （ヘッダ「V のスケール」）。パラメータが動いたフレームだけ張り直す。
void HjbPanel::rebuild_scale() {
    const ScaleKey key{last_.mu,    last_.r,              last_.sigma,
                       last_.gamma, last_.wealth.front(), last_.wealth.back()};
    if (have_scale_ && key == scale_key_) return;
    scale_key_  = key;
    have_scale_ = true;

    const core::HjbParams p{last_.mu,  last_.r,   last_.sigma,                 last_.gamma, last_.T,
                            key.w_min, key.w_max, scenes::HjbSnapshot::kNodes, 1};
    const double          ws[2] = {key.w_min, key.w_max};
    const double          ts[2] = {0.0, last_.T};

    double lo = 0.0, hi = 0.0;  // 未計算行のゼロ平面を範囲に含める
    for (const double w : ws) {
        for (const double t : ts) {
            const double v = core::merton_value(p, w, t);
            if (!std::isfinite(v)) continue;
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
    }
    if (!(hi > lo)) hi = lo + 1.0;  // 退化（NaN 混入など）でも軸を潰さない
    const double margin = 0.02 * (hi - lo);
    z_lo_               = static_cast<float>(lo - margin);
    z_hi_               = static_cast<float>(hi + margin);
}

/// 高さ・色に渡す値（ヘッダ「V のスケール」）。符号付き対数は狭義単調なので、面の上下関係も
/// 等高線の順序も保たれる（目盛りの間隔だけが対数になる）。
float HjbPanel::height_of(float v) const noexcept {
    if (!log_height_) return v;
    return std::copysign(std::log1p(std::abs(v)), v);
}

void HjbPanel::rebuild_mesh() {
    // x 軸は ln w（対数等間隔の格子が等間隔に並ぶ）。y 軸は −t（昇順、t = 0 の「今日」が最大 = 手前）。
    // どちらも狭義単調でなければ等間隔軸へ落とす（法線の差分商が 0 除算になる）。
    bool strict_x = true;
    for (std::size_t i = 1; i < Surf::kW && strict_x; ++i)
        strict_x = surface_.wealth[i] > surface_.wealth[i - 1] && surface_.wealth[i - 1] > 0.f;
    for (std::size_t i = 0; i < Surf::kW; ++i) {
        mesh_x_[i] = strict_x ? std::log(surface_.wealth[i])
                              : static_cast<float>(i) / static_cast<float>(Surf::kW - 1);
    }

    bool strict_y = true;
    for (std::size_t i = 1; i < Surf::kT && strict_y; ++i)
        strict_y = surface_.times[i] < surface_.times[i - 1];
    if (strict_y) {
        for (std::size_t i = 0; i < Surf::kT; ++i) mesh_y_[i] = -surface_.times[i];
    } else {
        const float t0 = surface_.times[0];
        for (std::size_t i = 0; i < Surf::kT; ++i)
            mesh_y_[i] = -t0 * (1.f - static_cast<float>(i) / static_cast<float>(Surf::kT - 1));
    }

    // 未計算行はモデルがゼロで寄越すが、V < 0（γ > 1）だとそのゼロ平面が計算済みの領域の**上**に来て、
    // 斜め上から見たときに解を隠してしまう（掃引がどこまで来たかが見えない）。描画ではスケールの
    // 下端に落とす: 色は最も暗い端になり「まだ無い」ことが一目で分かるうえ、データを隠さない。
    // γ < 1（V > 0）でも同じ規約で floor に落ちるので、見え方が符号で変わらない。
    const std::size_t filled  = surface_.filled_rows;
    const float       floor_h = height_of(z_lo_);
    for (std::size_t row = 0; row < Surf::kT; ++row) {
        const bool computed = row < filled;
        for (std::size_t j = 0; j < Surf::kW; ++j) {
            const std::size_t i = row * Surf::kW + j;
            mesh_z_[i]          = computed ? height_of(surface_.values[i]) : floor_h;
        }
    }

    mesh_.set_axes(std::span<const float>(mesh_x_), std::span<const float>(mesh_y_));
    mesh_.set_z(std::span<const float>(mesh_z_));
    mesh_.update_normals();

    mesh_key_   = scale_key_;  // このメッシュがどの面・どのレンジを写したか
    mesh_rows_  = surface_.filled_rows;
    mesh_valid_ = true;
}

// ---------------------------------------------------------------- V(w, t) surface (3D)
void HjbPanel::draw_surface() {
    ImGui::SetNextWindowSize(ImVec2(760, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, kPanelTop), ImGuiCond_FirstUseEver);
    // ホイールは 3D のズームに使うので、ウィンドウ側のスクロールには渡さない。
    ImGui::Begin("V(w, t) surface (3D)", nullptr,
                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    if (!glapi::gl_available()) {
        ImGui::TextDisabled("OpenGL functions unavailable - the 3D view is disabled");
        ImGui::TextDisabled("(gl_load() failed at startup; see stderr for the missing entry point)");
        ImGui::End();
        return;
    }

    if (!renderer_.ready() && !init_tried_) {
        init_tried_ = true;
        if (!renderer_.init(Surf::kW, Surf::kT))
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

    if (view_.draw(renderer_, mesh_, ImGui::GetContentRegionAvail(), height_of(z_lo_), height_of(z_hi_),
                   mesh_dirty_))
        mesh_dirty_ = false;  // 実際に VBO へ送れたフレームだけ下ろす

    ImGui::End();
}

// ---------------------------------------------------------------- V(w) now
void HjbPanel::draw_value() {
    ImGui::SetNextWindowSize(ImVec2(375, 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, 442), ImGuiCond_FirstUseEver);
    ImGui::Begin("V(w) now");

    if (received_ == 0) {
        ImGui::TextDisabled("%s", kWaitingSnapshot);
        ImGui::End();
        return;
    }

    if (ImPlot::BeginPlot("##vw", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("w (wealth, log)", "V", 0, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);  // 格子が対数等間隔（ヘッダ「w 軸は対数」）
        ImPlot::SetupAxisLimits(ImAxis_X1, last_.wealth.front(), last_.wealth.back(), ImPlotCond_Always);
        ImPlot::SetupLegend(ImPlotLocation_SouthEast);

        ImPlot::PlotLine("V(w) HJB", last_.wealth.data(), last_.values.data(), kNodeCount);
        ImPlot::SetNextLineStyle(ImVec4(1.00f, 0.60f, 0.20f, 0.9f), 1.5f);
        ImPlot::PlotLine("closed form", last_.wealth.data(), last_.analytic_v.data(), kNodeCount);
        ImPlot::EndPlot();
    }
    ImGui::End();
}

// ---------------------------------------------------------------- pi*(w)
void HjbPanel::draw_policy() {
    ImGui::SetNextWindowSize(ImVec2(375, 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(395, 442), ImGuiCond_FirstUseEver);
    ImGui::Begin("pi*(w)");

    if (received_ == 0) {
        ImGui::TextDisabled("%s", kWaitingSnapshot);
        ImGui::End();
        return;
    }

    ImGui::Text("max |err| %.1e", last_.max_abs_pi_error);  // 窓は内側 90 %（Control のテレメトリに明記）
    ImGui::SameLine();
    ImGui::Checkbox("zoom to the spread", &pi_zoom_);

    if (ImPlot::BeginPlot("##pi", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("w (wealth, log)", "pi*", 0, pi_zoom_ ? ImPlotAxisFlags_AutoFit : 0);
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
        ImPlot::SetupAxisLimits(ImAxis_X1, last_.wealth.front(), last_.wealth.back(), ImPlotCond_Always);
        // 既定の Y 範囲は解析値まわりに固定する。AutoFit にすると差が 1e-5 の 2 本の線に軸が
        // 張り付いて、「π* は定数」という肝心の絵が見えなくなる（見たい人は zoom を押す）。
        if (!pi_zoom_)
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, std::max(1.6 * last_.analytic_pi[0], kPiAxisFloor),
                                    ImPlotCond_Always);
        ImPlot::SetupLegend(ImPlotLocation_SouthEast);

        ImPlot::PlotLine("pi*(w) HJB", last_.wealth.data(), last_.pi_star.data(), kNodeCount);
        ImPlot::SetNextLineStyle(ImVec4(1.00f, 0.60f, 0.20f, 0.9f), 1.5f);
        ImPlot::PlotLine("analytic", last_.wealth.data(), last_.analytic_pi.data(), kNodeCount);
        ImPlot::EndPlot();
    }
    ImGui::End();
}

// ---------------------------------------------------------------- Controls → Command
void HjbPanel::draw_controls(Runner& runner) {
    using bridge::Command;
    using scenes::HjbModel;

    ImGui::SetNextWindowSize(ImVec2(420, 690), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(780, kPanelTop), ImGuiCond_FirstUseEver);
    ImGui::Begin("Control");

    // パラメータを変えると掃引は満期からやり直しになる（HJB の途中で係数は差し替えられない）。
    // 手元の面を先回りで捨てる必要は無い: R10 により一時停止中でも新しい Snapshot と面が 1 組届き、
    // seq / iteration の巻き戻りを見て `ingest()` が古い面を捨てる（同じフレームで入れ替わる）。
    //
    // スライダーは AlwaysClamp: Ctrl+クリックのテキスト入力はスライダーの範囲を無視するので、
    // 付けないと UI の値だけがモデルのクランプ後の値と食い違ったままになる。
    ImGui::SeparatorText("Model (changing any of these restarts the sweep)");
    constexpr ImGuiSliderFlags kClamp = ImGuiSliderFlags_AlwaysClamp;
    if (ImGui::SliderFloat("mu (drift)", &mu_, static_cast<float>(HjbModel::kMinMu),
                           static_cast<float>(HjbModel::kMaxMu), "%.4f", kClamp)) {
        mu_ = std::clamp(mu_, static_cast<float>(HjbModel::kMinMu), static_cast<float>(HjbModel::kMaxMu));
        runner.send(Command::set_param(HjbModel::kMu, static_cast<double>(mu_)));
    }
    if (ImGui::SliderFloat("r (risk-free)", &r_, static_cast<float>(HjbModel::kMinRate),
                           static_cast<float>(HjbModel::kMaxRate), "%.4f", kClamp)) {
        r_ = std::clamp(r_, static_cast<float>(HjbModel::kMinRate), static_cast<float>(HjbModel::kMaxRate));
        runner.send(Command::set_param(HjbModel::kRate, static_cast<double>(r_)));
    }
    if (ImGui::SliderFloat("sigma", &sigma_, static_cast<float>(HjbModel::kMinSigma),
                           static_cast<float>(HjbModel::kMaxSigma), "%.3f", kClamp)) {
        sigma_ = std::clamp(sigma_, static_cast<float>(HjbModel::kMinSigma),
                            static_cast<float>(HjbModel::kMaxSigma));
        runner.send(Command::set_param(HjbModel::kSigma, static_cast<double>(sigma_)));
    }
    if (ImGui::SliderFloat("gamma (CRRA)", &gamma_, static_cast<float>(HjbModel::kMinGamma),
                           static_cast<float>(HjbModel::kMaxGamma), "%.3f", kClamp)) {
        gamma_ = std::clamp(gamma_, static_cast<float>(HjbModel::kMinGamma),
                            static_cast<float>(HjbModel::kMaxGamma));
        runner.send(Command::set_param(HjbModel::kGamma, static_cast<double>(gamma_)));
    }
    ImGui::TextDisabled("gamma near 1 snaps to log utility;  pi capped at %.0f", core::kHjbPiMax);
    ImGui::TextDisabled("1 Step = 1 backward iteration;  surface %u x %u",
                        static_cast<unsigned>(Surf::kW), static_cast<unsigned>(Surf::kT));

    ImGui::SeparatorText("View");
    ImGui::SliderInt("contour lines", &view_.contour_lines, 0, 40);
    ImGui::Checkbox("wireframe", &view_.wire);
    ImGui::SameLine();
    // 高さの写像を切り替えたら、次の面が届くのを待たずにその場でメッシュを作り直す
    // （一時停止中は面が来ないので、待つと「効いていない」ように見える）。
    if (ImGui::Checkbox("log height", &log_height_) && have_surface_ && glapi::gl_available()) {
        rebuild_mesh();
        mesh_dirty_ = true;
    }
    if (ImGui::Button("Reset view", ImVec2(100, 0))) view_.camera = view_.home;
    ImGui::SameLine();
    ImGui::TextDisabled("drag / wheel / R on the 3D image");

    // Reset は掃引を満期へ巻き戻す。このシーンに描画側の履歴は無く、巻き戻った Snapshot と面は
    // R10 が次のフレームまでに届けるので、ここで捨てるものは無い。
    draw_clock_controls(clock_, runner, [] {});

    ImGui::SeparatorText("Telemetry");
    draw_runner_telemetry(runner, last_.seq, rate_.per_second());
    if (received_ == 0) {
        ImGui::TextDisabled("%s", kWaitingSnapshot);
    } else {
        const unsigned n_t = last_.iteration + last_.remaining;
        const double   tau = std::max(0.0, last_.T - last_.t_remaining);
        ImGui::Text("iteration      %u / %u   (remaining %u)", last_.iteration, n_t, last_.remaining);
        ImGui::Text("t (sweep left) %.4f y", last_.t_remaining);
        ImGui::Text("tau solved     %.4f y  (%.1f trading days)", tau, tau * kTradingDays);
        // 実効 γ（スライダーは 1.00005 のままでも、モデルは吸着後の 1.0 で走っている）。
        ImGui::Text("gamma (model)  %.6f%s", last_.gamma, last_.gamma == 1.0 ? "   log utility" : "");
        ImGui::Text("pi* analytic   %.6f", last_.analytic_pi[0]);
        ImGui::Text("pi* at w mid   %.6f", last_.pi_star[Snap::kNodes / 2]);
        ImGui::Text("max |pi err|   %.3e  (inner 90%%)", last_.max_abs_pi_error);
        // V の相対誤差も同じ窓で（境界は Dirichlet で閉形式そのものなので誤差 0 になり、見ても意味がない）。
        double v_err = 0.0;
        for (std::size_t i = kEdge; i + kEdge < Snap::kNodes; ++i) {
            const double ref = last_.analytic_v[i];
            if (std::abs(ref) > 0.0)
                v_err = std::max(v_err, std::abs(last_.values[i] - ref) / std::abs(ref));
        }
        ImGui::Text("max |V rel err| %.3e  (inner 90%%)", v_err);
        ImGui::Text("V(w mid)       %.6f  (closed form %.6f)", last_.values[Snap::kNodes / 2],
                    last_.analytic_v[Snap::kNodes / 2]);
        ImGui::Text("surfaces       %llu  (rows %u / %u)", static_cast<unsigned long long>(surfaces_),
                    surface_.filled_rows, static_cast<unsigned>(Surf::kT));
        ImGui::Text("fps            %.1f", static_cast<double>(ImGui::GetIO().Framerate));
    }

    ImGui::End();
}

// ---------------------------------------------------------------- シーン生成
std::unique_ptr<Scene> make_hjb_scene() {
    // Snapshot が ≈ 10 KB なのでリングは小さく。面は TripleBuffer（3 枚 ≈ 480 KB）なので、
    // RunnerScene ごと make_unique でヒープに置かれることが前提（`scene_registry.hpp`）。
    constexpr std::size_t kSnapCap = 64;

    scenes::HjbModel::Config model_cfg;  // 256 節点 × 200 ステップ、μ=8 %, r=3 %, σ=20 %, γ=3

    bridge::RunnerConfig run_cfg;
    run_cfg.dt                     = 1.0 / 252.0;  // モデルは使わない（刻みは時間格子が決める）
    run_cfg.clock.steps_per_second = 20.0;         // 1 ステップ = 1 反復。目で追える速さ
    run_cfg.clock.speed            = 1.0;
    run_cfg.publish_every          = 1;
    run_cfg.surface_every          = 1;

    // Panel は in-place 構築（面 + メッシュで 2 MB 超をスタックに積まない）。
    return std::make_unique<RunnerScene<scenes::HjbModel, HjbPanel, kSnapCap>>(
        scenes::HjbModel{model_cfg}, run_cfg, std::in_place, model_cfg, run_cfg.clock.speed);
}

}  // namespace quantviz::viz
