#include "panels/greeks_panel.hpp"

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

constexpr int kStrikeCount = static_cast<int>(scenes::GreeksSnapshot::kStrikes);

// 表示用の単位換算。Snapshot は「1.0 あたり」で持つので、ここで実務単位に直す。こうすると
// delta / vega / rho は O(1)、gamma / theta は O(0.01) にまとまり、2 軸できれいに分かれる。
constexpr double kVegaPerVolPoint = 1.0 / 100.0;  ///< ボラ 1 ポイント = 0.01
constexpr double kRhoPerPercent   = 1.0 / 100.0;  ///< 金利 1 % = 0.01
constexpr double kThetaPerDay     = 1.0 / 365.0;  ///< 1 日 = 1/365 年
// 注意: theta の「1 日」は暦日（1/365 年）で、実務の見積りに合わせている。一方 Telemetry の `t` と
// Streaming シーンの時間軸は取引日（1/252 年）。同じ「日」でも分母が違うので、両者を足し引きしないこと。

}  // namespace

GreeksPanel::GreeksPanel(const scenes::GreeksModel::Config& initial, double initial_speed)
    : clock_{static_cast<float>(initial_speed), false},
      r_(static_cast<float>(initial.r)),
      sigma_(static_cast<float>(initial.sigma)),
      maturity_(static_cast<float>(initial.maturity)),
      strike_span_(static_cast<float>(initial.strike_span)) {
    // 初期視点: S 軸の手前・斜め上から。尾根（S = K）が奥行き方向に走って見える角度。R キーでここへ戻る。
    view_.camera.distance = 3.0f;
    view_.camera.yaw      = 0.85f;
    view_.camera.pitch    = 0.50f;
    view_.home            = view_.camera;
}

void GreeksPanel::draw(Runner& runner) {
    ingest(runner);
    draw_strip();
    draw_surface();
    draw_controls(runner);
}

// ---------------------------------------------------------------- Snapshot 取り込み
void GreeksPanel::ingest(Runner& runner) {
    scenes::GreeksSnapshot s;
    bool                   got = false;
    while (runner.poll(s)) {
        // Streaming と違い seq == 0 の Snapshot も捨てずに採る: このシーンの配列はコンストラクタと
        // reset() で完全に埋まっており（軸・ストリップ・Γ 面すべて有限）、汚染される History も無い。
        // むしろ Reset 直後の 1 枚を採らないと、再開までパネルが古い状態を映し続けてしまう。
        last_ = s;
        ++received_;
        got = true;
        // 他の 3 パネルの `prev_seq_` ガード（seq が巻き戻ったら History を捨てる）に相当する
        // 処理はここには無い: 描くのは常に最新の 1 枚だけで、Reset をまたいで溜まる状態が無い。
    }

    rate_.sample(received_, now_seconds());

    // Γ 面の再構築は「変わったときだけ」。Snapshot は毎ステップ（100 /s）流れてくるが、
    // gamma_surface は (r, sigma, T) にしか依存せず、Model 側もこの 3 つが動いたときだけ
    // 張り直す（`GreeksModel::commit_pending`）。同じ判定をここでも行い、100 回/秒の
    // 転置 + 法線 + VBO 更新を避ける。Reset は 3 つを保つので面は同じ = 上げ直し不要。
    if (!got) return;
    if (surf_valid_ && last_.r == surf_r_ && last_.sigma == surf_sigma_ && last_.T == surf_T_) return;
    rebuild_mesh();
}

// ---------------------------------------------------------------- Γ 格子 → SurfaceMesh
void GreeksPanel::rebuild_mesh() {
    for (std::size_t i = 0; i < Snap::kGridS; ++i) mesh_xs_[i] = static_cast<float>(last_.grid_s[i]);
    for (std::size_t j = 0; j < Snap::kGridT; ++j) mesh_ys_[j] = static_cast<float>(last_.grid_t[j]);

    // [iS][iT]（Snapshot）→ [iT][iS]（SurfaceMesh は index = iy * n_x + ix）。
    // z 範囲はデータから取る。sigma = 0 など縮退した設定で非有限が混じっても面を黒く落とさないよう、
    // 非有限は 0 に潰す（`set_z` は有限であることを前提にしており、Debug では assert で落ちる）。
    float lo    = 0.f;
    float hi    = 0.f;
    bool  first = true;
    for (std::size_t j = 0; j < Snap::kGridT; ++j) {
        for (std::size_t i = 0; i < Snap::kGridS; ++i) {
            float v = static_cast<float>(last_.gamma_surface[i * Snap::kGridT + j]);
            if (!std::isfinite(v)) v = 0.f;
            mesh_z_[j * Snap::kGridS + i] = v;
            if (first) {
                lo    = v;
                hi    = v;
                first = false;
            } else {
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
        }
    }
    // 平坦な面（全部同じ値）で高さ正規化が潰れないように、幅ゼロなら 1 だけ開けておく。
    if (!(hi > lo)) hi = lo + 1.f;
    z_min_ = lo;
    z_max_ = hi;

    mesh_.set_axes(std::span<const float>(mesh_xs_), std::span<const float>(mesh_ys_));
    mesh_.set_z(std::span<const float>(mesh_z_));
    mesh_.update_normals();

    surf_r_     = last_.r;
    surf_sigma_ = last_.sigma;
    surf_T_     = last_.T;
    surf_valid_ = true;
    gpu_dirty_  = true;  // 実際の VBO 更新は 3D タブが描かれるフレームまで遅らせる
}

std::size_t GreeksPanel::atm_index() const noexcept {
    const double dk = last_.strikes[1] - last_.strikes[0];
    if (!(dk > 0.0)) return 0;
    const double raw = std::round((last_.spot - last_.strikes[0]) / dk);
    // NaN を size_t にキャストすると UB。`!(raw >= 0.0)` は NaN と負をまとめて弾く
    // （std::clamp は NaN をそのまま返すので、ここでは使えない）。
    if (!(raw >= 0.0)) return 0;
    const double hi = static_cast<double>(Snap::kStrikes - 1);
    return static_cast<std::size_t>(raw < hi ? raw : hi);
}

// ---------------------------------------------------------------- Greeks vs K
void GreeksPanel::draw_strip() {
    ImGui::SetNextWindowSize(ImVec2(760, 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, kPanelTop), ImGuiCond_FirstUseEver);
    ImGui::Begin("Greeks vs K");

    if (received_ == 0) {
        ImGui::TextDisabled("waiting for the first snapshot ...");
        ImGui::End();
        return;
    }

    for (std::size_t i = 0; i < Snap::kStrikes; ++i) {
        vega_disp_[i]  = last_.vega[i] * kVegaPerVolPoint;
        rho_disp_[i]   = last_.rho[i] * kRhoPerPercent;
        theta_disp_[i] = last_.theta[i] * kThetaPerDay;
    }

    if (ImPlot::BeginPlot("##greeks", ImVec2(-1, -1))) {
        // 実務単位に直すと桁が 2 つに分かれる: O(1) の delta / vega / rho と O(0.01) の gamma / theta。
        // 同じ軸に置くと gamma が潰れて読めないので Y2 を立てる。
        // K 軸も AutoFit にする: strike span を変えたときに表示範囲が付いてこないと、
        // 新しいストライクが画面外に出てしまう（軸は既定では初回だけフィットする）。
        ImPlot::SetupAxes("K (strike)", "delta, vega, rho", ImPlotAxisFlags_AutoFit,
                          ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxis(ImAxis_Y2, "gamma, theta", ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_AutoFit);
        ImPlot::SetupLegend(ImPlotLocation_North, ImPlotLegendFlags_Horizontal | ImPlotLegendFlags_Outside);

        ImPlot::SetAxis(ImAxis_Y1);
        ImPlot::PlotLine("delta", last_.strikes.data(), last_.delta.data(), kStrikeCount);
        ImPlot::PlotLine("vega /vol pt", last_.strikes.data(), vega_disp_.data(), kStrikeCount);
        ImPlot::PlotLine("rho /1%", last_.strikes.data(), rho_disp_.data(), kStrikeCount);

        ImPlot::SetAxis(ImAxis_Y2);
        ImPlot::PlotLine("gamma", last_.strikes.data(), last_.gamma.data(), kStrikeCount);
        ImPlot::PlotLine("theta /day", last_.strikes.data(), theta_disp_.data(), kStrikeCount);

        // 現在のスポット位置に縦線（Γ のピークがここに来る）。
        ImPlot::SetAxis(ImAxis_Y1);
        const double spot = last_.spot;
        ImPlot::PlotInfLines("spot", &spot, 1);
        ImPlot::EndPlot();
    }
    ImGui::End();
}

// ---------------------------------------------------------------- Gamma(S, T): 3D / heatmap
// 同じ面を 2 通りに見せるだけなので、ウィンドウを増やさずタブで切り替える（M2 Task 8）。
// 縦 760 px の画面で「ストリップ + 面 + Control」を並べると 3 段目は 200 px を切り、3D では
// 尾根が潰れて読めない。タブなら 3D にウィンドウ 1 枚ぶんの高さを丸ごと渡せる。
void GreeksPanel::draw_surface() {
    ImGui::SetNextWindowSize(ImVec2(760, 410), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, kPanelTop + 310), ImGuiCond_FirstUseEver);
    // ホイールは 3D のズームに使うので、ウィンドウ側のスクロールには渡さない。
    ImGui::Begin("Gamma(S, T) surface", nullptr,
                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    if (received_ == 0) {
        ImGui::TextDisabled("waiting for the first snapshot ...");
        ImGui::End();
        return;
    }

    if (ImGui::BeginTabBar("##gsurface_tabs")) {
        if (ImGui::BeginTabItem("3D")) {
            draw_surface_3d();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("heatmap")) {
            draw_surface_heatmap();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

// ---------------------------------------------------------------- 「3D」タブ
// GL が使えない環境（`gl_load()` 失敗・FBO を作れない等）でも viewer は落とさない。3D の中身が
// テキストに落ちるだけで、heatmap タブも時計も Telemetry もそのまま動く。
void GreeksPanel::draw_surface_3d() {
    if (!glapi::gl_available()) {
        ImGui::TextDisabled("OpenGL functions unavailable - the 3D view is disabled");
        ImGui::TextDisabled("(gl_load() failed at startup; see stderr for the missing entry point)");
        ImGui::TextDisabled("use the \"heatmap\" tab for the same surface in 2D");
        return;
    }

    if (!renderer_.ready() && !init_tried_) {
        init_tried_ = true;
        if (!renderer_.init(Snap::kGridS, Snap::kGridT))
            std::fprintf(stderr, "SurfaceRenderer::init failed: %s\n", renderer_.last_error().c_str());
        gpu_dirty_ = true;  // 新しい VBO は空なので、初期化直後は必ず 1 回上げる
    }
    if (!renderer_.ready()) {
        ImGui::TextDisabled("3D renderer unavailable:");
        ImGui::TextWrapped("%s", renderer_.last_error().c_str());
        return;
    }

    // `SurfaceView::draw` は「実際に upload したか」を返す（領域が小さすぎる等で描かなかった
    // フレームは false）。下ろすのは true のときだけ: そうしないと、描けなかったフレームで
    // 更新が永久に失われる。uploads_ もこの返値だけで数えるので、Telemetry の数字は
    // 「VBO へ送った回数」そのものになる。
    if (view_.draw(renderer_, mesh_, ImGui::GetContentRegionAvail(), z_min_, z_max_, gpu_dirty_)) {
        gpu_dirty_ = false;
        ++uploads_;
    }
}

// ---------------------------------------------------------------- 「heatmap」タブ
void GreeksPanel::draw_surface_heatmap() {
    // [iS][iT] → [row = T 降順][col = S 昇順]。ImPlot は row 0 を上端（bounds_max.y）に描くので、
    // row 0 に T_max を置くと T 軸が上向きになる。
    double vmax = 0.0;
    for (std::size_t row = 0; row < Snap::kGridT; ++row) {
        const std::size_t it = Snap::kGridT - 1 - row;
        for (std::size_t is = 0; is < Snap::kGridS; ++is) {
            const double v                 = last_.gamma_surface[is * Snap::kGridT + it];
            heat_[row * Snap::kGridS + is] = v;
            vmax                           = std::max(vmax, v);
        }
    }
    if (!(vmax > 0.0)) vmax = 1.0;  // sigma = 0 などで全ゼロのとき、色スケールを潰さない

    // ImPlot はセルを「等分した矩形」として描くのでセル中心は bmin + (k + 0.5)·range/N にある。
    // 一方サンプルは grid_s[k]（間隔 range/(N-1)）に載っている。bounds をサンプル範囲そのものにすると
    // 半セル分ずれ、Γ の尾根が spot の縦線から外れて見える。半セルずつ外へ広げて中心を一致させる。
    const double ds = (last_.grid_s.back() - last_.grid_s.front()) / static_cast<double>(Snap::kGridS - 1);
    const double dt = (last_.grid_t.back() - last_.grid_t.front()) / static_cast<double>(Snap::kGridT - 1);
    const ImPlotPoint bmin(last_.grid_s.front() - 0.5 * ds, last_.grid_t.front() - 0.5 * dt);
    const ImPlotPoint bmax(last_.grid_s.back() + 0.5 * ds, last_.grid_t.back() + 0.5 * dt);

    ImPlot::PushColormap(ImPlotColormap_Viridis);
    if (ImPlot::BeginPlot("##gsurface", ImVec2(-80, -1), ImPlotFlags_NoLegend)) {
        constexpr ImPlotAxisFlags kAxis = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines;
        ImPlot::SetupAxes("S (spot)", "T (years to expiry)", kAxis, kAxis);
        ImPlot::SetupAxisLimits(ImAxis_X1, bmin.x, bmax.x, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, bmin.y, bmax.y, ImPlotCond_Always);
        ImPlot::PlotHeatmap("gamma", heat_.data(), static_cast<int>(Snap::kGridT),
                            static_cast<int>(Snap::kGridS), 0.0, vmax, nullptr, bmin, bmax);
        // 既定色は現在のカラーマップ（Viridis）の 0 番 = 暗紫で、背景に埋もれる。白で明示する。
        const double spot = last_.spot;
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), 1.5f);
        ImPlot::PlotInfLines("spot", &spot, 1);
        ImPlot::EndPlot();
    }
    ImGui::SameLine();
    ImPlot::ColormapScale("##gscale", 0.0, vmax, ImVec2(70, -1), "%.3f");
    ImPlot::PopColormap();
}

// ---------------------------------------------------------------- Controls → Command
void GreeksPanel::draw_controls(Runner& runner) {
    using bridge::Command;
    using scenes::GreeksModel;

    ImGui::SetNextWindowSize(ImVec2(420, 720), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(780, kPanelTop), ImGuiCond_FirstUseEver);
    ImGui::Begin("Control");

    ImGui::SeparatorText("Option parameters (applied immediately)");
    if (ImGui::SliderFloat("r (rate)", &r_, -0.05f, 0.15f, "%.4f"))
        runner.send(Command::set_param(GreeksModel::kRate, static_cast<double>(r_)));
    if (ImGui::SliderFloat("sigma (pricing vol)", &sigma_, 0.01f, 1.00f, "%.3f"))
        runner.send(Command::set_param(GreeksModel::kSigma, static_cast<double>(sigma_)));
    if (ImGui::SliderFloat("T (years)", &maturity_, 0.05f, 3.00f, "%.3f"))
        runner.send(Command::set_param(GreeksModel::kMaturity, static_cast<double>(maturity_)));
    if (ImGui::SliderFloat("strike span", &strike_span_, 0.10f, 0.90f, "%.2f"))
        runner.send(Command::set_param(GreeksModel::kStrikeSpan, static_cast<double>(strike_span_)));
    ImGui::TextDisabled("K axis = S0 * [1 - span, 1 + span];  surface K = S0");

    ImGui::SeparatorText("Manual spot shock");
    if (ImGui::Button("Spot x1.05", ImVec2(100, 0)))
        runner.send(Command::set_param(GreeksModel::kSpotJump, 1.05));
    ImGui::SameLine();
    if (ImGui::Button("Spot x0.95", ImVec2(100, 0)))
        runner.send(Command::set_param(GreeksModel::kSpotJump, 0.95));

    ImGui::SeparatorText("3D view (Gamma tab)");
    ImGui::SliderInt("contour lines", &view_.contour_lines, 0, 40);
    ImGui::Checkbox("wireframe", &view_.wire);
    if (ImGui::Button("Reset view", ImVec2(100, 0))) view_.camera = view_.home;
    ImGui::SameLine();
    ImGui::TextDisabled("drag / wheel / R on the 3D image");

    // Reset で消す History を持たないシーンなので on_reset は何もしない（次の Snapshot で全部入れ替わる）。
    draw_clock_controls(clock_, runner, [] {});

    ImGui::SeparatorText("Telemetry");
    draw_runner_telemetry(runner, last_.seq, rate_.per_second());
    if (received_ == 0) {
        // 既定構築の Snapshot（全ゼロ）をシーン固有の行として出すと「spot 0」等の嘘になる。
        ImGui::TextDisabled("waiting for the first snapshot ...");
    } else {
        const std::size_t atm = atm_index();
        ImGui::Text("t              %.3f y  (%.2f trading days)", last_.t, last_.t * kTradingDays);
        ImGui::Text("spot           %.4f", last_.spot);
        ImGui::Text("ATM strike     %.4f", last_.strikes[atm]);
        ImGui::Text("ATM gamma      %.6f", last_.gamma[atm]);
        ImGui::Text("ATM delta      %.6f", last_.delta[atm]);
        ImGui::Text("ATM vega/pt    %.6f", last_.vega[atm] * kVegaPerVolPoint);
        ImGui::Text("ATM theta/day  %.6f", last_.theta[atm] * kThetaPerDay);
        // 面は (r, sigma, T) が動いたときだけ作り直して GPU へ上げる。Snapshot は毎ステップ
        // 来るので、この数字が snapshots/s と一緒に伸びていたら間引きが壊れている。
        ImGui::Text("uploads        %llu", static_cast<unsigned long long>(uploads_));
        ImGui::Text("gamma z range  [%.4f, %.4f]", static_cast<double>(z_min_),
                    static_cast<double>(z_max_));
        ImGui::Text("surface grid   %zu x %zu  (%zu triangles)", Snap::kGridS, Snap::kGridT,
                    mesh_.triangle_count());
    }

    ImGui::End();
}

// ---------------------------------------------------------------- シーン生成（main.cpp / SceneRegistry 用）
std::unique_ptr<Scene> make_greeks_scene() {
    constexpr double      kDt      = 1.0 / (252.0 * 390.0);  // 1 分足（年単位）
    constexpr std::size_t kSnapCap = 256;                    // Snapshot が 16 KB 強なのでリングは小さく

    scenes::GreeksModel::Config model_cfg;
    model_cfg.gbm         = {100.0, 0.05, 0.20};  // s0, mu, sigma（パスのボラティリティ）
    model_cfg.r           = 0.02;
    model_cfg.sigma       = 0.20;  // 価格付けのボラティリティ
    model_cfg.maturity    = 1.00;
    model_cfg.strike_span = 0.40;
    model_cfg.seed        = 42;

    bridge::RunnerConfig run_cfg;
    run_cfg.dt                     = kDt;
    run_cfg.clock.steps_per_second = 100.0;  // Snapshot が大きいので Streaming より控えめに
    run_cfg.clock.speed            = 1.0;
    run_cfg.publish_every          = 1;

    // Panel は in-place 構築（Snapshot 1 枚 + 表示バッファで 28 KB 超をスタックに積まない）。
    return std::make_unique<RunnerScene<scenes::GreeksModel, GreeksPanel, kSnapCap>>(
        scenes::GreeksModel{model_cfg}, run_cfg, std::in_place, model_cfg, run_cfg.clock.speed);
}

}  // namespace quantviz::viz
