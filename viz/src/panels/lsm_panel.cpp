#include "panels/lsm_panel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <utility>

#include <imgui.h>
#include <implot.h>

#include "panels/clock_panel.hpp"
#include "panels/panel_common.hpp"

namespace quantviz::viz {

namespace {

using scenes::LsmModel;
using scenes::LsmSceneSnapshot;

constexpr int kPathPoints = static_cast<int>(LsmSceneSnapshot::kPts);
constexpr int kFitPoints  = static_cast<int>(LsmSceneSnapshot::kFit);

constexpr ImVec4 kPathColor{0.62f, 0.68f, 0.80f, 0.30f};  ///< 16 本のパス（帯と中央値を邪魔しない薄さ）
constexpr ImVec4 kBandColor{0.35f, 0.60f, 0.95f, 1.00f};  ///< 分位帯
constexpr ImVec4 kExerciseColor{1.00f, 0.62f, 0.22f, 1.0f};  ///< 早期行使のマーカーと交点
constexpr ImVec4 kNowColor{0.85f, 0.85f, 0.90f, 0.90f};      ///< 現在の時点の縦線
constexpr ImVec4 kWarnColor{1.00f, 0.45f, 0.35f, 1.00f};

constexpr const char* kWaitingSnapshot = "waiting for the first snapshot ...";

/// basis コンボの項目（`core::LsmBasis` の並び）。
constexpr const char* const kBasisNames[] = {"Power x^j", "Laguerre (weighted)"};

double intrinsic_of(const LsmSceneSnapshot& s, double spot) noexcept {
    return s.type == core::OptionType::Call ? std::max(spot - s.K, 0.0) : std::max(s.K - spot, 0.0);
}

}  // namespace

LsmPanel::LsmPanel(const scenes::LsmModel::Config& initial, double initial_speed)
    : clock_{static_cast<float>(initial_speed), false},
      strike_(static_cast<float>(initial.K)),
      sigma_(static_cast<float>(initial.sigma)),
      rate_slider_(static_cast<float>(initial.r)),
      basis_index_(initial.basis == core::LsmBasis::Laguerre ? 1 : 0),
      n_basis_(static_cast<int>(initial.n_basis)),
      n_paths_(static_cast<int>(initial.n_paths)) {
    // 最初の Snapshot が届く前のフレームでも軸が破綻しないように種を入れておく。
    last_.K     = initial.K;
    last_.T     = initial.T;
    last_.s0    = initial.s0;
    last_.sigma = initial.sigma;
    last_.r     = initial.r;
    last_.type  = initial.type;
    rebuild_curves();
}

void LsmPanel::draw(Runner& runner) {
    ingest(runner);
    draw_paths();
    draw_fit();
    draw_controls(runner);
}

// ---------------------------------------------------------------- Snapshot 取り込み
void LsmPanel::ingest(Runner& runner) {
    bool             got = false;
    LsmSceneSnapshot s;
    while (runner.poll(s)) {  // リングを空にして最新 1 枚だけを使う（このシーンに描画側の履歴は無い）
        last_ = s;
        ++received_;
        got = true;
    }
    rate_.sample(received_, now_seconds());
    if (!got) return;

    // 履歴を持たないので巻き戻し（Reset / パラメータ変更）の検出は要らない。届いた Snapshot から
    // 派生量をまるごと張り直せば、巻き戻りでも同じ seq の再送（R10: 一時停止中の SetParam）でも
    // 正しい絵になる（いずれも安い純関数）。
    rebuild_curves();
}

/// Snapshot だけから決まる派生量。毎フレームではなく Snapshot が来たフレームだけ張り直す。
void LsmPanel::rebuild_curves() {
    const double dt = last_.T / static_cast<double>(LsmSceneSnapshot::kSteps);
    for (std::size_t k = 0; k < LsmSceneSnapshot::kPts; ++k)
        times_[k] = dt * static_cast<double>(k);

    for (std::size_t j = 0; j < LsmSceneSnapshot::kFit; ++j)
        intrinsic_[j] = intrinsic_of(last_, last_.fit_s[j]);

    // 早期行使したパスだけマーカーを打つ（kSteps = 満期まで持った）。
    ex_count_ = 0;
    for (std::size_t i = 0; i < LsmSceneSnapshot::kShow; ++i) {
        const std::int32_t e = last_.shown_exercise[i];
        if (e < 0 || e >= static_cast<std::int32_t>(LsmSceneSnapshot::kSteps)) continue;
        const std::size_t k = static_cast<std::size_t>(e);
        ex_x_[static_cast<std::size_t>(ex_count_)] = times_[k];
        ex_y_[static_cast<std::size_t>(ex_count_)] = last_.shown_paths[i * LsmSceneSnapshot::kPts + k];
        ++ex_count_;
    }

    // フィットと本源的価値の交点 = 行使境界（h(S) > c(S) の境目）。符号が変わる最初の区間を
    // 低い S の側から探して線形補間する: put では h が左で大きく右で 0 なので、これが行使境界。
    // （このシーンは put 固定 —— 種類を変える Param は無く、`Config::type` を直に書き換えたときだけ
    //  call になる。無配当の American call はそもそも早期行使しない = 交点が出ないので、
    //  下の "above / below" の call 側の枝は実質到達しない。）
    crossing_s_ = std::numeric_limits<double>::quiet_NaN();
    crossing_v_ = std::numeric_limits<double>::quiet_NaN();
    if (!last_.fit_valid) return;
    for (std::size_t j = 0; j + 1 < LsmSceneSnapshot::kFit; ++j) {
        const double d0 = intrinsic_[j] - last_.fit_v[j];
        const double d1 = intrinsic_[j + 1] - last_.fit_v[j + 1];
        if (!std::isfinite(d0) || !std::isfinite(d1)) continue;
        if ((d0 > 0.0) == (d1 > 0.0)) continue;
        const double w = d0 / (d0 - d1);  // d0 と d1 は符号が違うので分母は 0 にならない
        crossing_s_    = last_.fit_s[j] + w * (last_.fit_s[j + 1] - last_.fit_s[j]);
        crossing_v_    = intrinsic_[j] + w * (intrinsic_[j + 1] - intrinsic_[j]);
        break;
    }
}

// ---------------------------------------------------------------- Paths
void LsmPanel::draw_paths() {
    ImGui::SetNextWindowSize(ImVec2(760, 420), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, kPanelTop), ImGuiCond_FirstUseEver);
    ImGui::Begin("Paths");

    if (received_ == 0) {
        ImGui::TextDisabled("%s", kWaitingSnapshot);
        ImGui::End();
        return;
    }

    ImGui::Checkbox("show the 16 sample paths", &show_paths_);
    ImGui::SameLine();
    ImGui::TextDisabled("| band = 5/25/50/75/95 %% of all %u paths", last_.n_paths);

    // Y は 5–95 % 帯（+ 8 % の余白）に合わせる。AutoFit だと σ を上げたときに 1 本の暴れたパスが
    // 縦軸を占領して、帯と中央値が下端の一区画に潰れる。枠から出たパスは切れるが、「どこを見る絵か」
    // は帯のほうなのでそちらを優先する。
    const double* const q05_row = last_.quantiles.data();
    const double* const q95_row =
        last_.quantiles.data() + (LsmSceneSnapshot::kQuant - 1) * LsmSceneSnapshot::kPts;
    double band_lo = q05_row[0];
    double band_hi = q95_row[0];
    for (std::size_t k = 0; k < LsmSceneSnapshot::kPts; ++k) {
        band_lo = std::min(band_lo, q05_row[k]);
        band_hi = std::max(band_hi, q95_row[k]);
    }
    const double margin = 0.08 * (band_hi - band_lo);
    if (!(margin > 0.0)) {  // σ = 0 では帯が線になる（NaN もここへ落ちる）
        band_lo -= 0.1 * last_.K + 1.0;
        band_hi += 0.1 * last_.K + 1.0;
    } else {
        band_lo -= margin;
        band_hi += margin;
    }

    if (ImPlot::BeginPlot("##paths", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("t (years)", "S", 0, 0);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, last_.T, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, band_lo, band_hi, ImPlotCond_Always);
        // 凡例は枠の外へ: 中に置くと左上のパスと帯に重なる（項目が 6 個あるので面積が要る）。
        ImPlot::SetupLegend(ImPlotLocation_North, ImPlotLegendFlags_Horizontal | ImPlotLegendFlags_Outside);

        // 主役は帯と中央値、パスは「母集団がどう散るか」の添え物なので、**先に**薄く敷いてから
        // 帯を重ねる。16 本は同じ色で描く（1 本ずつ色を変えると帯と中央値が読めなくなる）。凡例に
        // 出すのは先頭の 1 本だけで、残りは "##" 付きの一意な ID にして凡例から外す。
        if (show_paths_) {
            for (std::size_t i = 0; i < LsmSceneSnapshot::kShow; ++i) {
                char id[24];
                std::snprintf(id, sizeof(id), "##path%zu", i);
                ImPlot::SetNextLineStyle(kPathColor, 1.0f);
                ImPlot::PlotLine(i == 0 ? "sample paths" : id, times_.data(),
                                 last_.shown_paths.data() + i * LsmSceneSnapshot::kPts, kPathPoints);
            }
        }

        // 分位帯。ImPlot 0.16 の PlotShaded は count <= 1 で落ちるが、ここは固定長 65 点なので安全。
        const double* q05 = last_.quantiles.data() + 0 * LsmSceneSnapshot::kPts;
        const double* q25 = last_.quantiles.data() + 1 * LsmSceneSnapshot::kPts;
        const double* q50 = last_.quantiles.data() + 2 * LsmSceneSnapshot::kPts;
        const double* q75 = last_.quantiles.data() + 3 * LsmSceneSnapshot::kPts;
        const double* q95 = last_.quantiles.data() + 4 * LsmSceneSnapshot::kPts;
        ImPlot::SetNextFillStyle(kBandColor, 0.15f);
        ImPlot::PlotShaded("5-95 %", times_.data(), q05, q95, kPathPoints);
        ImPlot::SetNextFillStyle(kBandColor, 0.30f);
        ImPlot::PlotShaded("25-75 %", times_.data(), q25, q75, kPathPoints);
        ImPlot::SetNextLineStyle(kBandColor, 1.6f);
        ImPlot::PlotLine("median", times_.data(), q50, kPathPoints);
        if (ex_count_ > 0) {
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0f, kExerciseColor, 1.5f, kExerciseColor);
            ImPlot::PlotScatter("exercised", ex_x_.data(), ex_y_.data(), ex_count_);
        }

        const double k_line = last_.K;
        ImPlot::SetNextLineStyle(ImVec4(0.80f, 0.80f, 0.85f, 0.5f), 1.0f);
        ImPlot::PlotInfLines("K", &k_line, 1, ImPlotInfLinesFlags_Horizontal);
        const double t_now = last_.t_now;
        ImPlot::SetNextLineStyle(kNowColor, 1.5f);
        ImPlot::PlotInfLines("t now", &t_now, 1);
        ImPlot::EndPlot();
    }
    ImGui::End();
}

// ---------------------------------------------------------------- Continuation fit
void LsmPanel::draw_fit() {
    ImGui::SetNextWindowSize(ImVec2(760, 286), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, 462), ImGuiCond_FirstUseEver);
    ImGui::Begin("Continuation fit @ t");

    if (received_ == 0) {
        ImGui::TextDisabled("%s", kWaitingSnapshot);
        ImGui::End();
        return;
    }

    if (!last_.fit_valid && last_.remaining > 0) {
        ImGui::TextDisabled("no regression yet - step the sweep back from maturity");
    } else if (!last_.fit_valid) {
        // 掃引は終わっているのに一度も回帰できなかった = どの行使日にも ITM パスが無かった
        // （例: sigma = 0 の ATM put は r > 0 なら S が K の上へ行くだけ）。
        ImGui::TextDisabled("no ITM path at any exercise date - nobody exercises; try a higher K or sigma");
    } else if (last_.fit_stale) {
        // 既定の ATM では最後の時点（t = 0）に ITM パスが無く、回帰そのものが存在しない。
        // そのときは直近のフィットを出したままにする（lsm_model.hpp「フィットは保持する」）。
        ImGui::TextColored(kWarnColor, "held fit from t index %u", last_.fit_t_index);
        ImGui::SameLine();
        if (last_.itm_now == 0)
            ImGui::TextDisabled("- no ITM path at the current point, so there is no regression");
        else
            ImGui::TextDisabled("- the regression at the current point did not produce a fit");
    } else {
        ImGui::Text("fit at t index %u  (%u ITM paths, rank %u / %u)", last_.fit_t_index,
                    last_.itm_paths, last_.fit_rank, last_.n_basis);
    }

    if (std::isfinite(crossing_s_))
        ImGui::TextDisabled("exercise where h(S) > c(S): %s S = %.3f",
                            last_.type == core::OptionType::Put ? "below" : "above", crossing_s_);
    else
        ImGui::TextDisabled("h(S) and c(S) do not cross on this range - nobody exercises here");

    if (ImPlot::BeginPlot("##fit", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("S (spot at that point)", "value", ImPlotAxisFlags_AutoFit, 0);
        // Y は本源的価値の高さで固定する。AutoFit にすると OTM 側のフィットの外挿（回帰は ITM
        // パスだけで作るので S > K では意味を持たない）が下へ伸びて縦軸を占領し、肝心の交点の
        // 周りが潰れる。外挿は枠の外へ出て行くので「その先は当てにならない」ことも読める。
        double h_max = 0.0;
        for (std::size_t j = 0; j < LsmSceneSnapshot::kFit; ++j) h_max = std::max(h_max, intrinsic_[j]);
        const double y_max = h_max > 0.0 ? 1.15 * h_max : 1.0;
        ImPlot::SetupAxisLimits(ImAxis_Y1, -0.15 * y_max, y_max, ImPlotCond_Always);
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);
        // 回帰に使った ITM パスの散布は Snapshot に載らない（最大 20000 点）。代わりに回帰の結果
        // （フィット曲線）と、それと比べられる本源的価値、そして交点 = 行使境界を描く。
        ImPlot::PlotLine("intrinsic h(S)", last_.fit_s.data(), intrinsic_.data(), kFitPoints);
        ImPlot::PlotLine("continuation c(S)", last_.fit_s.data(), last_.fit_v.data(), kFitPoints);
        if (std::isfinite(crossing_s_)) {
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 7.0f, kExerciseColor, 1.5f, kExerciseColor);
            ImPlot::PlotScatter("h = c (exercise boundary)", &crossing_s_, &crossing_v_, 1);
        }
        const double k_line = last_.K;
        ImPlot::SetNextLineStyle(ImVec4(0.80f, 0.80f, 0.85f, 0.5f), 1.0f);
        ImPlot::PlotInfLines("K", &k_line, 1);
        ImPlot::EndPlot();
    }

    ImGui::End();
}

// ---------------------------------------------------------------- Controls -> Command
void LsmPanel::draw_controls(Runner& runner) {
    using bridge::Command;

    ImGui::SetNextWindowSize(ImVec2(430, 716), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(780, kPanelTop), ImGuiCond_FirstUseEver);
    ImGui::Begin("Control");

    // どのパラメータを変えてもパスを作り直して満期からやり直す（回帰の途中で係数は差し替えられない）。
    // 1 回の適用は N = 4000 で約 9 ms、N = 20000 で約 45 ms かかるので、**ドラッグ中は送らず、
    // 離した（または入力を確定した）フレームだけ送る**。毎フレーム送ると計算スレッドに再初期化の
    // 行列が積もって、掃引が進まなくなる。
    ImGui::SeparatorText("Contract (any change restarts the sweep)");
    ImGui::TextDisabled("sliders apply when released");
    ImGui::SliderFloat("K (strike)", &strike_, 50.0f, 200.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemDeactivatedAfterEdit())
        runner.send(Command::set_param(LsmModel::kStrike, static_cast<double>(strike_)));

    ImGui::SliderFloat("sigma", &sigma_, 0.01f, 1.00f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemDeactivatedAfterEdit())
        runner.send(Command::set_param(LsmModel::kSigma, static_cast<double>(sigma_)));

    ImGui::SliderFloat("r (rate)", &rate_slider_, -0.05f, 0.20f, "%.4f", ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemDeactivatedAfterEdit())
        runner.send(Command::set_param(LsmModel::kRate, static_cast<double>(rate_slider_)));

    ImGui::SeparatorText("Regression");
    if (ImGui::Combo("basis", &basis_index_, kBasisNames, 2)) {  // クリック 1 回なので即送る
        basis_index_ = std::clamp(basis_index_, 0, 1);
        runner.send(Command::set_param(LsmModel::kBasis, static_cast<double>(basis_index_)));
    }
    // n_basis / n_paths は配列の添字ではないが、core の上限と揃えるために AlwaysClamp + 後クランプ。
    ImGui::SliderInt("n basis", &n_basis_, static_cast<int>(LsmModel::kMinBasis),
                     static_cast<int>(LsmModel::kMaxBasis), "%d", ImGuiSliderFlags_AlwaysClamp);
    n_basis_ = std::clamp(n_basis_, static_cast<int>(LsmModel::kMinBasis),
                          static_cast<int>(LsmModel::kMaxBasis));
    if (ImGui::IsItemDeactivatedAfterEdit())
        runner.send(Command::set_param(LsmModel::kNBasis, static_cast<double>(n_basis_)));

    ImGui::SliderInt("n paths", &n_paths_, static_cast<int>(LsmModel::kMinPaths),
                     static_cast<int>(LsmModel::kMaxPaths), "%d",
                     ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_Logarithmic);
    n_paths_ = std::clamp(n_paths_, static_cast<int>(LsmModel::kMinPaths),
                          static_cast<int>(LsmModel::kMaxPaths));
    if (ImGui::IsItemDeactivatedAfterEdit())
        runner.send(Command::set_param(LsmModel::kNPaths, static_cast<double>(n_paths_)));
    ImGui::TextDisabled("antithetic pairs;  1 Step = 1 of %u exercise dates",
                        static_cast<unsigned>(LsmSceneSnapshot::kSteps));

    // このパネルに履歴は無いので Reset で捨てるものも無い（Snapshot が絵を丸ごと持っている）。
    draw_clock_controls(clock_, runner, [] {});

    ImGui::SeparatorText("Telemetry");
    draw_runner_telemetry(runner, last_.seq, rate_.per_second());
    if (received_ == 0) {
        ImGui::TextDisabled("%s", kWaitingSnapshot);
        ImGui::End();
        return;
    }

    ImGui::Text("t index        %u / %u   (remaining %u)", last_.t_index,
                static_cast<unsigned>(LsmSceneSnapshot::kSteps), last_.remaining);
    ImGui::Text("t              %.4f y", last_.t_now);
    if (last_.remaining > 0)
        ImGui::Text("price          %.4f +/- %.4f  (partial)", last_.price, last_.std_error);
    else
        ImGui::Text("price          %.4f +/- %.4f  (LSM American)", last_.price, last_.std_error);
    ImGui::Text("European (MC)  %.4f", last_.european);
    ImGui::Text("early exercise %u / %u paths", last_.exercised_paths, last_.n_paths);
    ImGui::Text("ITM now        %u", last_.itm_now);
    if (last_.fit_valid) {
        ImGui::Text("fit            t index %u, %u ITM paths", last_.fit_t_index, last_.itm_paths);
        if (last_.fit_fallback)  // 文字数は Control の幅に収める（はみ出すと右端で切れる）
            ImGui::TextColored(kWarnColor, "fit rank       %u / %u  (truncated: singular)",
                               last_.fit_rank, last_.n_basis);
        else
            ImGui::Text("fit rank       %u / %u", last_.fit_rank, last_.n_basis);
        for (std::uint32_t j = 0; j < last_.n_basis && j < LsmSceneSnapshot::kMaxBasis; ++j)
            ImGui::Text("  c%u           %+.6f", j, last_.coeffs[j]);
    } else {
        ImGui::TextDisabled("fit            none yet");
    }
    ImGui::Text("fps            %.1f", static_cast<double>(ImGui::GetIO().Framerate));

    ImGui::End();
}

// ---------------------------------------------------------------- シーン生成
std::unique_ptr<Scene> make_lsm_scene() {
    // Snapshot が ≈ 12 KB なのでリングは 64 枚（≈ 780 KB）。RunnerScene ごと make_unique で
    // ヒープに置かれることが前提（`scene_registry.hpp`）。
    constexpr std::size_t kSnapCap = LsmPanel::kSnapshotCapacity;

    scenes::LsmModel::Config model_cfg;  // K=100, T=1, r=5%, σ=20%, ATM put, 4000 パス, Laguerre 3

    bridge::RunnerConfig run_cfg;
    run_cfg.dt                     = 1.0 / 252.0;  // モデルは使わない（刻みは行使時点が決める）
    run_cfg.clock.steps_per_second = 8.0;          // 1 ステップ = 1 時点。64 時点を 8 秒で見る速さ
    run_cfg.clock.speed            = 1.0;
    run_cfg.publish_every          = 1;
    run_cfg.measure_every          = 1;  // 1 step が µs 級なので毎ステップ計測（R11: 既定 16 は軽い step 向け）

    return std::make_unique<RunnerScene<scenes::LsmModel, LsmPanel, kSnapCap>>(
        scenes::LsmModel{model_cfg}, run_cfg, std::in_place, model_cfg, run_cfg.clock.speed);
}

}  // namespace quantviz::viz
