#include "panels/lob_panel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <span>

#include <imgui.h>
#include <implot.h>

#include "panels/clock_panel.hpp"
#include "panels/panel_common.hpp"

namespace quantviz::viz {

namespace {

using scenes::LobModel;
using scenes::LobSnapshot;

constexpr ImVec4 kBidColor{0.25f, 0.72f, 0.45f, 0.85f};   ///< 買い（bid / 買いテイカー）
constexpr ImVec4 kAskColor{0.90f, 0.36f, 0.34f, 0.85f};   ///< 売り（ask / 売りテイカー）
constexpr ImVec4 kMidColor{0.85f, 0.85f, 0.30f, 0.70f};   ///< mid の参照線

/// ヒートマップの色域の下限（数量）。板が薄いときに 1 枚の注文で色が振り切れないように。
constexpr float kHeatMinScale = 60.0f;

/// 取消比率スライダーの下限。0 にすると指値だけが溜まってプールが枯れる（lob_model.hpp）。
constexpr float kMinCancelFrac = 0.02f;

/// 約定マーカーの大きさ（数量 1〜100 → 3〜9 px）。
float marker_size(std::uint64_t qty) {
    const double q = std::min(static_cast<double>(qty), static_cast<double>(LobModel::kMaxOrderQty));
    return 3.0f + 6.0f * static_cast<float>(std::sqrt(q / static_cast<double>(LobModel::kMaxOrderQty)));
}

/// ティック → 価格。
double price_of(core::micro::Price ticks, double tick_size) {
    return static_cast<double>(ticks) * tick_size;
}

}  // namespace

LobPanel::LobPanel(const scenes::LobModel::Config& initial, double initial_speed)
    : clock_{static_cast<float>(initial_speed), false},
      mu_(static_cast<float>(initial.hawkes.mu)),
      alpha_(static_cast<float>(initial.hawkes.alpha)),
      beta_(static_cast<float>(initial.hawkes.beta)),
      market_frac_(static_cast<float>(initial.market_frac)),
      cancel_frac_(static_cast<float>(initial.cancel_frac)) {
    // 最初の Snapshot が届く前のフレームでも Control が意味のある値を出せるように種を入れておく。
    last_.tick_size   = initial.tick_size;
    last_.mu          = initial.hawkes.mu;
    last_.alpha       = initial.hawkes.alpha;
    last_.beta        = initial.hawkes.beta;
    last_.market_frac = initial.market_frac;
    last_.cancel_frac = initial.cancel_frac;
}

void LobPanel::draw(Runner& runner) {
    ingest(runner);
    draw_ladder();
    draw_heatmap();
    draw_trades();
    draw_controls(runner);
}

// ---------------------------------------------------------------- Snapshot → History
void LobPanel::ingest(Runner& runner) {
    LobSnapshot s;
    while (runner.poll(s)) {
        // seq が**厳密に**減ったら巻き戻り（Reset）。同じ seq の再送（R10: 一時停止中の SetParam）は
        // 巻き戻しではないので History は消さず、telemetry だけ更新する。
        if (s.seq < prev_seq_) clear_history();
        const bool republish = (received_ > 0 && s.seq == prev_seq_);
        last_                = s;
        prev_seq_            = s.seq;
        ++received_;
        if (s.seq == 0 || republish) continue;  // 未ステップの点と再送は履歴に積まない
        lambda_buy_.push(s.t, s.lambda_buy);
        lambda_sell_.push(s.t, s.lambda_sell);
        if (heat_skip_ == 0) {
            push_heat_column(s);
            heat_skip_ = static_cast<std::size_t>(heat_stride_) - 1;
        } else {
            --heat_skip_;
        }
    }
    rate_.sample(received_, now_seconds());
}

/// その Snapshot の mid を中心とした ±32 ティックの数量を 1 列積む。
/// 符号は色の選択だけに使う: bid = +qty、ask = −qty。ImPlot の RdBu は「低い値 = 赤」なので、
/// これで mid の上（ask）が赤、下（bid）が青になる。行 0 は ImPlot の上端 = mid + 31 ティック。
void LobPanel::push_heat_column(const LobSnapshot& s) {
    column_.fill(0.0f);
    for (std::uint32_t i = 0; i < s.n_asks; ++i) {
        const auto off = s.asks[i].price - s.mid_ticks;
        if (off < -kHalfRows || off >= kHalfRows) continue;
        column_[static_cast<std::size_t>(kHalfRows - 1 - off)] = -static_cast<float>(s.asks[i].qty);
    }
    for (std::uint32_t i = 0; i < s.n_bids; ++i) {
        const auto off = s.bids[i].price - s.mid_ticks;
        if (off < -kHalfRows || off >= kHalfRows) continue;
        column_[static_cast<std::size_t>(kHalfRows - 1 - off)] = static_cast<float>(s.bids[i].qty);
    }
    heat_.push_column(column_);
}

// ---------------------------------------------------------------- 深度ラダー
void LobPanel::draw_ladder() {
    ImGui::SetNextWindowSize(ImVec2(370, 350), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, kPanelTop), ImGuiCond_FirstUseEver);
    ImGui::Begin("Depth ladder");

    const double ts = last_.tick_size > 0.0 ? last_.tick_size : 0.01;
    for (std::uint32_t i = 0; i < last_.n_bids; ++i) {
        bid_qty_[i]   = -static_cast<double>(last_.bids[i].qty);
        bid_price_[i] = price_of(last_.bids[i].price, ts);
    }
    for (std::uint32_t i = 0; i < last_.n_asks; ++i) {
        ask_qty_[i]   = static_cast<double>(last_.asks[i].qty);
        ask_price_[i] = price_of(last_.asks[i].price, ts);
    }

    if (last_.n_bids > 0)
        ImGui::Text("best bid %.2f x %llu", bid_price_[0],
                    static_cast<unsigned long long>(last_.bids[0].qty));
    else
        ImGui::TextDisabled("best bid   --");
    ImGui::SameLine(170.0f);
    if (last_.n_asks > 0)
        ImGui::Text("best ask %.2f x %llu", ask_price_[0],
                    static_cast<unsigned long long>(last_.asks[0].qty));
    else
        ImGui::TextDisabled("best ask   --");

    if (ImPlot::BeginPlot("##ladder", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("qty  (bid = negative)", "price", ImPlotAxisFlags_AutoFit,
                          ImPlotAxisFlags_AutoFit);
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);
        const double bar = ts * 0.8;  // バーの厚み = 0.8 ティック
        if (last_.n_bids > 0) {
            ImPlot::SetNextFillStyle(kBidColor);
            ImPlot::PlotBars("bids", bid_qty_.data(), bid_price_.data(),
                             static_cast<int>(last_.n_bids), bar, ImPlotBarsFlags_Horizontal);
        }
        if (last_.n_asks > 0) {
            ImPlot::SetNextFillStyle(kAskColor);
            ImPlot::PlotBars("asks", ask_qty_.data(), ask_price_.data(),
                             static_cast<int>(last_.n_asks), bar, ImPlotBarsFlags_Horizontal);
        }
        if (last_.seq > 0 && last_.mid > 0.0) {
            ImPlot::SetNextLineStyle(kMidColor, 1.5f);
            ImPlot::PlotInfLines("mid", &last_.mid, 1, ImPlotInfLinesFlags_Horizontal);
        }
        // best の 2 本にだけ値札を付ける（板の一番外側は読めればよい）。
        if (last_.n_bids > 0)
            ImPlot::Annotation(bid_qty_[0], bid_price_[0], kBidColor, ImVec2(-6, 0), true, "%.2f",
                               bid_price_[0]);
        if (last_.n_asks > 0)
            ImPlot::Annotation(ask_qty_[0], ask_price_[0], kAskColor, ImVec2(6, 0), true, "%.2f",
                               ask_price_[0]);
        ImPlot::EndPlot();
    }
    ImGui::End();
}

// ---------------------------------------------------------------- 価格 × 時間ヒートマップ
void LobPanel::draw_heatmap() {
    ImGui::SetNextWindowSize(ImVec2(380, 350), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(390, kPanelTop), ImGuiCond_FirstUseEver);
    ImGui::Begin("Price x time heatmap");

    const std::size_t cols = heat_.count();
    if (cols == 0) {
        ImGui::TextDisabled("waiting for the first snapshot...");
        ImGui::End();
        return;
    }

    // 色域は表示中の最大数量から（薄い板で 1 枚の注文に振り切れないよう下限を置く）。
    const std::span<const float> values = heat_.ordered();
    float                        scale  = kHeatMinScale;
    for (const float v : values) scale = std::max(scale, std::abs(v));

    ImGui::TextDisabled("mid-relative bins (+/-%d ticks), ask red / bid blue", kHalfRows);
    ImPlot::PushColormap(ImPlotColormap_RdBu);
    ImPlot::ColormapScale("qty", -static_cast<double>(scale), static_cast<double>(scale),
                          ImVec2(60, -1));
    ImGui::SameLine();
    if (ImPlot::BeginPlot("##depthheat", ImVec2(-1, -1))) {
        constexpr ImPlotAxisFlags kAxis = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines;
        ImPlot::SetupAxes("columns (oldest -> newest)", "ticks from mid", kAxis, kAxis);
        ImPlot::SetupAxesLimits(0.0, static_cast<double>(cols), -kHalfRows, kHalfRows,
                                ImPlotCond_Always);
        ImPlot::PlotHeatmap("depth", values.data(), static_cast<int>(kHeatRows),
                            static_cast<int>(cols), -static_cast<double>(scale),
                            static_cast<double>(scale), nullptr, ImPlotPoint(0, -kHalfRows),
                            ImPlotPoint(static_cast<double>(cols), kHalfRows));
        ImPlot::EndPlot();
    }
    ImPlot::PopColormap();
    ImGui::End();
}

// ---------------------------------------------------------------- 直近約定 + λ
void LobPanel::draw_trades() {
    ImGui::SetNextWindowSize(ImVec2(760, 340), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(10, 390), ImGuiCond_FirstUseEver);
    ImGui::Begin("Trades / intensity");

    const float half = (ImGui::GetContentRegionAvail().y - 6.0f) * 0.5f;
    const double ts  = last_.tick_size > 0.0 ? last_.tick_size : 0.01;

    if (ImPlot::BeginPlot("##trades", ImVec2(-1, half))) {
        ImPlot::SetupAxes("trade seq", "price", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupLegend(ImPlotLocation_NorthWest);
        // 1 点ずつ描く（マーカーの大きさを数量に比例させるため。"##" 始まりは凡例に出ない）。
        for (std::uint32_t i = 0; i < last_.n_trades; ++i) {
            const auto&  f = last_.trades[i];
            const double x = static_cast<double>(f.seq);
            const double y = price_of(f.price, ts);
            const ImVec4 c = (f.taker_side == core::micro::Side::Bid) ? kBidColor : kAskColor;
            char         id[16];
            std::snprintf(id, sizeof(id), "##tr%u", i);
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, marker_size(f.qty), c, 1.0f, c);
            ImPlot::PlotScatter(id, &x, &y, 1);
        }
        if (last_.n_trades == 0) ImPlot::Annotation(0.5, 0.5, kMidColor, ImVec2(0, 0), true, "no trades yet");
        ImPlot::EndPlot();
    }

    if (ImPlot::BeginPlot("##intensity", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("t (s)", "lambda (events/s)", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
        setup_follow_axis(lambda_buy_, follow_, window_seconds_);
        if (lambda_buy_.count() > 1) {
            ImPlot::SetNextLineStyle(kBidColor, 1.5f);
            ImPlot::PlotLine("lambda buy", lambda_buy_.xs(), lambda_buy_.ys(), lambda_buy_.count(), 0,
                             lambda_buy_.offset());
        }
        if (lambda_sell_.count() > 1) {
            ImPlot::SetNextLineStyle(kAskColor, 1.5f);
            ImPlot::PlotLine("lambda sell", lambda_sell_.xs(), lambda_sell_.ys(), lambda_sell_.count(),
                             0, lambda_sell_.offset());
        }
        if (last_.mu > 0.0) {
            ImPlot::SetNextLineStyle(kMidColor, 1.0f);
            ImPlot::PlotInfLines("mu", &last_.mu, 1, ImPlotInfLinesFlags_Horizontal);
        }
        ImPlot::EndPlot();
    }
    ImGui::End();
}

// ---------------------------------------------------------------- Controls → Command
void LobPanel::draw_controls(Runner& runner) {
    using bridge::Command;

    ImGui::SetNextWindowSize(ImVec2(420, 722), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(780, kPanelTop), ImGuiCond_FirstUseEver);
    ImGui::Begin("Control");

    ImGui::SeparatorText("Order flow (one parameter set for both sides)");
    // ImGui のスライダーは float、Command は double（UI → Command 規約: 明示的に広げる）
    if (ImGui::SliderFloat("mu (events/s)", &mu_, 1.0f, 2000.0f, "%.0f", ImGuiSliderFlags_Logarithmic))
        runner.send(Command::set_param(LobModel::kMu, static_cast<double>(mu_)));
    if (ImGui::SliderFloat("alpha", &alpha_, 0.0f, 1000.0f, "%.0f"))
        runner.send(Command::set_param(LobModel::kAlpha, static_cast<double>(alpha_)));
    if (ImGui::SliderFloat("beta (1/s)", &beta_, 1.0f, 2000.0f, "%.0f", ImGuiSliderFlags_Logarithmic))
        runner.send(Command::set_param(LobModel::kBeta, static_cast<double>(beta_)));
    // モデルは分岐比 α/β < 0.95 を保つために α を縮めることがあるので、効いている値を出す。
    const double branching = last_.beta > 0.0 ? last_.alpha / last_.beta : 0.0;
    ImGui::TextDisabled("in effect %.0f/%.0f/%.0f, branching %.2f -> %.0f /s/side", last_.mu,
                        last_.alpha, last_.beta, branching,
                        branching < 1.0 ? last_.mu / (1.0 - branching) : 0.0);

    if (ImGui::SliderFloat("market frac", &market_frac_, 0.0f, static_cast<float>(LobModel::kMaxFrac),
                           "%.2f", ImGuiSliderFlags_AlwaysClamp))
        runner.send(Command::set_param(LobModel::kMarketFrac, static_cast<double>(market_frac_)));
    // 下限は 0 ではなく kMinCancelFrac: 取消 0 では板が単調に厚くなってプールが枯れる。
    if (ImGui::SliderFloat("cancel frac", &cancel_frac_, kMinCancelFrac,
                           static_cast<float>(LobModel::kMaxFrac), "%.2f", ImGuiSliderFlags_AlwaysClamp))
        runner.send(Command::set_param(LobModel::kCancelFrac, static_cast<double>(cancel_frac_)));
    // 取消確率は板の厚みに比例する（lob_model.hpp）。効いている規則を 1 行で出す。
    ImGui::TextDisabled("cancel p = frac x orders / %.0f; the rest are limit orders",
                        LobModel::kCancelReference);

    ImGui::SeparatorText("Large order injection");
    ImGui::SliderFloat("inject qty", &inject_qty_, 50.0f, 5000.0f, "%.0f",
                       ImGuiSliderFlags_Logarithmic);
    const double q = static_cast<double>(inject_qty_);
    char         label[32];
    std::snprintf(label, sizeof(label), "Inject buy %.0f", q);
    if (ImGui::Button(label, ImVec2(190, 0))) runner.send(Command::set_param(LobModel::kInjectBuy, q));
    ImGui::SameLine();
    std::snprintf(label, sizeof(label), "Inject sell %.0f", q);
    if (ImGui::Button(label, ImVec2(190, 0))) runner.send(Command::set_param(LobModel::kInjectSell, q));
    ImGui::TextDisabled("queued: buy %llu  sell %llu (applied at the next step)",
                        static_cast<unsigned long long>(last_.pending_buy),
                        static_cast<unsigned long long>(last_.pending_sell));

    draw_clock_controls(clock_, runner, [this] { clear_history(); });

    ImGui::SeparatorText("View");
    ImGui::Checkbox("follow latest", &follow_);
    ImGui::SliderFloat("plot window (s)", &window_seconds_, 0.5f, 16.0f, "%.1f");
    // 配列添字ではないが、stride はカウンタに使うので必ず 1 以上に留める（CLAUDE.md の規約）。
    if (ImGui::SliderInt("heatmap stride", &heat_stride_, 1, 16, "%d", ImGuiSliderFlags_AlwaysClamp))
        heat_skip_ = 0;
    heat_stride_ = std::clamp(heat_stride_, 1, 16);
    // 一時停止中・受信前は秒換算が意味を持たない（レート計が 0 に向かう）ので列数だけ出す。
    const double rate = rate_.per_second();
    if (received_ > 0 && !clock_.paused && rate > 0.0)
        ImGui::TextDisabled("%zu columns = %.1f s of book history", heat_.count(),
                            static_cast<double>(heat_.count() *
                                                static_cast<std::size_t>(heat_stride_)) /
                                rate);
    else
        ImGui::TextDisabled("%zu columns of book history", heat_.count());

    ImGui::SeparatorText("Telemetry");
    // 共通の行（seq / snapshots per sec / ring / steps / frame）は clock_panel.hpp、この下はシーン固有。
    draw_runner_telemetry(runner, last_.seq, rate_.per_second());
    ImGui::Text("t              %.2f s", last_.t);
    ImGui::Text("mid / spread   %.2f / %.2f", last_.mid, last_.spread);
    ImGui::Text("orders in book %u  (bid qty %llu, ask qty %llu)", last_.orders_in_book,
                static_cast<unsigned long long>(last_.book_qty_bid),
                static_cast<unsigned long long>(last_.book_qty_ask));
    ImGui::Text("pool free      %u / %zu", last_.pool_free, LobModel::kMaxOrders);
    // clamped = 値幅の壁（板の価格レベル配列の端）に丸めて置いた指値。0 でなくなったら Reset の合図。
    ImGui::Text("sub/cxl/rej/cl %llu / %llu / %llu / %llu",
                static_cast<unsigned long long>(last_.orders_submitted),
                static_cast<unsigned long long>(last_.orders_cancelled),
                static_cast<unsigned long long>(last_.orders_rejected),
                static_cast<unsigned long long>(last_.orders_clamped));
    ImGui::Text("fills          %llu  (truncated %llu, discarded qty %llu)",
                static_cast<unsigned long long>(last_.fills),
                static_cast<unsigned long long>(last_.orders_truncated),
                static_cast<unsigned long long>(last_.discarded_qty));

    ImGui::End();
}

// ---------------------------------------------------------------- シーン生成（main.cpp / SceneRegistry 用）
std::unique_ptr<Scene> make_lob_scene() {
    scenes::LobModel::Config model_cfg;
    model_cfg.seed = 20240912;

    bridge::RunnerConfig run_cfg;
    run_cfg.dt                     = 0.001;   // 1 ステップ = 1 ms
    run_cfg.clock.steps_per_second = 1000.0;  // speed 1x = 実時間
    run_cfg.clock.speed            = 1.0;
    run_cfg.publish_every          = 4;  // 250 Snapshot/s（リングは 256 枚 × ~2.5 KB）

    return std::make_unique<RunnerScene<scenes::LobModel, LobPanel, LobPanel::kSnapshotCapacity>>(
        scenes::LobModel{model_cfg}, run_cfg, std::in_place, model_cfg, run_cfg.clock.speed);
}

void LobPanel::clear_history() {
    lambda_buy_.clear();
    lambda_sell_.clear();
    heat_.clear();
    heat_skip_ = 0;
    // 板の内容は捨てる（次の Snapshot まで描かない）。パラメータ（mu/alpha/beta/比率）と
    // tick_size は残す: 全体をゼロにすると Reset 直後の 1 フレームだけ Control が 0 を表示する。
    last_.seq              = 0;
    last_.t                = 0.0;
    last_.n_bids           = 0;
    last_.n_asks           = 0;
    last_.n_trades         = 0;
    last_.spread           = 0.0;
    last_.orders_in_book   = 0;
    last_.orders_submitted = 0;
    last_.orders_cancelled = 0;
    last_.orders_rejected  = 0;
    last_.orders_clamped   = 0;
    last_.fills            = 0;
    last_.orders_truncated = 0;
    last_.discarded_qty    = 0;
    last_.book_qty_bid     = 0;
    last_.book_qty_ask     = 0;
    last_.pending_buy      = 0;
    last_.pending_sell     = 0;
    last_.pool_free        = 0;
}

}  // namespace quantviz::viz
