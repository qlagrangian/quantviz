#pragma once
// viz/panels/panel_common.hpp — 全パネル共通の小物（viz 層: ImGui/ImPlot 依存。ヘッダオンリー）
//
// 責務: 4 つのパネルが同じ形で書き写していた「表示用の壁時計」「年 → 取引日の換算」「メニューバー
// 下の初期位置」「最新点に追従する X 軸」を 1 か所に集めるだけ。判断ロジックは置かない
// （数値は core、GUI 非依存の純ロジックは vizcore の `rate_meter.hpp` / `clock_controls.hpp`）。

#include <chrono>
#include <cstddef>

#include <implot.h>

#include "quantviz/viz/history.hpp"

namespace quantviz::viz {

/// 表示用の壁時計（秒）。レート計に渡すだけなので単調増加すれば十分（steady_clock）。
inline double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

/// 1 年 = 252 取引日。Snapshot の t（年）を「取引日」の時間軸に直すときの換算。
inline constexpr double kTradingDays = 252.0;

/// パネルの初期位置 y（`ImGuiCond_FirstUseEver`）。メニューバー（高さ約 22 px）の下に置く。
inline constexpr float kPanelTop = 32.0f;

/// follow が有効なら X 軸を「最新点から window_days ぶん過去まで」に毎フレーム固定する。
/// 履歴が空のときは何もしない（範囲が [−window, 0] に飛んで初回の描画が跳ねるのを避ける）。
template <std::size_t N>
void setup_follow_axis(const History<N>& x_source, bool follow, float window_days) {
    if (!follow || x_source.empty()) return;
    const double x1 = x_source.latest_x();
    ImPlot::SetupAxisLimits(ImAxis_X1, x1 - static_cast<double>(window_days), x1, ImPlotCond_Always);
}

}  // namespace quantviz::viz
