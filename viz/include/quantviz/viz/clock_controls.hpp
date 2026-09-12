#pragma once
// viz/clock_controls.hpp — 共通「時計 Control」の状態と Command 生成（vizcore: GUI 非依存）
//
// 責務: どのシーンでも同じ speed / pause / step / reset の UI 状態を 1 つの POD にまとめ、
// 「状態遷移 → 送るべき Command」を純関数として提供する。ImGui に触れないので単体テストできる
// （VIZ-03）。実際のウィジェット描画は viz 層の `panels/clock_panel.hpp` が担当する。

#include "quantviz/bridge/command.hpp"

namespace quantviz::viz {

/// 共通 Control の UI 状態。ImGui のスライダーは float なので speed も float で持つ。
struct ClockControlState {
    float speed  = 1.0f;    ///< 倍率。UI は 0.01〜100 の対数スライダー
    bool  paused = false;   ///< 一時停止中か（Runner/SimClock の状態のミラー）
};

/// paused を反転し、対応する Pause / Resume コマンドを返す。
inline bridge::Command toggle_pause(ClockControlState& s) noexcept {
    s.paused = !s.paused;
    return s.paused ? bridge::Command::pause() : bridge::Command::resume();
}

/// speed を更新し、SetSpeed コマンドを返す（UI は float、Command は double）。
inline bridge::Command set_speed(ClockControlState& s, float v) noexcept {
    s.speed = v;
    return bridge::Command::set_speed(static_cast<double>(v));
}

/// 1 ステップだけ進める。状態を持たない。
inline bridge::Command step_once() noexcept { return bridge::Command::step_once(); }

/// 現在の seed で巻き戻す（seed 0 = 現在の seed）。状態を持たない。
inline bridge::Command reset() noexcept { return bridge::Command::reset(); }

/// Step ボタンを押せるのは一時停止中だけ。
inline bool step_allowed(const ClockControlState& s) noexcept { return s.paused; }

}  // namespace quantviz::viz
