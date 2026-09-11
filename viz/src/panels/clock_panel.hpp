#pragma once
// viz/panels/clock_panel.hpp — 全シーン共通の「Simulation clock」と Telemetry 行（viz 層: ImGui 依存）
//
// 責務: `viz::ClockControlState` の純関数（`clock_controls.hpp`）に ImGui のウィジェットを被せるだけ。
// 判断ロジックはここに置かない（ここは手動チェックリストの範囲で、自動テストしない）。
// Runner はテンプレート引数で受けるのでシーンごとの Model に依存しない。ヘッダオンリー。

#include <cstdint>
#include <utility>

#include <imgui.h>

#include "quantviz/viz/clock_controls.hpp"

namespace quantviz::viz {

/// "Simulation clock" セクション: speed スライダー / Pause|Resume / Step / Reset。
/// on_reset は Reset ボタン押下時に呼ばれる（パネルが History を clear するため）。
template <class Runner, class OnReset>
void draw_clock_controls(ClockControlState& st, Runner& runner, OnReset&& on_reset) {
    ImGui::SeparatorText("Simulation clock");

    float speed = st.speed;  // 確定は set_speed()（状態更新と Command 生成を 1 か所に閉じ込める）
    if (ImGui::SliderFloat("speed (x)", &speed, 0.01f, 100.0f, "%.2fx", ImGuiSliderFlags_Logarithmic))
        runner.send(set_speed(st, speed));

    if (ImGui::Button(st.paused ? "Resume" : "Pause", ImVec2(100, 0))) runner.send(toggle_pause(st));
    ImGui::SameLine();
    ImGui::BeginDisabled(!step_allowed(st));  // Step は一時停止中だけ
    if (ImGui::Button("Step", ImVec2(100, 0))) runner.send(step_once());
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Reset", ImVec2(100, 0))) {
        runner.send(reset());
        std::forward<OnReset>(on_reset)();
    }
}

/// "Telemetry" セクションの共通行: seq, snapshots/s, queued, dropped, core steps, frame。
/// シーン固有の行（t, spot, ...）は各パネルが自分で描く。
template <class Runner>
void draw_runner_telemetry(const Runner& runner, std::uint64_t seq, double snapshots_per_sec) {
    ImGui::Text("seq            %llu", static_cast<unsigned long long>(seq));
    ImGui::Text("snapshots/s    %.0f", snapshots_per_sec);
    ImGui::Text("queued         %zu / %zu", runner.queued_snapshots(), Runner::snapshot_capacity());
    ImGui::Text("dropped        %llu", static_cast<unsigned long long>(runner.dropped_snapshots()));
    ImGui::Text("core steps     %llu", static_cast<unsigned long long>(runner.total_steps()));
    ImGui::Text("frame          %.2f ms", 1000.0 / static_cast<double>(ImGui::GetIO().Framerate));
}

}  // namespace quantviz::viz
