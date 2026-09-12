#pragma once
// viz/panels/lob_panel.hpp — シーン 7「Order book」の描画パネル群
//
// 責務: Runner から Snapshot を poll → 深度ラダー / 価格×時間ヒートマップ / 直近約定 + λ を描き、
// 注文フローのパラメータと「大口注入」ボタンを Command に変える。Model には一切触らない。
//
// ヒートマップの再センタリング規則: 各列は **その Snapshot の mid**（`mid_ticks`）を中心とした
// ±32 ティックの数量で作る（符号は色の選択だけに使う: mid の下の bid = +qty で青、
// 上の ask = −qty で赤。ImPlot の RdBu は「低い値 = 赤」）。ビンが mid に追従するので、mid が動いた区間は
// 「帯がずれる」のではなく「帯が中央に留まったまま、過去の列と価格が揃わない」形になる。
// 絶対価格の軸で描くには列ごとに原点が要る（M3 の範囲外）ので、ここでは mid 相対で読む
// —— 深さの厚み・板が掃かれて空く様子を見るには十分。y 軸のラベルも「mid からのティック」。

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "quantviz/bridge/runner.hpp"
#include "quantviz/scenes/lob_model.hpp"
#include "quantviz/viz/clock_controls.hpp"
#include "quantviz/viz/history.hpp"
#include "quantviz/viz/history2d.hpp"
#include "quantviz/viz/rate_meter.hpp"
#include "quantviz/viz/scene_registry.hpp"

namespace quantviz::viz {

class LobPanel {
public:
    /// Snapshot が ~2.5 KB あるのでリングは 256 枚（計画書「Snapshot のサイズ方針」）。
    static constexpr std::size_t kSnapshotCapacity = 256;
    using Runner = bridge::Runner<scenes::LobModel, kSnapshotCapacity>;

    LobPanel(const scenes::LobModel::Config& initial, double initial_speed);

    /// 毎フレーム 1 回。poll → 描画 → Command 送信。
    void draw(Runner& runner);

private:
    void ingest(Runner& runner);
    void push_heat_column(const scenes::LobSnapshot& s);
    void draw_ladder();
    void draw_heatmap();
    void draw_trades();
    void draw_controls(Runner& runner);
    void clear_history();

    static constexpr std::size_t kHistory  = 4096;  ///< λ の履歴（250 Snapshot/s で約 16 秒）
    static constexpr std::size_t kHeatRows = 64;    ///< mid ± 32 ティック
    static constexpr std::size_t kHeatCols = 256;   ///< 列数（stride 1 なら約 1 秒）
    static constexpr int         kHalfRows = 32;    ///< 中心から上下のティック数

    History<kHistory>             lambda_buy_;
    History<kHistory>             lambda_sell_;
    History2D<kHeatRows, kHeatCols> heat_;

    scenes::LobSnapshot last_{};
    std::uint64_t       prev_seq_ = 0;  ///< 巻き戻り検出用（seq が厳密に減ったら History を捨てる）
    std::uint64_t       received_ = 0;
    std::size_t         heat_skip_ = 0;  ///< stride カウンタ（0 のときだけ列を積む）
    RateMeter           rate_;           ///< 受信 Snapshot/秒（表示用）

    /// ラダー用の描画バッファ（PlotBars は double 配列しか受けない）。
    std::array<double, scenes::LobSnapshot::kLevels> bid_qty_{};
    std::array<double, scenes::LobSnapshot::kLevels> bid_price_{};
    std::array<double, scenes::LobSnapshot::kLevels> ask_qty_{};
    std::array<double, scenes::LobSnapshot::kLevels> ask_price_{};
    /// ヒートマップの 1 列（mid 相対のビンごとの数量。ask は正、bid は負）。
    std::array<float, kHeatRows> column_{};

    // UI 状態（ImGui のスライダーは float / int）
    ClockControlState clock_{};
    float             mu_;
    float             alpha_;
    float             beta_;
    float             market_frac_;
    float             cancel_frac_;
    float             inject_qty_     = 500.0f;
    int               heat_stride_    = 1;      ///< 何 Snapshot に 1 列積むか（1〜16）
    float             window_seconds_ = 4.0f;   ///< λ プロットの表示幅（秒）
    bool              follow_         = true;
};

/// main.cpp / SceneRegistry 用のファクトリ: LOB の Runner + Panel を 1 つのシーンに束ねる。
std::unique_ptr<Scene> make_lob_scene();

}  // namespace quantviz::viz
