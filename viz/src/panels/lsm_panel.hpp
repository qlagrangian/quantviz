#pragma once
// viz/panels/lsm_panel.hpp — シーン「LSM American」の描画パネル群
//
// 責務: Runner から `LsmSceneSnapshot` を poll し、3 つのウィンドウに描く。Model には触らない
// （Snapshot の純関数 + Command の生成だけ）。
//
//   1. "Paths"                 — 16 本のパス + 5–95 / 25–75 % の分位帯 + 中央値、現在の時点の縦線、
//                                早期行使したパスはその時点に点を打つ
//   2. "Continuation fit @ t"  — 継続価値のフィット c(S)、本源的価値 (K − S)+、両者の交点（= 行使境界）
//   3. "Control"               — K / σ / r・基底・回帰子の数・パス数・時計・テレメトリ
//
// 【散布図を描かない理由】教科書の図は「ITM パスの (S_i, y_i) の散布 + フィット曲線」だが、
// 散布点は N = 最大 20000 個あり Snapshot（固定長 POD、≤ 16 KiB）には載らない。載せるには
// 「ITM パスを何本かに間引く」必要があり、間引き方が絵の意味を変えてしまう（回帰は全 ITM パスで
// 行われている）。そこで**回帰の結果そのもの**、つまりフィット曲線と本源的価値、そしてその交点
// （h(S) > c(S) ⇔ 行使、の境目）を描く。「どこから左で行使するか」という LSM の答えはこれで読める。
//
// 【履歴を持たない】このパネルに `viz::History` は無い。Snapshot が「パス・分位帯・フィット」を
// 丸ごと持っているので、描画側で時系列を積む必要が無い。したがって他シーンの巻き戻しガード
// （`seq < prev_seq_` → `clear_history()`）に相当する処理も**要らない**: 捨てるべき履歴が無く、
// 派生量（時間軸・本源的価値・行使マーカー・交点）は Snapshot が届いたフレームに毎回まるごと
// 張り直すので、Reset でもパラメータ変更でも古い値は 1 フレームも残らない。
//
// 【一時停止中の操作】Reset / パラメータ変更は掃引を満期へ巻き戻す。bridge の R10 により、
// ステップが走らない tick でも Snapshot が 1 枚 publish されるので、一時停止中でも次のフレームで
// 新しい状態に入れ替わる。待機表示が出るのは「まだ 1 枚も受け取っていない」起動直後だけ。

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "quantviz/bridge/runner.hpp"
#include "quantviz/scenes/lsm_model.hpp"
#include "quantviz/viz/clock_controls.hpp"
#include "quantviz/viz/rate_meter.hpp"
#include "quantviz/viz/scene_registry.hpp"

namespace quantviz::viz {

class LsmPanel {
public:
    /// SnapCap は make_lsm_scene() の RunnerScene と一致させること（Snapshot が ≈ 12 KB）。
    static constexpr std::size_t kSnapshotCapacity = 64;
    using Runner = bridge::Runner<scenes::LsmModel, kSnapshotCapacity>;

    LsmPanel(const scenes::LsmModel::Config& initial, double initial_speed);

    /// 毎フレーム 1 回。poll → 描画 → Command 送信。
    void draw(Runner& runner);

private:
    using Snap = scenes::LsmSceneSnapshot;

    void ingest(Runner& runner);
    void rebuild_curves();  ///< 時間軸・本源的価値・行使マーカー・交点を今の Snapshot から張り直す
    void draw_paths();
    void draw_fit();
    void draw_controls(Runner& runner);

    Snap          last_{};
    std::uint64_t received_ = 0;  ///< 受信 Snapshot の累計（`RateMeter` 用なので単調増加）
    RateMeter     rate_;

    std::array<double, Snap::kPts> times_{};      ///< パス・分位帯の時間軸（0 … T、年）
    std::array<double, Snap::kFit> intrinsic_{};  ///< fit_s 上の本源的価値 (K − S)+ / (S − K)+
    std::array<double, Snap::kShow> ex_x_{};      ///< 早期行使マーカー（時刻）
    std::array<double, Snap::kShow> ex_y_{};      ///< 〃（S）
    int                             ex_count_ = 0;
    /// フィットと本源的価値の交点（行使境界）。交わらなければ NaN。
    double                          crossing_s_ = 0.0;
    double                          crossing_v_ = 0.0;

    // UI 状態（ImGui のスライダーは float / int）
    ClockControlState clock_{};
    float             strike_;
    float             sigma_;
    float             rate_slider_;
    int               basis_index_;  ///< 0 = Power, 1 = Laguerre（core::LsmBasis の並び）
    int               n_basis_;
    int               n_paths_;
    bool              show_paths_ = true;
};

/// main.cpp / SceneRegistry 用のファクトリ。
std::unique_ptr<Scene> make_lsm_scene();

}  // namespace quantviz::viz
