#pragma once
// viz/rate_meter.hpp — 受信 Snapshot レートの表示用メーター（vizcore: GUI 非依存）
//
// 責務: 単調増加する累計カウンタ（例: パネルが受け取った Snapshot の総数）と、そのときの壁時計
// 秒を受け取り、「毎秒いくつ届いているか」を滑らかな値として返す。時刻源は持たない（`SimClock`
// と同じ方針で、壁時計は呼び出し側が渡す）ので、合成時刻で決定的に単体テストできる（VIZ-05）。
// 更新は 0.25 秒の窓が閉じたときだけ行い、瞬間値 Δcount/Δt を係数 0.2 の EMA に通す
// （毎フレーム更新すると 1 フレームぶんのばらつきで数字が踊り、読めなくなる）。最初の窓が
// 閉じるまでは 0 を返す。各パネルはこれを 1 つ持ち、`ingest()` の最後で `sample()` を呼ぶ。

#include <cstdint>

namespace quantviz::viz {

class RateMeter {
public:
    /// 窓長（秒）。これより短い間隔で呼ばれた分は無視する。
    static constexpr double kWindowSeconds = 0.25;
    /// EMA の重み（新しい瞬間値の取り分）。
    static constexpr double kEmaWeight = 0.2;

    /// 累計 `total` と現在時刻 `now_s`（秒）を与える。最初の呼び出しは基準を latch するだけで、
    /// 以降は窓が閉じるたびに瞬間値 Δtotal/Δt を EMA に取り込む。
    void sample(std::uint64_t total, double now_s) noexcept {
        if (last_wall_ == 0.0) {
            last_wall_  = now_s;
            last_count_ = total;
        } else if (now_s - last_wall_ >= kWindowSeconds) {
            const double inst = static_cast<double>(total - last_count_) / (now_s - last_wall_);
            ema_        = ema_ == 0.0 ? inst : (1.0 - kEmaWeight) * ema_ + kEmaWeight * inst;
            last_wall_  = now_s;
            last_count_ = total;
        }
    }

    /// 直近の推定レート（件/秒）。最初の窓が閉じるまでは 0。
    [[nodiscard]] double per_second() const noexcept { return ema_; }

    /// 0 に戻す。次の `sample()` が「最初の呼び出し」として基準を latch し直すので、
    /// Reset でカウンタが飛んでも跳ね上がらない。
    void reset() noexcept {
        ema_        = 0.0;
        last_wall_  = 0.0;
        last_count_ = 0;
    }

private:
    double        ema_        = 0.0;
    double        last_wall_  = 0.0;  ///< 直近に窓を閉じた時刻（0 = まだ latch していない）
    std::uint64_t last_count_ = 0;
};

}  // namespace quantviz::viz
