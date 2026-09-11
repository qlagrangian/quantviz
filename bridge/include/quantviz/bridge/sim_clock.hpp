#pragma once
// bridge/sim_clock.hpp — シミュレーション時刻と壁時計の変換（純ロジック、時間源を持たない）
//
//   due_steps(elapsed_wall) = floor( accumulated( elapsed_wall * speed * steps_per_second ) ) + pending
//
//   * 一時停止中は壁時間を蓄積しない → 再開時のバースト無し
//   * request_step() は一時停止中でも 1 ステップだけ許す（デバッガの "step" と同じ）
//   * max_steps_per_tick で spiral of death を防ぐ。超過分は捨てる（追いつこうとしない）
//   * 時間源を外から渡すので単体テスト可能

#include <cmath>
#include <cstddef>

namespace quantviz::bridge {

class SimClock {
public:
    struct Config {
        double      steps_per_second   = 1000.0;  ///< speed = 1 のとき、壁 1 秒あたりの sim ステップ数
        double      speed              = 1.0;     ///< 倍率（0 で停止相当）
        std::size_t max_steps_per_tick = 10000;   ///< 1 回の due_steps が返す上限
        bool        start_paused       = false;
    };

    SimClock() noexcept : SimClock(Config{}) {}
    explicit SimClock(Config cfg) noexcept : cfg_(cfg), paused_(cfg.start_paused) { set_speed(cfg.speed); }

    /// 壁時計で elapsed 秒が経過したとき、実行すべき sim ステップ数を返す。
    std::size_t due_steps(double elapsed_wall_seconds) noexcept {
        if (!(elapsed_wall_seconds > 0.0)) elapsed_wall_seconds = 0.0;  // 負・NaN は 0 扱い

        std::size_t steps = pending_;
        pending_          = 0;

        if (!paused_) {
            accumulator_ += elapsed_wall_seconds * cfg_.speed * cfg_.steps_per_second;
            const double whole = std::floor(accumulator_);
            accumulator_ -= whole;
            steps += static_cast<std::size_t>(whole);
        }
        if (steps > cfg_.max_steps_per_tick) {
            steps        = cfg_.max_steps_per_tick;
            accumulator_ = 0.0;  // 超過分は捨てる
        }
        return steps;
    }

    void pause() noexcept { paused_ = true; }
    void resume() noexcept { paused_ = false; }
    bool paused() const noexcept { return paused_; }

    void request_step() noexcept { ++pending_; }
    std::size_t pending_steps() const noexcept { return pending_; }

    /// 負・NaN は 0 にクランプ。
    void set_speed(double speed) noexcept { cfg_.speed = (speed > 0.0) ? speed : 0.0; }
    double speed() const noexcept { return cfg_.speed; }

    const Config& config() const noexcept { return cfg_; }

private:
    Config      cfg_;
    bool        paused_      = false;
    double      accumulator_ = 0.0;
    std::size_t pending_     = 0;
};

}  // namespace quantviz::bridge
