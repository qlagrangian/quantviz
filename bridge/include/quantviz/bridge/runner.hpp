#pragma once
// bridge/runner.hpp — Model を計算スレッドで回し、Snapshot を吐き、Command を受ける
//
//   描画スレッド ──send(Command)──▶ [commands_ ring] ──▶ Runner::dispatch ──▶ SimClock / Model::apply
//   描画スレッド ◀──poll(Snapshot)── [snapshots_ ring] ◀── Runner::publish ◀── Model::snapshot()
//
// * Model と SimClock は計算スレッドが所有する。start() 後に外から触ってはならない。
// * tick(elapsed) は 1 ループ分を同期実行する。テストと単一スレッド運用のために公開している。
// * リングが満杯なら Snapshot は捨てて dropped_ を数える（コアを止めない）。

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stop_token>
#include <thread>
#include <utility>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/bridge/sim_clock.hpp"
#include "quantviz/bridge/spsc_ring.hpp"

namespace quantviz::bridge {

/// Runner の設定。テンプレート引数（リング容量）に依存しないよう、クラス外に置く。
struct RunnerConfig {
    double                    dt            = 1.0 / 252.0;  ///< 1 sim ステップの時間（年）
    SimClock::Config          clock{};
    std::size_t               publish_every = 1;            ///< k ステップに 1 回 Snapshot を出す
    std::chrono::microseconds idle_sleep{200};              ///< 進めるものが無いときの休止
};

template <Model M, std::size_t SnapshotCapacity = 4096, std::size_t CommandCapacity = 256>
class Runner {
public:
    using Snapshot = typename M::Snapshot;
    using Config   = RunnerConfig;

    static constexpr std::size_t snapshot_capacity() noexcept { return SnapshotCapacity; }
    static constexpr std::size_t command_capacity() noexcept { return CommandCapacity; }

    Runner(M model, Config cfg) : model_(std::move(model)), cfg_(cfg), clock_(cfg.clock) {
        if (cfg_.publish_every == 0) cfg_.publish_every = 1;
    }
    ~Runner() { stop(); }

    Runner(const Runner&)            = delete;
    Runner& operator=(const Runner&) = delete;

    // ------------------------------------------------------------------ 描画スレッド側 API
    bool          send(const Command& c) noexcept { return commands_.try_push(c); }
    bool          poll(Snapshot& out) noexcept { return snapshots_.try_pop(out); }
    std::uint64_t dropped_snapshots() const noexcept { return dropped_.load(std::memory_order_relaxed); }
    std::uint64_t total_steps() const noexcept { return steps_.load(std::memory_order_relaxed); }
    std::size_t   queued_snapshots() const noexcept { return snapshots_.size_approx(); }
    bool          running() const noexcept { return thread_.joinable(); }

    void start() {
        if (thread_.joinable()) return;
        thread_ = std::jthread([this](std::stop_token st) { loop(st); });
    }
    /// 停止要求 → 未消化 Command の適用 → join。以後 model() を読んでも競合しない。
    void stop() {
        if (thread_.joinable()) {
            thread_.request_stop();
            thread_.join();
        }
    }

    // ------------------------------------------------------------------ 同期 API（テスト／単一スレッド）
    /// 1 ループ分: コマンド消化 → 期限ステップ実行 → Snapshot 発行。実行したステップ数を返す。
    std::size_t tick(double elapsed_wall_seconds) {
        drain_commands();
        const std::size_t n = clock_.due_steps(elapsed_wall_seconds);
        for (std::size_t i = 0; i < n; ++i) {
            model_.step(cfg_.dt);
            const std::uint64_t s = steps_.fetch_add(1, std::memory_order_relaxed) + 1;
            if (s % cfg_.publish_every == 0) publish();
        }
        return n;
    }

    /// start() 中に呼ぶのはデータ競合。テストと単一スレッド運用のみ。
    const M&        model() const noexcept { return model_; }
    const SimClock& clock() const noexcept { return clock_; }
    const Config&   config() const noexcept { return cfg_; }

private:
    void publish() {
        if (!snapshots_.try_push(model_.snapshot())) dropped_.fetch_add(1, std::memory_order_relaxed);
    }

    void drain_commands() {
        Command c;
        while (commands_.try_pop(c)) dispatch(c);
    }

    void dispatch(const Command& c) {
        switch (c.type) {
            case CommandType::Pause:    clock_.pause(); break;
            case CommandType::Resume:   clock_.resume(); break;
            case CommandType::StepOnce: clock_.request_step(); break;
            case CommandType::SetSpeed: clock_.set_speed(c.value); break;
            case CommandType::SetParam:
            case CommandType::Reset:    model_.apply(c); break;
        }
    }

    void loop(std::stop_token st) {
        using clock = std::chrono::steady_clock;
        auto last   = clock::now();
        while (!st.stop_requested()) {
            const auto   now     = clock::now();
            const double elapsed = std::chrono::duration<double>(now - last).count();
            last                 = now;
            if (tick(elapsed) == 0) std::this_thread::sleep_for(cfg_.idle_sleep);
        }
        drain_commands();  // 契約: stop() 前に send() が true を返した Command は必ず適用される
    }

    M        model_;
    Config   cfg_;
    SimClock clock_;

    SpscRing<Snapshot, SnapshotCapacity> snapshots_;
    SpscRing<Command, CommandCapacity>   commands_;

    std::atomic<std::uint64_t> dropped_{0};
    std::atomic<std::uint64_t> steps_{0};

    std::jthread thread_;  // 最後に宣言 → 最初に破棄（join）される
};

}  // namespace quantviz::bridge
