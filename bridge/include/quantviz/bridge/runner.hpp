#pragma once
// bridge/runner.hpp — Model を計算スレッドで回し、Snapshot を吐き、Command を受ける
//
//   描画スレッド ──send(Command)──▶ [commands_ ring] ──▶ Runner::dispatch ──▶ SimClock / Model::apply
//   描画スレッド ◀──poll(Snapshot)── [snapshots_ ring] ◀── Runner::publish ◀── Model::snapshot()
//   描画スレッド ◀─poll_surface()─── [surface_ triple buffer] ◀─ Runner::publish_surface ◀─ Model::surface()
//
// * Model と SimClock は計算スレッドが所有する。start() 後に外から触ってはならない。
// * tick(elapsed) は 1 ループ分を同期実行する。テストと単一スレッド運用のために公開している。
// * リングが満杯なら Snapshot は捨てて dropped_ を数える（コアを止めない）。
// * R10（M3）: ステップが 0 の tick でモデル系 Command（SetParam / Reset）を適用したら、その場で
//   Snapshot を 1 枚（SurfaceModel なら面も）publish する。一時停止中の操作が「次の Step まで
//   画面に出ない」のを防ぐため。seq は同じ値で再送されうる（SetParam）し、Reset では 0 に戻る。
//   描画側は「seq が**厳密に**減った or 0」を巻き戻しとして扱うこと（同じ seq の再送は巻き戻しではない）。
// * Model が SurfaceModel を満たすときだけ「面チャネル」が生える（M2）。グリッド大の状態は履歴が
//   不要なので、リングではなく TripleBuffer で最新 1 枚だけを渡す。満たさない Model では
//   detail::SurfaceChannel が空の基底クラスになり、Runner のサイズも振る舞いも一切変わらない。

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
#include "quantviz/bridge/triple_buffer.hpp"

namespace quantviz::bridge {

namespace detail {

/// 面チャネル。Runner の基底として使う。
/// 非 SurfaceModel では空クラス → 空基底最適化（EBO）で Runner のサイズも API も一切変わらない。
/// MSVC が EBO を適用するのは「空の基底が 1 つだけ」の場合に限られる。将来 2 本目の空チャネル基底を
/// 足すなら Runner に __declspec(empty_bases) が要る（RUNNER-09 の EboProbe がコンパイル時に捕まえる）。
/// SurfaceModel のときだけ TripleBuffer と描画スレッド側 API（poll_surface / surfaces_published）が生える。
/// メンバ関数のシグネチャを基底側に置くのは、Runner 側に `requires` 付きで書くと
/// 非 SurfaceModel のクラス実体化時に `typename M::Surface` の置換が hard error になるため。
template <class M, bool HasSurface = SurfaceModel<M>>
struct SurfaceChannel {};

template <class M>
struct SurfaceChannel<M, true> {
    using Surface = typename M::Surface;

    /// 描画スレッド専用。未読の面があれば out に取り込み true（常に最新 1 枚。古い面は捨てられる）。
    bool          poll_surface(Surface& out) noexcept { return surface_.read(out); }
    std::uint64_t surfaces_published() const noexcept {
        return surfaces_published_.load(std::memory_order_relaxed);
    }

protected:
    TripleBuffer<Surface>      surface_{};              ///< 最新 1 枚（計算 → 描画）
    std::atomic<std::uint64_t> surfaces_published_{0};  ///< publish した面の総数
};

}  // namespace detail

/// Runner の設定。テンプレート引数（リング容量）に依存しないよう、クラス外に置く。
struct RunnerConfig {
    double                    dt            = 1.0 / 252.0;  ///< 1 sim ステップの時間（年）
    SimClock::Config          clock{};
    std::size_t               publish_every = 1;            ///< k ステップに 1 回 Snapshot を出す
    std::size_t               surface_every = 1;            ///< k ステップに 1 回 面を出す（SurfaceModel）
    std::chrono::microseconds idle_sleep{200};              ///< 進めるものが無いときの休止
};

template <Model M, std::size_t SnapshotCapacity = 4096, std::size_t CommandCapacity = 256>
class Runner : public detail::SurfaceChannel<M> {
public:
    using Snapshot = typename M::Snapshot;
    using Config   = RunnerConfig;

    static constexpr std::size_t snapshot_capacity() noexcept { return SnapshotCapacity; }
    static constexpr std::size_t command_capacity() noexcept { return CommandCapacity; }

    Runner(M model, Config cfg) : model_(std::move(model)), cfg_(cfg), clock_(cfg.clock) {
        if (cfg_.publish_every == 0) cfg_.publish_every = 1;
        if (cfg_.surface_every == 0) cfg_.surface_every = 1;
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

    // 面チャネル（SurfaceModel のときだけ）: poll_surface / surfaces_published は
    // detail::SurfaceChannel<M> から継承する。

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
        const std::size_t model_commands = drain_commands();
        const std::size_t n              = clock_.due_steps(elapsed_wall_seconds);
        for (std::size_t i = 0; i < n; ++i) {
            model_.step(cfg_.dt);
            const std::uint64_t s = steps_.fetch_add(1, std::memory_order_relaxed) + 1;
            if (s % cfg_.publish_every == 0) publish();
            if constexpr (SurfaceModel<M>) {
                if (s % cfg_.surface_every == 0) publish_surface();
            }
        }
        // R10: ステップが 1 つも走らなかった tick でモデルが変わったなら、その場で 1 枚出す。
        // 出さないと一時停止中の Reset / SetParam が次の Step まで画面に現れない。ステップが
        // 走った tick では上のループが既に出しているので二重には出さない。面も同時に出す
        // （publish_every / surface_every の間引きは「流れ続ける列」への間引きなので、この
        // 単発の 1 枚には掛けない。掛けると位相次第で変更が反映されないことがある）。
        if (n == 0 && model_commands > 0) {
            publish();
            if constexpr (SurfaceModel<M>) publish_surface();
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

    /// 面は「最新 1 枚」なので、リングと違い落とす数は数えない（古い面は黙って捨てられる）。
    void publish_surface() noexcept {
        if constexpr (SurfaceModel<M>) {
            model_.surface(this->surface_.back());
            this->surface_.publish();
            this->surfaces_published_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    /// 溜まっている Command を全て適用し、そのうち**モデル系**（SetParam / Reset）の数を返す。
    /// 時計系（Pause/Resume/StepOnce/SetSpeed）はモデルの状態を変えないので数えない。
    std::size_t drain_commands() {
        std::size_t model_commands = 0;
        Command     c;
        while (commands_.try_pop(c)) {
            if (!is_clock_command(c.type)) ++model_commands;
            dispatch(c);
        }
        return model_commands;
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
        // 契約: stop() 前に send() が true を返した Command は必ず適用される。ここは停止後なので
        // 読み手はもういない → R10 の再 publish はしない（返り値は捨てる）。
        drain_commands();
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
