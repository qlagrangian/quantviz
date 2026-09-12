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
// * R12（M4）: 一時停止中の StepOnce（SimClock::request_step() 由来の pending ステップ）で進んだ tick は、
//   その tick の**最後のステップ**を publish_every / surface_every の位相に関わらず publish する。手動
//   ステップは 1 回ずつ見せるための教材操作なので、間引きの位相のせいで画面が最大 publish_every−1
//   ステップぶん遅れる（LOB シーンは publish_every = 4）のを防ぐ。位相にも当たっているステップを
//   二重に出すことはない。走行中（非 pause）の間引きは一切変えない。
// * Model が SurfaceModel を満たすときだけ「面チャネル」が生える（M2）。グリッド大の状態は履歴が
//   不要なので、リングではなく TripleBuffer で最新 1 枚だけを渡す。満たさない Model では
//   detail::SurfaceChannel が空の基底クラスになり、Runner のサイズも振る舞いも一切変わらない。
// * M5: step の所要時間を StepHistogram（対数ビン・atomic relaxed）に記録する。計算スレッドが
//   書き、描画スレッドが読む「だいたいの姿」。RunnerConfig::measure_every = 0 で完全に切れる。

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
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

// --------------------------------------------------------------------------- M5: step 時間の計測

/// StepHistogram のビン数。
inline constexpr std::size_t kStepHistogramBins = 32;

/// StepHistogram を描画スレッドが読み取った複製（atomic を含まないただの POD）。
struct StepHistogramSample {
    std::array<std::uint64_t, kStepHistogramBins> bins{};
    std::uint64_t                                 count    = 0;  ///< サンプル数（ステップ数ではない）
    std::uint64_t                                 total_ns = 0;  ///< サンプル値の総和
    std::uint64_t                                 max_ns   = 0;  ///< サンプル値の最大

    /// 1 ステップあたりの平均 ns（サンプルが無ければ 0）。
    [[nodiscard]] double mean_ns() const noexcept {
        return count == 0 ? 0.0 : static_cast<double>(total_ns) / static_cast<double>(count);
    }
};

/// step 所要時間（ns/step）の対数ビン・ヒストグラム。
///
/// ビン境界は 100 ns から 1 s までを 32 ビン、すなわち 1 ビンあたり 10^(7/32) ≈ 1.655 倍
/// （span 10^7 を 32 等分）。`bin_lower_ns(i) = round(100 × 10^(7i/32))` を表で持ち、`bin_of` は
/// その表の二分探索なので両者は厳密に逆写像になる（浮動小数の丸めで食い違わない）。
///   * 100 ns 未満は全部ビン 0（アンダーフロー。ビン 0 の「名目」下端は 100 ns）
///   * 最上位ビン 31 は [604 ms, ∞) で上端が無い（1 s も 1000 s もここ）。`bin_lower_ns(32)` は
///     「上限なし」を表す UINT64_MAX を返す
///
/// スレッド: 計算スレッドだけが書き（`add`）、描画スレッドが読む（`snapshot`）。全ての atomic は
/// **relaxed** で、4 つのカウンタは互いに同期しない。描画側が欲しいのは「だいたいの姿」であって
/// 一貫したスナップショットではないので、走行中に読むと bins の合計と count が数個ずれることが
/// ある（停止後に読めば厳密に一致する）。ロックも割り当ても例外も無い。
///
/// サイズ: 32×8 + 3×8 = 280 B。`alignas(kCacheLineSize)` で自分のキャッシュライン群に載せ、
/// 描画スレッドの読みがリングの head/tail と false sharing しないようにしている（実サイズは
/// 64 の倍数に切り上がって 320 B）。
struct alignas(kCacheLineSize) StepHistogram {
    std::array<std::atomic<std::uint64_t>, kStepHistogramBins> bins{};
    std::atomic<std::uint64_t>                                 count{0};
    std::atomic<std::uint64_t>                                 total_ns{0};
    std::atomic<std::uint64_t>                                 max_ns{0};

    /// ビン i の下端 ns。i ≥ 32 は「上限なし」= UINT64_MAX。
    /// constexpr: パネルが軸ラベル（"100 ns" "1 us" …）をコンパイル時に組めるように。
    [[nodiscard]] static constexpr std::uint64_t bin_lower_ns(std::size_t i) noexcept {
        return i < kStepHistogramBins ? kLowerNs[i] : std::numeric_limits<std::uint64_t>::max();
    }

    /// ns が入るビン番号（0..31）。100 ns 未満は 0、上端超えは 31 に飽和する。
    [[nodiscard]] static std::size_t bin_of(std::uint64_t ns) noexcept {
        std::size_t lo = 0;                   // kLowerNs[lo] <= ns（ns < kLowerNs[0] なら lo = 0 のまま）
        std::size_t hi = kStepHistogramBins;  // ns < kLowerNs[hi]（番兵: 上端は無限大）
        while (hi - lo > 1) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (ns >= kLowerNs[mid]) lo = mid;
            else                     hi = mid;
        }
        return lo;
    }

    /// 1 サンプル（ns/step）を加える。計算スレッド専用、relaxed。
    void add(std::uint64_t ns) noexcept {
        bins[bin_of(ns)].fetch_add(1, std::memory_order_relaxed);
        count.fetch_add(1, std::memory_order_relaxed);
        total_ns.fetch_add(ns, std::memory_order_relaxed);
        // max は CAS で入れ替える。load してから store する書き方だと、その隙に描画スレッドが
        // reset_step_histogram() を呼んだとき「リセット前の古い最大値」が蘇ってしまう
        // （壊れるのは 1 サンプルではなくリセットそのもの）。CAS なら書き込むのは常に今の
        // サンプル ns であり、リセット後の値と比べて大きいときしか入らない。
        std::uint64_t cur = max_ns.load(std::memory_order_relaxed);
        while (ns > cur &&
               !max_ns.compare_exchange_weak(cur, ns, std::memory_order_relaxed,
                                             std::memory_order_relaxed)) {
            // compare_exchange_weak が cur を最新値で更新するので、条件を見直して再試行する
        }
    }

    /// 描画スレッド用の読み出し。relaxed なので「だいたいの姿」（上のスレッド注記を参照）。
    [[nodiscard]] StepHistogramSample snapshot() const noexcept {
        StepHistogramSample out;
        for (std::size_t i = 0; i < kStepHistogramBins; ++i)
            out.bins[i] = bins[i].load(std::memory_order_relaxed);
        out.count    = count.load(std::memory_order_relaxed);
        out.total_ns = total_ns.load(std::memory_order_relaxed);
        out.max_ns   = max_ns.load(std::memory_order_relaxed);
        return out;
    }

    /// 全カウンタを 0 に戻す（走行中に呼んでも安全だが、進行中の add とは同期しない）。
    void reset() noexcept {
        for (auto& b : bins) b.store(0, std::memory_order_relaxed);
        count.store(0, std::memory_order_relaxed);
        total_ns.store(0, std::memory_order_relaxed);
        max_ns.store(0, std::memory_order_relaxed);
    }

private:
    /// round(100 × 10^(7i/32))、i = 0..31。100 ns（= 1 ステップの下限の目安）から 1 s まで。
    static constexpr std::array<std::uint64_t, kStepHistogramBins> kLowerNs = {
        100,      165,      274,       453,       750,       1241,      2054,      3398,
        5623,     9306,     15399,     25483,     42170,     69783,     115478,    191095,
        316228,   523299,   865964,    1433013,   2371374,   3924190,   6493816,   10746078,
        17782794, 29427272, 48696753,  80584219,  133352143, 220673407, 365174127, 604296390};
};

// ホットパス（step ごとの加算）なので、mutex に落ちる実装は許さない。
static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
              "StepHistogram: 64-bit atomics must be lock-free (the step hot path takes no mutex)");

/// Runner の設定。テンプレート引数（リング容量）に依存しないよう、クラス外に置く。
struct RunnerConfig {
    double                    dt            = 1.0 / 252.0;  ///< 1 sim ステップの時間（年）
    SimClock::Config          clock{};
    std::size_t               publish_every = 1;            ///< k ステップに 1 回 Snapshot を出す
    std::size_t               surface_every = 1;            ///< k ステップに 1 回 面を出す（SurfaceModel）
    std::chrono::microseconds idle_sleep{200};              ///< 進めるものが無いときの休止
    /// k ステップを 1 サンプル（区間/k = ns/step）にまとめて計測する。**0 なら計測しない**
    /// （steady_clock を 1 回も読まない = 計測導入前と同じコード）。
    ///
    /// **16 が既定**（軽い step では clock 読みのバイアスが ≈ 25 ns/k 乗るため）。**1 step が µs 級の
    /// シーン（FDM / LSM / HJB）は 1 に**すると 1 ステップずつの分布が見える。
    /// StreamingModel（真の step ≈ 20 ns）の 1e6 ステップ実測（GCC 15 -O3, WSL2）:
    ///   k = 0  → 19.7〜20.2 ns/step（計測無しの HEAD と ±0.1 ns = 完全に同じ）
    ///   k = 1  → 80〜82 ns/step（ループ 4.1 倍）、ヒストグラム平均 46〜48 ns ← clock 読みの下駄
    ///   k = 16 → 23.6〜24.4 ns/step（ループ +20 %）、ヒストグラム平均 21.0〜21.6 ns ← ほぼ真値
    ///
    /// 測る区間は k = 1 なら `model_.step(dt)` ちょうど 1 回（`publish()` を含まない）、k > 1 なら
    /// 途中の k−1 回の publish も入る。パネルの軸は "ns/step (incl. publish when k>1)" と読む。
    std::size_t               measure_every = 16;
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

    /// step 時間のヒストグラム（M5）。走行中でも読んでよい（relaxed = だいたいの姿）。
    /// 単位は ns/step。`measure_every = 1` の区間は `Model::step` ちょうど 1 回で、k > 1 では
    /// 途中の k−1 回の publish も含む（パネルの軸は "ns/step (incl. publish when k>1)"）。
    const StepHistogram& step_histogram() const noexcept { return step_hist_; }
    /// 計測だけを 0 に戻す。**Model の Reset（Command::reset）では消さない**: ヒストグラムは
    /// モデルの状態ではなくテレメトリなので、パスを引き直しても計測は積み上げ続ける。
    /// パネルが「計測をやり直す」ボタンを持つならここを呼ぶ（描画スレッドから呼んでよい）。
    void reset_step_histogram() noexcept { step_hist_.reset(); }

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
        // R12: この tick のステップが一時停止中の StepOnce（pending）由来なら、最後の 1 歩は
        // 間引きの位相に関わらず publish する。走行中は last_pending が混ざりうるので対象外。
        const bool manual = clock_.paused() && clock_.last_pending() > 0;
        // 計測の有無はテンプレート引数で分ける。measure_every = 0 の側には steady_clock の
        // コードが 1 命令も残らない（計測導入前と同じコード生成になる）。
        if (cfg_.measure_every == 0) run_steps<false>(n, manual);
        else                         run_steps<true>(n, manual);
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
    /// n ステップ実行し、publish の位相（R12 の manual を含む）を捌く。
    ///
    /// Measure = true のときだけ `measure_every` ステップごとに steady_clock を 2 回読み、
    /// 1 サンプル（区間 / 実ステップ数 = ns/step）を StepHistogram に入れる。
    ///   * 測る区間は「グループ先頭の step の直前 〜 グループ末尾の step の直後」。k = 1 では
    ///     `model_.step(dt)` ちょうど 1 回（publish を含まない）。k > 1 では途中の k−1 回の publish も
    ///     区間に入る（k ステップぶんのコアループ所要時間を k で割った値になる）。
    ///   * グループは tick を跨がない（跨ぐと tick 間の待ち時間まで step に計上されるため）。
    ///     端数（m < k ステップ）は m で割って 1 サンプルにする。clock 読みのコスト（20〜25 ns）は
    ///     k ではなく **その端数自身の m** でしか薄まらないので、バイアスは ≈ 25/m ns になる。
    ///     実際の viewer では 1 tick が n ≈ steps_per_second / fps ステップ（Streaming 500/s → n ≈ 8、
    ///     LOB 1000/s → n ≈ 16）なので、k = 16 のサンプルはほとんどが端数グループで、バイアスは
    ///     ≈ 25/n ns 程度に落ち着く（k を n より大きくしても、それ以上は薄まらない）。
    template <bool Measure>
    void run_steps(std::size_t n, bool manual) {
        using clock = std::chrono::steady_clock;
        // Measure = false の実体化では触られない（MSVC /W4 の C4189 よけ）
        [[maybe_unused]] const std::size_t k     = Measure ? cfg_.measure_every : 0;
        [[maybe_unused]] std::size_t       group = 0;  // 現在のグループで走ったステップ数
        [[maybe_unused]] clock::time_point started{};
        for (std::size_t i = 0; i < n; ++i) {
            if constexpr (Measure) {
                if (group == 0) started = clock::now();
            }
            model_.step(cfg_.dt);
            if constexpr (Measure) {
                if (++group == k) {
                    record_group(started, clock::now(), group);
                    group = 0;
                }
            }
            const std::uint64_t s      = steps_.fetch_add(1, std::memory_order_relaxed) + 1;
            const bool          forced = manual && (i + 1 == n);
            if (s % cfg_.publish_every == 0 || forced) publish();
            if constexpr (SurfaceModel<M>) {
                if (s % cfg_.surface_every == 0 || forced) publish_surface();
            }
        }
        if constexpr (Measure) {
            if (group > 0) record_group(started, clock::now(), group);
        }
    }

    /// [t0, t1] を steps 本で割って 1 サンプル（ns/step）記録する。時計が巻き戻っても 0 に倒す。
    void record_group(std::chrono::steady_clock::time_point t0, std::chrono::steady_clock::time_point t1,
                      std::size_t steps) noexcept {
        const auto          d  = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        const std::uint64_t ns = d > 0 ? static_cast<std::uint64_t>(d) : 0;
        step_hist_.add(ns / static_cast<std::uint64_t>(steps));
    }

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

    StepHistogram step_hist_{};  ///< 自前のキャッシュライン（alignas）。計算が書き、描画が読む

    std::jthread thread_;  // 最後に宣言 → 最初に破棄（join）される
};

}  // namespace quantviz::bridge
