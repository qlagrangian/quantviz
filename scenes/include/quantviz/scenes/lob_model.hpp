#pragma once
// scenes/lob_model.hpp — シーン 7「Order book」: 2 本の Hawkes 流で駆動する指値板
//
// 買い側と売り側に独立な Hawkes 過程（λ(t) = μ + Σ α e^{−β(t−tᵢ)}）を置き、1 ステップ = dt 秒の
// 窓 [t, t+dt) に届いた事象を 1 つずつ `core::micro::MatchingEngine` へ流す。各事象は
//   成行（確率 market_frac）/ 取消（板の厚みに比例、下記）/ 指値（残り）
// のどれかになる。描画が要るもの（上位 16 レベル × 2 側、直近 32 約定、λ、カウンタ）だけを
// 縮約した ~2.5 KB の POD Snapshot に載せる（面チャネルは使わない）。
//
// 時間は **秒**（viewer は dt = 1 ms、steps_per_second = 1000 で実時間）。Hawkes のパラメータも
// 「毎秒何件」で読む（既定 μ = 200 /s, α = 100, β = 200 → 分岐比 0.5、定常 400 /s/側）。
//
// 設計上の選択（根拠）:
// * **窓ごとの thinning**。`hawkes_simulate` は t = 0 から空の履歴で始めるので、ステップを跨いで
//   励起を持ち越せない。ここでは `HawkesIntensity` を側ごとに状態として持ち、Ogata の thinning を
//   窓 [t, t+dt) に対してだけ回す（候補 1 件につき一様乱数 2 個、事象数に比例、確保なし）。
// * **取消は板の厚みに比例**（Cont–Stoikov 型の θ·depth）: 1 事象あたりの取消確率を
//   cancel_frac × 生存注文数 / kCancelReference とする。固定確率にすると指値の流入が取消と約定を
//   上回って板が単調に厚くなり（既定の比率なら毎秒 +200 件超）、数十秒でプールが枯れる。
//   比例させると板の厚みが平衡（既定で 1 側あたり ≈ kCancelReference × 2 件）に落ち着く。
//   指値が枯れないよう、取消確率は kMaxFrac − market_frac で頭を打つ（= 指値は常に 10 % 以上）。
// * **live-id 配列は側ごとにプール容量ぶん**持ち、swap-remove で消す。約定で消えた注文は通知が
//   来ないので配列には「死んだ id」が混ざる: 取消の抽選で引いたら `cancel` が false を返すので
//   その場で落とし（kCancelAttempts 回まで引き直す）、満杯になったら板に残っている id だけを
//   残して詰め直す（`book().find` は O(1)。満杯時のみ O(kMaxOrders) で、既定の流れでは数十秒に 1 回）。
// * **Reset は engine.reset() と live-id 配列の破棄を同じ操作で行う**。id はプールのスロット添字を
//   含み clear() を跨ぐと一意でない（order_book.hpp）ので、持ち越すと再利用されたスロットの
//   別注文を取り消してしまう。
// * **μ / α / β のどれを変えても励起履歴は捨てる**（λ は μ から積み直す）。Σ α e^{−βΔ} は α, β が
//   変わると意味が変わるため 3 つで揃える。時定数 1/β（既定 5 ms）で元の水準へ戻る。
// * **mid は両側が居るときだけ動かす**。片側が空（大口注入の直後）なら直前の mid を保ち、
//   spread は 0（未定義）にする。ヒートマップの中心ティックも同じ規則に従う。
// * **価格には壁がある**（clamp_price）。板は [base, base + kMaxLevels) のティックしか持てず、
//   mid は無平均回帰のランダムウォークなので、長く回せばいつかは端に着く。端に着いた指値は
//   丸めて置かれ（拒否ではない）、そのぶん板が端に積み上がる。壁に当たった件数は
//   `orders_clamped` で見える。kMaxLevels = 4096（mid ± 2048 ティック = ±20.48 価格単位）なら
//   実測の mid の拡散（2000 秒で標準偏差 ≈ 190 ティック）に対して初到達時間は 1024 のときの
//   16 倍（≈ 距離²）になる。壁に着いたら Reset で敷き直す。
//
// ステップ予算（Release, 既定パラメータ, dt = 1 ms）: 1 ステップあたり平均 0.8 事象で ≈ 1 µs。
// Debug では `engine_.set_check_every(0)` で 1 操作ごとの検査を止め、step() の最後に
// `check_invariants(false)`（O(kMaxLevels + 生存注文数)）を 1 回だけ assert する。
//
// ホットパス（step / snapshot）はアロケーション・例外・mutex なし。engine のヒープ確保は
// コンストラクタだけ（プール 8192 × 40 B + レベル 2 × 4096 × 24 B ≈ 0.5 MB）。

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/core/micro/matching_engine.hpp"
#include "quantviz/core/micro/order_book.hpp"
#include "quantviz/core/models/hawkes.hpp"
#include "quantviz/core/rng.hpp"

namespace quantviz::scenes {

/// 板の「今」を縮約した POD。上位 16 レベル × 2 側 + 直近 32 約定 + λ + カウンタ。
struct LobSnapshot {
    static constexpr std::size_t kLevels = 16;  ///< 片側あたりの深度ラダーのレベル数
    static constexpr std::size_t kTrades = 32;  ///< 直近約定の件数（新しい順）

    double t         = 0.0;  ///< sim 時刻（秒）
    double tick_size = 0.0;  ///< 1 ティックの価格
    double mid       = 0.0;  ///< (best bid + best ask) / 2（片側が空なら直前の値）
    double spread    = 0.0;  ///< best ask − best bid（片側が空なら 0 = 未定義）

    double lambda_buy  = 0.0;  ///< 買い流の Hawkes 強度 λ(t⁺)（件/秒）
    double lambda_sell = 0.0;  ///< 売り流の Hawkes 強度
    double mu          = 0.0;  ///< 実効 Hawkes パラメータ（クランプ後。参照線用）
    double alpha       = 0.0;
    double beta        = 0.0;
    double market_frac = 0.0;  ///< 実効の成行比率
    double cancel_frac = 0.0;  ///< 実効の取消比率（kCancelReference 件の板に対する値）

    core::micro::Price mid_ticks = 0;  ///< mid のティック（ヒートマップの中心）

    std::uint64_t orders_submitted = 0;  ///< 受理した注文数（engine の統計）
    std::uint64_t orders_cancelled = 0;
    std::uint64_t orders_rejected  = 0;  ///< 価格範囲外 / プール枯渇 / 数量 0
    std::uint64_t orders_clamped   = 0;  ///< 価格レベル配列の端に丸めて置いた指値（= 値幅の壁に当たった）
    std::uint64_t fills            = 0;  ///< 約定の総件数
    std::uint64_t orders_truncated = 0;  ///< fills バッファ満杯で残りを捨てた注文数
    std::uint64_t discarded_qty    = 0;  ///< 板が尽きて捨てた数量（両側の合計）
    std::uint64_t book_qty_bid     = 0;  ///< 板の合計数量
    std::uint64_t book_qty_ask     = 0;
    std::uint64_t pending_buy      = 0;  ///< 次のステップの先頭で入る大口注入（数量）
    std::uint64_t pending_sell     = 0;
    std::uint64_t seq              = 0;  ///< ステップ通番（0 = 未ステップ／Reset 直後）

    std::array<core::micro::LevelView, kLevels> bids{};    ///< best から降順（空は qty 0）
    std::array<core::micro::LevelView, kLevels> asks{};    ///< best から昇順
    std::array<core::micro::Fill, kTrades>      trades{};  ///< 直近約定（新しい順）

    std::uint32_t n_bids         = 0;  ///< bids の有効レベル数
    std::uint32_t n_asks         = 0;
    std::uint32_t n_trades       = 0;  ///< trades の有効件数
    std::uint32_t orders_in_book = 0;  ///< 板に載っている注文数（両側）
    std::uint32_t pool_free      = 0;  ///< 注文プールの空きスロット数
};
static_assert(std::is_trivially_copyable_v<LobSnapshot>);
static_assert(std::is_standard_layout_v<LobSnapshot>);
// 上位 16 レベル × 2 側（24 B）+ 直近 32 約定（48 B）= 2304 B + スカラー ≈ 2.5 KB
static_assert(sizeof(LobSnapshot) <= 4096);

class LobModel {
public:
    using Snapshot  = LobSnapshot;
    using Price     = core::micro::Price;
    using Qty       = core::micro::Qty;
    using OrderId   = core::micro::OrderId;
    using Side      = core::micro::Side;
    using Fill      = core::micro::Fill;
    using LevelView = core::micro::LevelView;

    /// 注文プールと価格レベル。mid は配列の中央 kMidLevel に置く（両側に ±2048 ティックの余地）。
    static constexpr std::size_t kMaxOrders = 8192;
    static constexpr std::size_t kMaxLevels = 4096;
    static constexpr Price       kMidLevel  = static_cast<Price>(kMaxLevels / 2);
    using Engine = core::micro::MatchingEngine<kMaxOrders, kMaxLevels>;

    enum Param : std::uint32_t {
        kMu         = 1,  ///< 両側共通の基底強度 μ（件/秒）
        kAlpha      = 2,  ///< 跳ね幅 α
        kBeta       = 3,  ///< 減衰率 β（時定数 1/β 秒）
        kMarketFrac = 4,  ///< 事象が成行になる確率
        kCancelFrac = 5,  ///< kCancelReference 件の板に対する取消確率
        kInjectBuy  = 6,  ///< 大口買い注入（value = 数量。次ステップの先頭で成行）
        kInjectSell = 7,  ///< 大口売り注入
    };

    // ---- 初期板 ----
    static constexpr std::uint32_t kInitialLevels = 10;   ///< 片側に敷くレベル数（1 レベル 1 注文）
    static constexpr Qty           kInitialQtyMin = 50;   ///< 初期数量の下限
    static constexpr Qty           kInitialQtyMax = 150;  ///< 初期数量の上限

    // ---- 流れの規則 ----
    static constexpr Qty    kMaxOrderQty      = 100;   ///< 指値・成行の数量は 1〜100 の対数一様
    static constexpr double kOffsetP          = 0.35;  ///< 指値の距離（幾何分布）。平均 ≈ 1.9 ティック
    static constexpr Price  kMaxOffset        = 64;    ///< 距離の上限（ティック）
    static constexpr double kCancelReference  = 25.0;  ///< 取消確率が cancel_frac になる片側の注文数
    static constexpr int    kCancelAttempts   = 4;     ///< 死んだ id を引いたときの引き直し回数
    static constexpr std::size_t kFillBuffer  = 256;   ///< 1 ステップぶんの約定バッファ
    static constexpr std::size_t kMaxArrivals = 1024;  ///< 1 ステップ・1 側あたりの候補時刻の上限

    // ---- UI 入力のクランプ幅 ----
    static constexpr double kMinMu        = 1e-6;
    static constexpr double kMaxMu        = 1e4;   ///< 1 ms の窓に 10 件（kMaxArrivals の内側）
    static constexpr double kMaxAlpha     = 1e6;
    static constexpr double kMinBeta      = 1e-6;
    static constexpr double kMaxBeta      = 1e6;
    static constexpr double kMaxBranching = 0.95;  ///< α/β ≤ 0.95 に保つ（越えたら α を縮める）
    static constexpr double kMaxFrac      = 0.9;   ///< 成行・取消の比率の上限（指値を 10 % 残す）
    static constexpr double kMaxInject    = 1e9;   ///< 1 回の注入数量の上限

    struct Config {
        double             tick_size  = 0.01;                  ///< 1 ティックの価格
        double             base_price = 100.00;                ///< 初期 mid
        core::HawkesParams hawkes{200.0, 100.0, 200.0};        ///< 両側共通（件/秒）
        double             market_frac = 0.1;
        double             cancel_frac = 0.2;
        std::uint64_t      seed        = 42;
    };

    // GCC の既定引数バグ回避のため委譲コンストラクタにする（CLAUDE.md）
    LobModel() : LobModel(Config{}) {}

    explicit LobModel(Config cfg)
        : engine_(base_tick(cfg)),
          rng_(cfg.seed),
          seed_(cfg.seed),
          tick_size_(sane_tick(cfg.tick_size)),
          base_tick_(base_tick(cfg)),
          hawkes_(cfg.hawkes),
          intensity_{{core::HawkesIntensity(cfg.hawkes), core::HawkesIntensity(cfg.hawkes)}},
          market_frac_(clamp_ui(cfg.market_frac, 0.0, kMaxFrac)),
          cancel_frac_(clamp_ui(cfg.cancel_frac, 0.0, kMaxFrac)) {
        // Debug の 1 操作ごとの検査は止め、step() の最後に 1 回だけ自分で呼ぶ（matching_engine.hpp）
        engine_.set_check_every(0);
        sanitize_hawkes();
        reset(seed_);
    }

    /// 1 ステップ = dt 秒の窓に届いた全事象。注入 → 買い流 → 売り流の順に処理する。
    void step(double dt) {
        const double t0 = t_;
        const double t1 = t0 + (dt > 0.0 ? dt : 0.0);
        n_fills_        = 0;

        // 1. 大口注入（前フレームのボタン）はステップの先頭で成行として入れる
        for (const Side side : {Side::Bid, Side::Ask}) {
            Qty& pending = pending_[idx(side)];
            if (pending != 0) {
                engine_.submit_market(side, pending, fills_, n_fills_);
                pending = 0;
            }
        }

        // 2. 両側の Hawkes 流（買い → 売りの順。1 ms の窓では順序の影響は無視できる）
        generate(Side::Bid, t0, t1);
        generate(Side::Ask, t0, t1);

        // 3. 直近約定リング（注入ぶんも含めて 1 ステップぶんをまとめて）
        for (std::size_t k = 0; k < n_fills_; ++k) push_recent(fills_[k]);

        t_ = t1;
        update_mid();
        ++seq_;
        assert(engine_.book().check_invariants(false));
    }

    Snapshot snapshot() const noexcept {
        Snapshot s{};
        s.t           = t_;
        s.tick_size   = tick_size_;
        s.mid         = mid_price_;
        s.spread      = spread_;
        s.lambda_buy  = intensity_[idx(Side::Bid)].at(t_);
        s.lambda_sell = intensity_[idx(Side::Ask)].at(t_);
        s.mu          = hawkes_.mu;
        s.alpha       = hawkes_.alpha;
        s.beta        = hawkes_.beta;
        s.market_frac = market_frac_;
        s.cancel_frac = cancel_frac_;
        s.mid_ticks   = mid_tick_;

        const auto& book = engine_.book();
        s.n_bids         = static_cast<std::uint32_t>(book.depth(Side::Bid, s.bids));
        s.n_asks         = static_cast<std::uint32_t>(book.depth(Side::Ask, s.asks));

        s.n_trades = static_cast<std::uint32_t>(recent_count_);
        for (std::size_t k = 0; k < recent_count_; ++k)  // 新しい順
            s.trades[k] = recent_[(recent_head_ + Snapshot::kTrades - 1 - k) % Snapshot::kTrades];

        s.orders_submitted = engine_.orders_submitted();
        s.orders_cancelled = engine_.orders_cancelled();
        s.orders_rejected  = engine_.orders_rejected();
        s.orders_clamped   = orders_clamped_;
        s.fills            = engine_.fills();
        s.orders_truncated = engine_.orders_truncated();
        s.discarded_qty    = engine_.discarded_qty(Side::Bid) + engine_.discarded_qty(Side::Ask);
        s.book_qty_bid     = book.total_qty(Side::Bid);
        s.book_qty_ask     = book.total_qty(Side::Ask);
        s.pending_buy      = pending_[idx(Side::Bid)];
        s.pending_sell     = pending_[idx(Side::Ask)];
        s.orders_in_book   = static_cast<std::uint32_t>(book.order_count());
        s.pool_free        = static_cast<std::uint32_t>(book.pool_free());
        s.seq              = seq_;
        return s;
    }

    void apply(const bridge::Command& c) {
        switch (c.type) {
            case bridge::CommandType::SetParam:
                switch (c.param_id) {
                    case kMu:
                        hawkes_.mu = clamp_ui(c.value, kMinMu, kMaxMu);
                        sanitize_hawkes();
                        break;
                    case kAlpha:
                        hawkes_.alpha = clamp_ui(c.value, 0.0, kMaxAlpha);
                        sanitize_hawkes();
                        break;
                    case kBeta:
                        hawkes_.beta = clamp_ui(c.value, kMinBeta, kMaxBeta);
                        sanitize_hawkes();
                        break;
                    case kMarketFrac: market_frac_ = clamp_ui(c.value, 0.0, kMaxFrac); break;
                    case kCancelFrac: cancel_frac_ = clamp_ui(c.value, 0.0, kMaxFrac); break;
                    case kInjectBuy:  queue_inject(Side::Bid, c.value); break;
                    case kInjectSell: queue_inject(Side::Ask, c.value); break;
                    default:          break;  // 未知の param_id は無視
                }
                break;
            case bridge::CommandType::Reset: reset(c.seed != 0 ? c.seed : seed_); break;
            default:                         break;  // 時計系は Runner が処理済み
        }
    }

    /// 板・統計・λ・live-id を捨てて初期板を敷き直す。μ/α/β と比率は保つ。
    void reset(std::uint64_t seed) {
        seed_ = seed;
        rng_.reseed(seed);
        engine_.reset();                          // id はここで 1 から振り直される（order_book.hpp）
        for (auto& live : live_) live.count = 0;  // ので live-id は同じ操作で捨てる
        for (auto& lam : intensity_) lam = core::HawkesIntensity(hawkes_);
        pending_        = {0, 0};
        orders_clamped_ = 0;
        n_fills_        = 0;
        recent_head_    = 0;
        recent_count_   = 0;
        t_              = 0.0;
        seq_            = 0;
        mid_tick_       = base_tick_ + kMidLevel;
        seed_book();
        update_mid();
    }

    // ---- テスト用の読み取りアクセサ（start() 後の Runner から呼んではならない） ----
    [[nodiscard]] const Engine& engine() const noexcept { return engine_; }
    /// 取消の抽選に使う live-id の件数（死んだ id を含みうる）。
    [[nodiscard]] std::size_t live_count(Side s) const noexcept { return live_[idx(s)].count; }

private:
    /// 取消の抽選に使う「板に出した id」の固定容量集合（swap-remove。死んだ id が混ざりうる）。
    struct LiveList {
        std::array<OrderId, kMaxOrders> ids{};
        std::size_t                     count = 0;
    };

    static constexpr std::size_t idx(Side s) noexcept { return static_cast<std::size_t>(s); }

    /// UI 値を [lo, hi] に入れる。NaN は lo に落ちる（比較が偽になるため）。
    static constexpr double clamp_ui(double v, double lo, double hi) noexcept {
        if (!(v > lo)) return lo;
        if (v > hi) return hi;
        return v;
    }

    static double sane_tick(double ts) noexcept { return clamp_ui(ts, 1e-9, 1e6); }

    /// 価格レベル配列の下端。mid（base_price / tick_size）がレベル添字 kMidLevel に来るように取る。
    static Price base_tick(const Config& cfg) noexcept {
        const double mid = cfg.base_price / sane_tick(cfg.tick_size);
        const double m   = clamp_ui(mid, 0.0, 1e15);
        return static_cast<Price>(std::llround(m)) - kMidLevel;
    }

    /// μ > 0, α ≥ 0, β > 0, α/β ≤ kMaxBranching を保ち、励起履歴を捨てて λ を μ から積み直す。
    void sanitize_hawkes() noexcept {
        hawkes_.mu    = clamp_ui(hawkes_.mu, kMinMu, kMaxMu);
        hawkes_.alpha = clamp_ui(hawkes_.alpha, 0.0, kMaxAlpha);
        hawkes_.beta  = clamp_ui(hawkes_.beta, kMinBeta, kMaxBeta);
        if (hawkes_.alpha > kMaxBranching * hawkes_.beta) hawkes_.alpha = kMaxBranching * hawkes_.beta;
        for (auto& lam : intensity_) lam = core::HawkesIntensity(hawkes_);
    }

    void queue_inject(Side side, double value) noexcept {
        const double q = clamp_ui(value, 0.0, kMaxInject);
        Qty&         p = pending_[idx(side)];
        p += static_cast<Qty>(q);
        if (static_cast<double>(p) > kMaxInject) p = static_cast<Qty>(kMaxInject);
    }

    /// mid の周りに kInitialLevels レベルずつ、数量 [kInitialQtyMin, kInitialQtyMax] で敷く。
    void seed_book() noexcept {
        for (Price i = 1; i <= static_cast<Price>(kInitialLevels); ++i) {
            place_initial(Side::Bid, mid_tick_ - i);
            place_initial(Side::Ask, mid_tick_ + i);
        }
    }

    void place_initial(Side side, Price price) noexcept {
        const double span = static_cast<double>(kInitialQtyMax - kInitialQtyMin + 1);
        Qty          q    = kInitialQtyMin + static_cast<Qty>(rng_.uniform() * span);
        if (q > kInitialQtyMax) q = kInitialQtyMax;
        const OrderId id = engine_.submit_limit(side, clamped_price(price), q, fills_, n_fills_);
        if (id != 0 && engine_.book().find(id)) push_live(side, id);
    }

    /// 窓 [t0, t1) の事象を Ogata の thinning で引く（事象の間は λ が非増加なので λ(t⁺) が上界）。
    void generate(Side side, double t0, double t1) {
        core::HawkesIntensity& lam = intensity_[idx(side)];
        double                 cur = t0;
        for (std::size_t guard = 0; guard < kMaxArrivals; ++guard) {
            const double bound = lam.at(cur);
            if (!(bound > 0.0)) break;  // μ > 0 なので起きないが、無限ループを作らない保険
            cur += -std::log1p(-rng_.uniform()) / bound;
            if (!(cur < t1)) break;
            if (rng_.uniform() * bound <= lam.at(cur)) {  // 採択確率 λ(cur)/λ*
                lam.add_event(cur);
                handle_arrival(side);
            }
        }
    }

    /// 1 事象: 成行 / 取消 / 指値。取消確率は板の厚みに比例させる（ファイル先頭のコメント）。
    void handle_arrival(Side side) {
        const double depth    = static_cast<double>(engine_.book().order_count(side));
        const double room     = kMaxFrac - market_frac_;  // 指値に残す取り分（≥ 0.0）
        double       p_cancel = cancel_frac_ * depth / kCancelReference;
        if (!(p_cancel < room)) p_cancel = room > 0.0 ? room : 0.0;
        const double u = rng_.uniform();
        if (u < market_frac_) {
            engine_.submit_market(side, draw_qty(), fills_, n_fills_);
        } else if (u < market_frac_ + p_cancel) {
            cancel_random(side);
        } else {
            submit_limit(side);
        }
    }

    /// 反対側の best から幾何分布ぶん離した価格に指値を出す（距離 0 = 交差 → その場で約定）。
    void submit_limit(Side side) {
        const Qty   q   = draw_qty();
        const Price off = draw_offset();
        const auto  opp = engine_.book().best(core::micro::opposite(side));
        const Price ref = opp ? *opp : (side == Side::Bid ? mid_tick_ + 1 : mid_tick_ - 1);
        const Price price   = clamped_price(side == Side::Bid ? ref - off : ref + off);
        const OrderId id    = engine_.submit_limit(side, price, q, fills_, n_fills_);
        if (id != 0 && engine_.book().find(id)) push_live(side, id);  // 板に載ったものだけ控える
    }

    /// live-id からランダムに 1 つ取り消す。死んだ id を引いたら落として引き直す（最大 kCancelAttempts）。
    void cancel_random(Side side) {
        LiveList& live = live_[idx(side)];
        for (int attempt = 0; attempt < kCancelAttempts && live.count > 0; ++attempt) {
            const std::size_t i  = pick_index(live.count);
            const OrderId     id = live.ids[i];
            live.ids[i]          = live.ids[--live.count];  // swap-remove
            if (engine_.cancel(id)) return;
        }
    }

    void push_live(Side side, OrderId id) noexcept {
        LiveList& live = live_[idx(side)];
        // 死んだ id が生存注文の 4 倍を超えたら（+8 は薄い板での往復を避ける下駄）詰め直す。
        // compact 後は count == 生存数なので、次の詰め直しまでに最低 3 倍の push が要る = 償却 O(1)。
        if (live.count == kMaxOrders || live.count > 8 + 4 * engine_.book().order_count(side))
            compact(live);
        if (live.count == kMaxOrders) return;  // プールが本当に一杯（次の指値は engine が拒否する）
        live.ids[live.count++] = id;
    }

    /// 板に残っている id だけを残して詰め直す（配列が満杯になったときだけ。O(kMaxOrders)）。
    void compact(LiveList& live) noexcept {
        std::size_t n = 0;
        for (std::size_t i = 0; i < live.count; ++i)
            if (engine_.book().find(live.ids[i])) live.ids[n++] = live.ids[i];
        live.count = n;
    }

    void push_recent(const Fill& f) noexcept {
        recent_[recent_head_] = f;
        recent_head_          = (recent_head_ + 1) % Snapshot::kTrades;
        if (recent_count_ < Snapshot::kTrades) ++recent_count_;
    }

    /// 両側に best があるときだけ mid / spread / 中心ティックを動かす。
    void update_mid() noexcept {
        const auto bid = engine_.book().best_bid();
        const auto ask = engine_.book().best_ask();
        if (bid && ask) {
            mid_tick_  = (*bid + *ask) / 2;
            mid_price_ = 0.5 * static_cast<double>(*bid + *ask) * tick_size_;
            spread_    = static_cast<double>(*ask - *bid) * tick_size_;
        } else {
            spread_ = 0.0;  // 片側が空: mid は直前の値を保つ
        }
    }

    /// 1〜kMaxOrderQty の対数一様（整数への切り捨てで上端も出るよう span は kMaxOrderQty + 1）。
    /// u ∈ [0,1) なので q ∈ [1, kMaxOrderQty+1) — 下限のクランプは要らない。上限のクランプは
    /// u が 1 に極めて近いときの丸めに対する保険（数学的には到達しない）。
    Qty draw_qty() noexcept {
        const double q = std::exp(rng_.uniform() * std::log(static_cast<double>(kMaxOrderQty) + 1.0));
        const Qty    n = static_cast<Qty>(q);
        return n > kMaxOrderQty ? kMaxOrderQty : n;
    }

    /// 反対側の best からの距離（ティック）。幾何分布 P(k) = p(1−p)^k、平均 (1−p)/p。
    Price draw_offset() noexcept {
        double g = std::log1p(-rng_.uniform()) / std::log1p(-kOffsetP);
        if (!(g < static_cast<double>(kMaxOffset))) g = static_cast<double>(kMaxOffset);  // NaN / ∞ も
        return static_cast<Price>(g);
    }

    std::size_t pick_index(std::size_t n) noexcept {
        const std::size_t i = static_cast<std::size_t>(rng_.uniform() * static_cast<double>(n));
        return i < n ? i : n - 1;
    }

    /// 値幅の壁（ファイル先頭のコメント）に丸めた価格。丸めたら orders_clamped を 1 増やす。
    Price clamped_price(Price p) noexcept {
        const Price c = clamp_price(p);
        if (c != p) ++orders_clamped_;
        return c;
    }

    /// 価格レベル配列の内側へ（端は best の更新で走査が伸びるので 1 レベルぶん余裕を取る）。
    [[nodiscard]] Price clamp_price(Price p) const noexcept {
        const Price lo = base_tick_ + 1;
        const Price hi = base_tick_ + static_cast<Price>(kMaxLevels) - 2;
        return p < lo ? lo : (p > hi ? hi : p);
    }

    Engine        engine_;
    core::Rng     rng_;
    std::uint64_t seed_;

    double tick_size_;
    Price  base_tick_;
    Price  mid_tick_   = 0;
    double mid_price_  = 0.0;
    double spread_     = 0.0;

    core::HawkesParams                   hawkes_;
    std::array<core::HawkesIntensity, 2> intensity_;
    double                               market_frac_;
    double                               cancel_frac_;

    double        t_              = 0.0;
    std::uint64_t seq_             = 0;
    std::uint64_t orders_clamped_  = 0;  ///< 値幅の壁に丸めて置いた指値の件数

    std::array<Qty, 2>                          pending_{0, 0};  ///< 大口注入の待ち行列（数量）
    std::array<Fill, kFillBuffer>               fills_{};        ///< 1 ステップぶんの約定
    std::size_t                                 n_fills_ = 0;
    std::array<Fill, LobSnapshot::kTrades>      recent_{};       ///< 直近約定リング（古い順に上書き）
    std::size_t                                 recent_head_  = 0;
    std::size_t                                 recent_count_ = 0;
    std::array<LiveList, 2>                     live_{};
};

static_assert(bridge::Model<LobModel>, "LobModel must satisfy the Model contract");

}  // namespace quantviz::scenes
