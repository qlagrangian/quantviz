// LOBSCENE-01..04 — scenes/lob_model.hpp の契約 / 単体 / 決定性テスト（docs/03_tdd_spec.md §6.3）
//
// このシーンは「1 step = dt 秒の窓に届いた全事象を処理する」ので、テストも窓の中の事象で書く。
// 流れそのものを止めたい場合（注入の効果だけを見たいとき）は μ を極小にした Config を使う:
// 1 ms の窓に事象が来る確率が 1e-9 未満になるので、板は敷いたまま動かない。
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/core/micro/order_book.hpp"
#include "quantviz/core/models/hawkes.hpp"
#include "quantviz/scenes/lob_model.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::bridge::Command;
using quantviz::core::HawkesParams;
using quantviz::core::micro::LevelView;
using quantviz::core::micro::OrderId;
using quantviz::core::micro::Side;
using quantviz::scenes::LobModel;
using quantviz::scenes::LobSnapshot;

namespace {

constexpr double kDt = 0.001;  // 1 ms（viewer と同じ）

/// 流れを実質止めた板（μ = 1e-6 /s → 1 ms の窓に事象が来る期待値 1e-9）。
LobModel::Config quiet_config() {
    LobModel::Config c;
    c.hawkes = {1e-6, 0.0, 1.0};
    c.seed   = 12345;
    return c;
}

/// 既定の流れ（μ 200 /s, α 100, β 200 → 定常 400 /s/側）。
LobModel::Config busy_config(std::uint64_t seed = 20240912) {
    LobModel::Config c;
    c.seed = seed;
    return c;
}

void advance(LobModel& m, int n) {
    for (int i = 0; i < n; ++i) m.step(kDt);
}

bool levels_equal(const LevelView& a, const LevelView& b) {
    return a.price == b.price && a.qty == b.qty && a.orders == b.orders;
}

bool fills_equal(const quantviz::core::micro::Fill& a, const quantviz::core::micro::Fill& b) {
    return a.maker == b.maker && a.taker == b.taker && a.taker_side == b.taker_side &&
           a.price == b.price && a.qty == b.qty && a.seq == b.seq;
}

/// Snapshot の全フィールドの一致（padding を除いた「bit 一致」）。
bool snapshots_equal(const LobSnapshot& a, const LobSnapshot& b) {
    if (a.t != b.t || a.tick_size != b.tick_size || a.mid != b.mid || a.spread != b.spread) return false;
    if (a.lambda_buy != b.lambda_buy || a.lambda_sell != b.lambda_sell) return false;
    if (a.mu != b.mu || a.alpha != b.alpha || a.beta != b.beta) return false;
    if (a.market_frac != b.market_frac || a.cancel_frac != b.cancel_frac) return false;
    if (a.mid_ticks != b.mid_ticks) return false;
    if (a.n_bids != b.n_bids || a.n_asks != b.n_asks || a.n_trades != b.n_trades) return false;
    for (std::uint32_t i = 0; i < a.n_bids; ++i)
        if (!levels_equal(a.bids[i], b.bids[i])) return false;
    for (std::uint32_t i = 0; i < a.n_asks; ++i)
        if (!levels_equal(a.asks[i], b.asks[i])) return false;
    for (std::uint32_t i = 0; i < a.n_trades; ++i)
        if (!fills_equal(a.trades[i], b.trades[i])) return false;
    return a.orders_submitted == b.orders_submitted && a.orders_cancelled == b.orders_cancelled &&
           a.orders_rejected == b.orders_rejected && a.orders_clamped == b.orders_clamped &&
           a.fills == b.fills &&
           a.orders_truncated == b.orders_truncated && a.discarded_qty == b.discarded_qty &&
           a.book_qty_bid == b.book_qty_bid && a.book_qty_ask == b.book_qty_ask &&
           a.pending_buy == b.pending_buy && a.pending_sell == b.pending_sell &&
           a.orders_in_book == b.orders_in_book && a.pool_free == b.pool_free && a.seq == b.seq;
}

}  // namespace

TEST_CASE("LOBSCENE-01: satisfies the Model contract with a fixed-size POD snapshot",
          "[lobscene][contract]") {
    STATIC_REQUIRE(quantviz::bridge::Model<LobModel>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<LobSnapshot>);
    STATIC_REQUIRE(std::is_standard_layout_v<LobSnapshot>);
    STATIC_REQUIRE(std::is_default_constructible_v<LobSnapshot>);
    // 上位 16 レベル × 2 側（24 B/レベル）+ 直近 32 約定（48 B/約定）= 2304 B + スカラー。
    STATIC_REQUIRE(sizeof(LobSnapshot) <= 4096);
    STATIC_REQUIRE(LobSnapshot::kLevels == 16);
    STATIC_REQUIRE(LobSnapshot::kTrades == 32);

    SECTION("a fresh model is seeded with ten levels a side, mid 100.00 and seq 0") {
        const LobModel     m;
        const LobSnapshot  s = m.snapshot();
        CHECK(s.seq == 0);
        CHECK(s.t == 0.0);
        CHECK(s.n_bids == LobModel::kInitialLevels);
        CHECK(s.n_asks == LobModel::kInitialLevels);
        CHECK_THAT(s.tick_size, WithinRel(0.01, 1e-15));
        CHECK_THAT(s.mid, WithinRel(100.0, 1e-12));
        CHECK_THAT(s.spread, WithinAbs(0.02, 1e-12));  // best bid 99.99 / best ask 100.01
        CHECK(s.n_trades == 0);
        CHECK(s.fills == 0);
        CHECK(s.orders_in_book == 2 * LobModel::kInitialLevels);
        CHECK(s.orders_submitted == 2 * LobModel::kInitialLevels);
        CHECK(s.pool_free == LobModel::kMaxOrders - 2 * LobModel::kInitialLevels);
        // 事象がまだ無いので λ = μ
        CHECK(s.lambda_buy == s.mu);
        CHECK(s.lambda_sell == s.mu);
        // bid は価格降順・ask は昇順、初期数量は [50, 150]
        for (std::uint32_t i = 1; i < s.n_bids; ++i) CHECK(s.bids[i].price < s.bids[i - 1].price);
        for (std::uint32_t i = 1; i < s.n_asks; ++i) CHECK(s.asks[i].price > s.asks[i - 1].price);
        for (std::uint32_t i = 0; i < s.n_bids; ++i) {
            CHECK(s.bids[i].qty >= LobModel::kInitialQtyMin);
            CHECK(s.bids[i].qty <= LobModel::kInitialQtyMax);
            CHECK(s.bids[i].orders == 1);
        }
        CHECK(s.bids[0].price == s.mid_ticks - 1);
        CHECK(s.asks[0].price == s.mid_ticks + 1);
    }

    SECTION("Reset rewinds seq to 0 and the sim clock to 0") {
        LobModel m(busy_config());
        advance(m, 50);
        REQUIRE(m.snapshot().seq == 50);
        REQUIRE(m.snapshot().t > 0.0);
        m.apply(Command::reset());
        const LobSnapshot s = m.snapshot();
        CHECK(s.seq == 0);
        CHECK(s.t == 0.0);
    }

    SECTION("unknown param ids and clock commands leave the model alone") {
        LobModel m(busy_config());
        advance(m, 20);
        const LobSnapshot before = m.snapshot();
        m.apply(Command::set_param(0, 1.0));      // 0 は予約（未知）
        m.apply(Command::set_param(999, 12.0));   // 未知
        m.apply(Command::pause());                // 時計系は Runner が処理済み
        m.apply(Command::resume());
        m.apply(Command::step_once());
        m.apply(Command::set_speed(4.0));
        CHECK(snapshots_equal(before, m.snapshot()));
    }

    SECTION("NaN and out-of-range parameters clamp into the stable region") {
        LobModel m;
        m.apply(Command::set_param(LobModel::kMu, std::nan("")));
        CHECK(m.snapshot().mu == LobModel::kMinMu);
        m.apply(Command::set_param(LobModel::kMu, -3.0));
        CHECK(m.snapshot().mu == LobModel::kMinMu);
        m.apply(Command::set_param(LobModel::kMu, 1e30));
        CHECK(m.snapshot().mu == LobModel::kMaxMu);

        m.apply(Command::set_param(LobModel::kBeta, 200.0));
        m.apply(Command::set_param(LobModel::kAlpha, std::nan("")));
        CHECK(m.snapshot().alpha == 0.0);
        m.apply(Command::set_param(LobModel::kAlpha, -1.0));
        CHECK(m.snapshot().alpha == 0.0);
        // 分岐比 α/β は kMaxBranching 以下に保つ（α を縮める）
        m.apply(Command::set_param(LobModel::kAlpha, 1e9));
        CHECK_THAT(m.snapshot().alpha, WithinRel(LobModel::kMaxBranching * 200.0, 1e-12));
        m.apply(Command::set_param(LobModel::kBeta, std::nan("")));
        CHECK(m.snapshot().beta == LobModel::kMinBeta);
        CHECK(m.snapshot().alpha <= LobModel::kMaxBranching * m.snapshot().beta);

        m.apply(Command::set_param(LobModel::kMarketFrac, std::nan("")));
        CHECK(m.snapshot().market_frac == 0.0);
        m.apply(Command::set_param(LobModel::kMarketFrac, 7.0));
        CHECK(m.snapshot().market_frac == LobModel::kMaxFrac);
        m.apply(Command::set_param(LobModel::kCancelFrac, std::nan("")));
        CHECK(m.snapshot().cancel_frac == 0.0);
        m.apply(Command::set_param(LobModel::kCancelFrac, 7.0));
        CHECK(m.snapshot().cancel_frac == LobModel::kMaxFrac);

        // 注入数量も NaN / 負を落とす
        m.apply(Command::set_param(LobModel::kInjectBuy, std::nan("")));
        m.apply(Command::set_param(LobModel::kInjectSell, -50.0));
        CHECK(m.snapshot().pending_buy == 0);
        CHECK(m.snapshot().pending_sell == 0);

        // クランプ後も回り続ける（NaN が板に漏れていない）
        advance(m, 20);
        CHECK(m.engine().book().check_invariants());
    }
}

TEST_CASE("LOBSCENE-02: an injected market order sweeps one side and leaves the other intact",
          "[lobscene][unit]") {
    SECTION("kInjectBuy lifts the best ask") {
        LobModel m(quiet_config());
        const LobSnapshot before = m.snapshot();
        m.apply(Command::set_param(LobModel::kInjectBuy, 500.0));
        CHECK(m.snapshot().pending_buy == 500);  // 次のステップの先頭で入る
        CHECK(m.snapshot().asks[0].price == before.asks[0].price);
        m.step(kDt);
        const LobSnapshot after = m.snapshot();
        CHECK(after.pending_buy == 0);
        CHECK(after.asks[0].price > before.asks[0].price);  // 買いの成行が ask を食い上げた
        CHECK(after.bids[0].price == before.bids[0].price);  // 反対側は無傷
        CHECK(after.bids[0].qty == before.bids[0].qty);
        CHECK(after.fills > 0);
        CHECK(after.n_trades == after.fills);
        CHECK(after.trades[0].taker_side == Side::Bid);
        CHECK(after.book_qty_ask == before.book_qty_ask - 500);  // 掃かれた数量ぶんだけ減る
        CHECK(after.mid > before.mid);
    }

    SECTION("kInjectSell drops the best bid") {
        LobModel m(quiet_config());
        const LobSnapshot before = m.snapshot();
        m.apply(Command::set_param(LobModel::kInjectSell, 500.0));
        m.step(kDt);
        const LobSnapshot after = m.snapshot();
        CHECK(after.bids[0].price < before.bids[0].price);
        CHECK(after.asks[0].price == before.asks[0].price);
        CHECK(after.asks[0].qty == before.asks[0].qty);
        CHECK(after.trades[0].taker_side == Side::Ask);
        CHECK(after.book_qty_bid == before.book_qty_bid - 500);
        CHECK(after.mid < before.mid);
    }

    SECTION("an injection larger than the book empties that side; counters stay consistent") {
        LobModel m(quiet_config());
        const LobSnapshot before = m.snapshot();
        m.apply(Command::set_param(LobModel::kInjectBuy, 1'000'000.0));
        m.step(kDt);
        const LobSnapshot after = m.snapshot();
        CHECK(after.n_asks == 0);
        CHECK(after.book_qty_ask == 0);
        CHECK(after.spread == 0.0);           // 片側が空 → スプレッドは未定義（0）
        CHECK(after.mid == before.mid);       // mid は直前の値を保つ
        CHECK(after.n_bids == before.n_bids);  // 反対側は無傷
        for (std::uint32_t i = 0; i < before.n_bids; ++i) CHECK(levels_equal(after.bids[i], before.bids[i]));
        CHECK(after.fills == LobModel::kInitialLevels);  // ask 側の 10 注文
        CHECK(after.n_trades == after.fills);  // 10 約定なので直近 32 件のリングに全部載る
        CHECK(after.orders_truncated == 0);  // 10 約定は fills バッファ（256）に収まる
        CHECK(after.discarded_qty == 1'000'000 - before.book_qty_ask);
        CHECK(after.orders_in_book == before.n_bids);
        CHECK(m.engine().book().check_invariants());

        // 板は指値の流入で回復する（流れを戻して数ステップ）
        m.apply(Command::set_param(LobModel::kMu, 2000.0));
        advance(m, 50);
        CHECK(m.snapshot().n_asks > 0);
    }
}

TEST_CASE("LOBSCENE-03: the ladder mirrors the engine and the same seed replays bit-identically",
          "[lobscene][determinism]") {
    LobModel a(busy_config(777));
    LobModel b(busy_config(777));
    REQUIRE(snapshots_equal(a.snapshot(), b.snapshot()));

    std::array<LevelView, LobSnapshot::kLevels> bid_ref{};
    std::array<LevelView, LobSnapshot::kLevels> ask_ref{};

    for (int i = 0; i < 2000; ++i) {
        if (i == 500) {  // 同じ Command 列を両方に流す
            a.apply(Command::set_param(LobModel::kInjectBuy, 400.0));
            b.apply(Command::set_param(LobModel::kInjectBuy, 400.0));
        }
        if (i == 1200) {
            a.apply(Command::set_param(LobModel::kMarketFrac, 0.25));
            b.apply(Command::set_param(LobModel::kMarketFrac, 0.25));
        }
        a.step(kDt);
        b.step(kDt);
        const LobSnapshot sa = a.snapshot();
        REQUIRE(snapshots_equal(sa, b.snapshot()));

        // Snapshot の深度は板の depth(16) そのもの
        const std::size_t nb = a.engine().book().depth(Side::Bid, bid_ref);
        const std::size_t na = a.engine().book().depth(Side::Ask, ask_ref);
        REQUIRE(sa.n_bids == nb);
        REQUIRE(sa.n_asks == na);
        for (std::size_t k = 0; k < nb; ++k) REQUIRE(levels_equal(sa.bids[k], bid_ref[k]));
        for (std::size_t k = 0; k < na; ++k) REQUIRE(levels_equal(sa.asks[k], ask_ref[k]));
    }

    // 数量保存（LOB-08）がシーン越しにも成り立つ: 側ごとに
    //   投入 == 約定 + 取消 + 破棄 + 板に残っている数量、かつ約定数量は買い == 売り
    const auto& eng = a.engine();
    for (const Side side : {Side::Bid, Side::Ask}) {
        CHECK(eng.submitted_qty(side) == eng.filled_qty(side) + eng.cancelled_qty(side) +
                                             eng.discarded_qty(side) + eng.book().total_qty(side));
    }
    CHECK(eng.filled_qty(Side::Bid) == eng.filled_qty(Side::Ask));
    CHECK(eng.filled_qty(Side::Bid) > 0);

    const LobSnapshot end = a.snapshot();
    CHECK(end.seq == 2000);
    CHECK(end.fills > 100);             // 2 秒で十分な約定が起きている
    CHECK(end.orders_submitted > 500);  // 流れが実際に届いている
    CHECK(end.n_bids > 0);
    CHECK(end.n_asks > 0);
    CHECK(a.engine().book().check_invariants());
}

TEST_CASE("LOBSCENE-04: Reset empties and re-seeds the book, rewinds lambda to mu and drops live ids",
          "[lobscene][unit]") {
    LobModel m(busy_config(4242));
    advance(m, 500);
    const LobSnapshot busy = m.snapshot();
    REQUIRE(busy.lambda_buy > busy.mu);  // 励起が乗っている
    REQUIRE(busy.fills > 0);
    REQUIRE(busy.orders_in_book > 0);
    REQUIRE(busy.n_bids > 0);  // front(Bid) を参照外しする前に板が空でないことを確かめる
    // リセット前の生存注文の id（板から取り出す。id はプールのスロット再利用で使い回される）
    const OrderId old_id = m.engine().book().front(Side::Bid)->id;
    REQUIRE(m.engine().book().find(old_id).has_value());

    m.apply(Command::reset());
    const LobSnapshot s = m.snapshot();
    CHECK(s.seq == 0);
    CHECK(s.t == 0.0);
    CHECK(s.lambda_buy == s.mu);
    CHECK(s.lambda_sell == s.mu);
    CHECK(s.mu == busy.mu);  // Reset はパラメータを保つ
    CHECK(s.n_bids == LobModel::kInitialLevels);
    CHECK(s.n_asks == LobModel::kInitialLevels);
    CHECK_THAT(s.mid, WithinRel(100.0, 1e-12));
    CHECK_THAT(s.spread, WithinAbs(0.02, 1e-12));
    CHECK(s.fills == 0);
    CHECK(s.n_trades == 0);
    CHECK(s.orders_cancelled == 0);
    CHECK(s.orders_submitted == 2 * LobModel::kInitialLevels);
    CHECK(s.orders_in_book == 2 * LobModel::kInitialLevels);

    // live-id リストは engine.reset() と同じ操作で捨てる（敷き直した 20 件だけが残る）
    CHECK(m.live_count(Side::Bid) == LobModel::kInitialLevels);
    CHECK(m.live_count(Side::Ask) == LobModel::kInitialLevels);
    // リセット前の id は板に居ない（持ち越すと再利用されたスロットの別注文を取り消してしまう）
    CHECK(!m.engine().book().find(old_id).has_value());

    // Reset 直後は同 seed の新品と一致し、以後の系列も一致する
    LobModel fresh(busy_config(4242));
    CHECK(snapshots_equal(s, fresh.snapshot()));
    advance(m, 200);
    advance(fresh, 200);
    CHECK(snapshots_equal(m.snapshot(), fresh.snapshot()));

    // Command::reset(seed) は seed を差し替える
    m.apply(Command::reset(99));
    LobModel other(busy_config(99));
    CHECK(snapshots_equal(m.snapshot(), other.snapshot()));
}

TEST_CASE("LOBSCENE-05: the per-step thinning produces the closed-form number of arrivals",
          "[lobscene][statistical]") {
    // 成行・取消を 0 にすると全ての到着が指値になり（価格は必ず板の範囲内にクランプされるので
    // 拒否は起きない）、engine の orders_submitted から初期板の 2×kInitialLevels を引いた値が
    // そのまま「窓 [0, T) に生成した事象数（買い + 売り）」になる。T = 10000 ステップ × 1 ms = 10 s。
    //
    // 期待値は hawkes.hpp の閉形式（空の履歴から始めた有限窓の厳密値）:
    //   E[N(T)] = μ/(1−η)·[T − η(1 − e^{−β(1−η)T})/(β(1−η))]、η = α/β  …… 片側
    // 許容は 4 SE（SE は同じ設定を 200 seed 回した標本標準偏差。docs/03_tdd_spec.md §2.4）。
    constexpr double kT   = 10.0;
    constexpr int    kN   = 10000;
    constexpr double kDt5 = kT / kN;

    auto arrivals = [](HawkesParams p) {
        LobModel::Config c;
        c.hawkes      = p;
        c.market_frac = 0.0;  // 全事象が指値 = 到着 1 件につき orders_submitted が 1 増える
        c.cancel_frac = 0.0;
        c.seed        = 4242;
        LobModel m(c);
        for (int i = 0; i < kN; ++i) m.step(kDt5);
        const LobSnapshot s = m.snapshot();
        CHECK(s.orders_rejected == 0);  // プール枯渇も範囲外もなし（= 到着数の数え方が正しい）
        CHECK(s.orders_clamped == 0);   // 値幅の壁にも当たっていない
        return static_cast<double>(s.orders_submitted) - 2.0 * LobModel::kInitialLevels;
    };

    SECTION("alpha = 0 is a Poisson stream: E = 2 x mu x T = 4000, SE = sqrt(4000) = 63.2") {
        // 両側 μ = 200 /s、T = 10 s → E = 4000。Poisson なので SE = sqrt(E) = 63.2
        // （200 seed の標本標準偏差 63.3 と一致）。許容 4 SE = 253。
        const double n = arrivals({200.0, 0.0, 200.0});
        CHECK_THAT(n, WithinAbs(4000.0, 4.0 * 63.2));
    }

    SECTION("eta = 0.5 self-excitation: E = 7996 from the finite-horizon formula, SE = 186") {
        // μ = 200, α = 100, β = 200 → η = 0.5。片側 E = 400·[10 − 0.5(1 − e^{−1000})/100] = 3998、
        // 両側 7996。SE は 200 seed の標本標準偏差 186.1（漸近式 sqrt(2·E_side/(1−η)²) = 178.8 と
        // 同程度）。許容 4 SE = 744。200 seed の平均は 8002.4（E から 0.5 SE 以内）。
        const double n = arrivals({200.0, 100.0, 200.0});
        CHECK_THAT(n, WithinAbs(7996.0, 4.0 * 186.0));
    }
}
