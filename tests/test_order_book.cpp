// LOB-xx — core/micro/order_book.hpp, core/micro/matching_engine.hpp の仕様テスト
// 空の板・best の更新・列挙順（LOB-01..03）、価格時間優先・部分約定・取消（LOB-05..07）、
// 成行の複数レベル掃き・交差指値（LOB-09..10）、深度（LOB-13）、プール枯渇と無アロケーション（LOB-14）、
// seed 固定の合成フロー 1e5 件で不変条件・数量保存・非交差・決定性（LOB-04, 08, 11, 12）を検証する。
//
// LOB-14 のアロケーション計数: このファイルはグローバル operator new / delete を置き換え、呼び出し回数を
// 数える（テストバイナリ全体に効くが、数えるだけで挙動は変えない）。ウォームアップ（コンストラクタ）後に
// カウンタが動かないことを確認する。
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <optional>
#include <span>
#include <vector>

#include "quantviz/core/micro/matching_engine.hpp"
#include "quantviz/core/micro/order_book.hpp"
#include "quantviz/core/rng.hpp"

using quantviz::core::Rng;
using quantviz::core::micro::Fill;
using quantviz::core::micro::LevelView;
using quantviz::core::micro::MatchingEngine;
using quantviz::core::micro::OrderBook;
using quantviz::core::micro::OrderId;
using quantviz::core::micro::Price;
using quantviz::core::micro::Qty;
using quantviz::core::micro::Side;

// ---------------------------------------------------------------------------------------------------
// グローバル new/delete の置き換え（LOB-14 用の計数。malloc/free に委譲するだけ）
// GCC は置き換えた operator delete の free をインライン先で「new と対応しない解放」と誤検知する
// （-Wmismatched-new-delete の既知の偽陽性）ので、この 4 関数の間だけ抑止する。
// ---------------------------------------------------------------------------------------------------
namespace {
std::size_t g_alloc_count = 0;
}

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void* operator new(std::size_t n) {
    ++g_alloc_count;
    if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) {
    ++g_alloc_count;
    if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace {

// 単体テスト用の小さな板: 64 注文、価格 [990, 1022)
constexpr Price kBase = 990;
using SmallEngine     = MatchingEngine<64, 32>;
using SmallBook       = OrderBook<64, 32>;

// 合成フロー用: 8192 注文、価格 [744, 1256)。このフローは 1 操作あたり約 0.045 件ずつ板が厚くなる
// （1e5 操作で ≈ 4500 件）ので 4096 では枯渇する。Debug では公開操作ごとに check_invariants()
//（O(MaxOrders + MaxLevels + 生存数)）が走るので、これ以上は増やさない（LOB-04 が Debug で 10 s 未満）。
constexpr Price kFlowBase   = 744;
constexpr Price kFlowAnchor = 1000;
using FlowEngine            = MatchingEngine<8192, 512>;

struct FillBuf {
    std::array<Fill, 64> fills{};
    std::size_t          n = 0;
    std::span<Fill>      span() noexcept { return fills; }
};

/// 合成フロー 1 件を engine に流す。70 % 指値 / 10 % 成行 / 20 % 取消、価格は anchor ± 20 ティック、
/// 数量 1〜100。取消対象は live に残した id からランダムに選ぶ（既に約定で消えた id も混ざる =
/// cancel が false を返す経路も踏む）。all_fills が非 null なら約定を追記する。
struct FlowStep {
    Side          side;
    Price         price;
    Qty           qty;
    OrderId       id;           // 指値: 付与 id（0 = 拒否）。取消: 対象 id
    Qty           filled;       // 成行の約定数量
    int           kind;         // 0 指値 1 成行 2 取消
    bool          cancel_ok;
    Side          cancel_side;  // 取消が成功したとき: 取り消した注文の側と残数量（取消前に find で控える）
    Qty           cancel_qty;
};

FlowStep flow_step(FlowEngine& e, Rng& rng, std::vector<OrderId>& live, std::vector<Fill>* all_fills) {
    FillBuf  buf;
    FlowStep s{};
    const double u = rng.uniform();
    s.side         = rng.uniform() < 0.5 ? Side::Bid : Side::Ask;
    s.qty          = 1 + static_cast<Qty>(rng.uniform() * 100.0);
    s.price        = kFlowAnchor + static_cast<Price>(rng.uniform() * 41.0) - 20;
    if (u < 0.7) {
        s.kind = 0;
        s.id   = e.submit_limit(s.side, s.price, s.qty, buf.span(), buf.n);
        if (s.id != 0) live.push_back(s.id);
    } else if (u < 0.8) {
        s.kind   = 1;
        s.filled = e.submit_market(s.side, s.qty, buf.span(), buf.n);
    } else {
        s.kind = 2;
        if (!live.empty()) {
            const std::size_t k = static_cast<std::size_t>(rng.uniform() * static_cast<double>(live.size()));
            s.id                = live[k];
            live[k]             = live.back();
            live.pop_back();
            if (const auto v = e.book().find(s.id)) {
                s.cancel_side = v->side;
                s.cancel_qty  = v->qty;
            }
            s.cancel_ok = e.cancel(s.id);
        }
    }
    if (all_fills)
        for (std::size_t i = 0; i < buf.n; ++i) all_fills->push_back(buf.fills[i]);
    return s;
}

bool uncrossed(const FlowEngine& e) {
    const auto bb = e.book().best_bid();
    const auto ba = e.book().best_ask();
    return !bb || !ba || *bb < *ba;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------

TEST_CASE("LOB-01: an empty book has no best bid and no best ask", "[lob][unit]") {
    SmallBook book(kBase);
    CHECK_FALSE(book.best_bid().has_value());
    CHECK_FALSE(book.best_ask().has_value());
    CHECK(book.order_count() == 0);
    CHECK(book.pool_free() == 64);
    CHECK(book.total_qty(Side::Bid) == 0);
    CHECK(book.total_qty(Side::Ask) == 0);
    CHECK(book.check_invariants());

    SmallEngine engine(kBase);
    CHECK_FALSE(engine.book().best_bid().has_value());
    CHECK_FALSE(engine.book().best_ask().has_value());
    CHECK(engine.next_seq() == 1);
}

TEST_CASE("LOB-02: a limit bid becomes the best bid", "[lob][unit]") {
    SmallEngine engine(kBase);
    FillBuf     buf;
    const OrderId id = engine.submit_limit(Side::Bid, 1005, 10, buf.span(), buf.n);
    REQUIRE(id != 0);
    CHECK(buf.n == 0);
    REQUIRE(engine.book().best_bid().has_value());
    CHECK(*engine.book().best_bid() == 1005);
    CHECK_FALSE(engine.book().best_ask().has_value());
    CHECK(engine.book().level_qty(Side::Bid, 1005) == 10);
    CHECK(engine.book().level_orders(Side::Bid, 1005) == 1);
    CHECK(engine.book().order_count() == 1);

    // 高い bid が来れば best が更新され、低い bid では変わらない
    REQUIRE(engine.submit_limit(Side::Bid, 1003, 5, buf.span(), buf.n) != 0);
    CHECK(*engine.book().best_bid() == 1005);
    REQUIRE(engine.submit_limit(Side::Bid, 1007, 5, buf.span(), buf.n) != 0);
    CHECK(*engine.book().best_bid() == 1007);

    // ask 側も対称
    REQUIRE(engine.submit_limit(Side::Ask, 1010, 5, buf.span(), buf.n) != 0);
    REQUIRE(engine.submit_limit(Side::Ask, 1012, 5, buf.span(), buf.n) != 0);
    REQUIRE(engine.book().best_ask().has_value());
    CHECK(*engine.book().best_ask() == 1010);
    CHECK(buf.n == 0);
    CHECK(engine.book().check_invariants());
}

TEST_CASE("LOB-03: bids enumerate in descending and asks in ascending price order", "[lob][property]") {
    FlowEngine engine(kFlowBase);
    Rng        rng(2718);
    FillBuf    buf;
    // 交差しない受動的な注文だけ: bid は [900, 990]、ask は [1010, 1100]
    Qty bid_total = 0, ask_total = 0;
    for (int i = 0; i < 500; ++i) {
        const Price pb = 900 + static_cast<Price>(rng.uniform() * 91.0);
        const Price pa = 1010 + static_cast<Price>(rng.uniform() * 91.0);
        const Qty   q  = 1 + static_cast<Qty>(rng.uniform() * 100.0);
        REQUIRE(engine.submit_limit(Side::Bid, pb, q, buf.span(), buf.n) != 0);
        REQUIRE(engine.submit_limit(Side::Ask, pa, q, buf.span(), buf.n) != 0);
        bid_total += q;
        ask_total += q;
    }
    REQUIRE(buf.n == 0);

    std::vector<LevelView> out(512);
    const std::size_t      nb = engine.book().depth(Side::Bid, out);
    REQUIRE(nb > 1);
    Qty seen = 0;
    for (std::size_t i = 0; i < nb; ++i) {
        CHECK(out[i].qty > 0);
        CHECK(out[i].orders > 0);
        CHECK(out[i].qty == engine.book().level_qty(Side::Bid, out[i].price));
        if (i > 0) CHECK(out[i].price < out[i - 1].price);
        seen += out[i].qty;
    }
    CHECK(seen == bid_total);
    CHECK(out[0].price == *engine.book().best_bid());

    const std::size_t na = engine.book().depth(Side::Ask, out);
    REQUIRE(na > 1);
    seen = 0;
    for (std::size_t i = 0; i < na; ++i) {
        CHECK(out[i].qty > 0);
        if (i > 0) CHECK(out[i].price > out[i - 1].price);
        seen += out[i].qty;
    }
    CHECK(seen == ask_total);
    CHECK(out[0].price == *engine.book().best_ask());
    CHECK(engine.book().check_invariants());
}

TEST_CASE("LOB-04: after 1e5 random operations best bid < best ask and all invariants hold",
          "[lob][property]") {
    FlowEngine           engine(kFlowBase);
    Rng                  rng(31415);
    std::vector<OrderId> live;
    live.reserve(8192);

    constexpr int kOps   = 100'000;
    constexpr int kBlock = 100;
    int           crossed = 0, broken = 0;
    std::uint64_t n_limit = 0, n_market = 0, n_cancel_ok = 0, n_cancel_gone = 0, n_rejected = 0;
    for (int i = 0; i < kOps; ++i) {
        const FlowStep s = flow_step(engine, rng, live, nullptr);
        if (s.kind == 0) {
            ++n_limit;
            if (s.id == 0) ++n_rejected;
        } else if (s.kind == 1) {
            ++n_market;
        } else if (s.id != 0) {
            if (s.cancel_ok) ++n_cancel_ok;
            else ++n_cancel_gone;
        }
        if (!uncrossed(engine)) ++crossed;
        if (i % 10 == 9 && !engine.book().check_invariants(false)) ++broken;  // 高速版（free list なし）
        if (i % kBlock == kBlock - 1 && !engine.book().check_invariants()) ++broken;  // 完全版
    }
    CHECK(crossed == 0);
    CHECK(broken == 0);
    CHECK(engine.book().check_invariants());
    CHECK(uncrossed(engine));
    // フローが実際に全経路を踏んだこと（退化していない）
    CHECK(n_limit > 60'000);
    CHECK(n_market > 8'000);
    CHECK(n_cancel_ok > 1'000);
    CHECK(n_cancel_gone > 100);
    CHECK(n_rejected == 0);  // プール 8192 は枯渇しない（このフローは ≈ 4500 件で頭打ち）
    CHECK(engine.next_seq() > 10'000);
}

TEST_CASE("LOB-05: at the same price the earlier order fills first (price-time priority)", "[lob][unit]") {
    SmallEngine engine(kBase);
    FillBuf     buf;
    const OrderId a = engine.submit_limit(Side::Ask, 1010, 10, buf.span(), buf.n);
    const OrderId b = engine.submit_limit(Side::Ask, 1010, 10, buf.span(), buf.n);
    const OrderId c = engine.submit_limit(Side::Ask, 1010, 10, buf.span(), buf.n);
    REQUIRE(a != 0);
    REQUIRE(b != 0);
    REQUIRE(c != 0);
    CHECK(a < b);
    CHECK(b < c);
    CHECK(engine.book().level_orders(Side::Ask, 1010) == 3);

    const Qty filled = engine.submit_market(Side::Bid, 15, buf.span(), buf.n);
    CHECK(filled == 15);
    REQUIRE(buf.n == 2);
    CHECK(buf.fills[0].maker == a);
    CHECK(buf.fills[0].qty == 10);
    CHECK(buf.fills[0].price == 1010);
    CHECK(buf.fills[0].taker_side == Side::Bid);
    CHECK(buf.fills[1].maker == b);
    CHECK(buf.fills[1].qty == 5);
    CHECK(buf.fills[1].seq == buf.fills[0].seq + 1);
    CHECK(buf.fills[0].taker == buf.fills[1].taker);
    CHECK(buf.fills[0].taker != 0);

    // a は消え、b は 5 残り、c は手つかず
    CHECK_FALSE(engine.book().find(a).has_value());
    REQUIRE(engine.book().find(b).has_value());
    CHECK(engine.book().find(b)->qty == 5);
    CHECK(engine.book().find(c)->qty == 10);
    CHECK(engine.book().level_orders(Side::Ask, 1010) == 2);
    CHECK(engine.book().level_qty(Side::Ask, 1010) == 15);
    CHECK_FALSE(engine.cancel(a));
    CHECK(engine.book().check_invariants());
}

TEST_CASE("LOB-06: a partial fill leaves the remaining quantity in the book", "[lob][unit]") {
    SmallEngine engine(kBase);
    FillBuf     buf;

    SECTION("resting ask is partially taken by a smaller crossing bid") {
        const OrderId ask = engine.submit_limit(Side::Ask, 1005, 100, buf.span(), buf.n);
        REQUIRE(ask != 0);
        const OrderId bid = engine.submit_limit(Side::Bid, 1005, 30, buf.span(), buf.n);
        REQUIRE(bid != 0);
        REQUIRE(buf.n == 1);
        CHECK(buf.fills[0].maker == ask);
        CHECK(buf.fills[0].taker == bid);
        CHECK(buf.fills[0].qty == 30);
        CHECK(buf.fills[0].price == 1005);
        CHECK(engine.book().level_qty(Side::Ask, 1005) == 70);
        CHECK(engine.book().find(ask)->qty == 70);
        CHECK_FALSE(engine.book().find(bid).has_value());  // 全約定したテイカーは板に載らない
        CHECK_FALSE(engine.book().best_bid().has_value());
        CHECK(*engine.book().best_ask() == 1005);
    }
    SECTION("incoming limit is partially filled and the rest rests") {
        REQUIRE(engine.submit_limit(Side::Ask, 1005, 30, buf.span(), buf.n) != 0);
        const OrderId bid = engine.submit_limit(Side::Bid, 1005, 100, buf.span(), buf.n);
        REQUIRE(bid != 0);
        REQUIRE(buf.n == 1);
        CHECK(buf.fills[0].qty == 30);
        CHECK_FALSE(engine.book().best_ask().has_value());
        CHECK(*engine.book().best_bid() == 1005);
        CHECK(engine.book().level_qty(Side::Bid, 1005) == 70);
        CHECK(engine.book().find(bid)->qty == 70);
        CHECK(engine.book().find(bid)->side == Side::Bid);
    }
    CHECK(engine.book().check_invariants());
}

TEST_CASE("LOB-07: cancel removes the order and an emptied level disappears", "[lob][unit]") {
    SmallEngine engine(kBase);
    FillBuf     buf;
    const OrderId b1 = engine.submit_limit(Side::Bid, 1000, 10, buf.span(), buf.n);
    const OrderId b2 = engine.submit_limit(Side::Bid, 1000, 20, buf.span(), buf.n);
    const OrderId b3 = engine.submit_limit(Side::Bid, 999, 5, buf.span(), buf.n);
    REQUIRE(b1 != 0);
    REQUIRE(b2 != 0);
    REQUIRE(b3 != 0);
    CHECK(engine.book().order_count() == 3);
    CHECK(engine.book().pool_free() == 61);

    CHECK(engine.cancel(b1));
    CHECK(engine.book().level_orders(Side::Bid, 1000) == 1);
    CHECK(engine.book().level_qty(Side::Bid, 1000) == 20);
    CHECK(*engine.book().best_bid() == 1000);
    CHECK(engine.book().check_invariants());

    CHECK(engine.cancel(b2));
    CHECK(engine.book().level_orders(Side::Bid, 1000) == 0);
    CHECK(engine.book().level_qty(Side::Bid, 1000) == 0);
    CHECK(*engine.book().best_bid() == 999);  // best は次の非空レベルへ
    CHECK(engine.book().order_count() == 1);
    CHECK(engine.book().pool_free() == 63);
    CHECK(engine.book().check_invariants());

    CHECK_FALSE(engine.cancel(b2));  // 二重取消
    CHECK_FALSE(engine.cancel(0));
    CHECK_FALSE(engine.cancel(b3 + 12345));  // 存在しない id
    CHECK(engine.cancelled_qty(Side::Bid) == 30);
    CHECK(engine.orders_cancelled() == 2);

    CHECK(engine.cancel(b3));
    CHECK_FALSE(engine.book().best_bid().has_value());
    CHECK(engine.book().order_count() == 0);
    CHECK(engine.book().pool_free() == 64);
    CHECK(engine.book().check_invariants());

    // 解放されたスロットに新しい注文が入っても、古い id では取り消せない
    const OrderId b4 = engine.submit_limit(Side::Bid, 1001, 7, buf.span(), buf.n);
    REQUIRE(b4 != 0);
    CHECK(b4 > b3);
    CHECK_FALSE(engine.cancel(b1));
    CHECK_FALSE(engine.cancel(b2));
    CHECK_FALSE(engine.cancel(b3));
    CHECK(engine.book().order_count() == 1);
    CHECK(engine.book().check_invariants());
}

TEST_CASE("LOB-08: quantity conserved: buy == sell fills, submitted == filled+cancelled+discarded+resting",
          "[lob][property]") {
    FlowEngine           engine(kFlowBase);
    Rng                  rng(31415);
    std::vector<OrderId> live;
    std::vector<Fill>    fills;
    live.reserve(8192);
    fills.reserve(200'000);

    // テスト側で独立に集計する台帳
    Qty submitted[2] = {0, 0}, cancelled[2] = {0, 0}, discarded[2] = {0, 0};
    for (int i = 0; i < 100'000; ++i) {
        const std::size_t before = fills.size();
        const FlowStep    s      = flow_step(engine, rng, live, &fills);
        Qty               new_fill_qty = 0;
        for (std::size_t k = before; k < fills.size(); ++k) new_fill_qty += fills[k].qty;
        const auto si = static_cast<std::size_t>(s.side);
        if (s.kind == 0 && s.id != 0) {
            submitted[si] += s.qty;
            const auto rest    = engine.book().find(s.id);
            const Qty  resting = rest ? rest->qty : 0;
            REQUIRE(s.qty >= new_fill_qty + resting);
            discarded[si] += s.qty - new_fill_qty - resting;  // fills バッファ満杯で捨てた分
        } else if (s.kind == 1) {
            submitted[si] += s.qty;
            REQUIRE(s.filled == new_fill_qty);
            discarded[si] += s.qty - s.filled;
        } else if (s.kind == 2 && s.cancel_ok) {
            REQUIRE(s.cancel_qty > 0);
            cancelled[static_cast<std::size_t>(s.cancel_side)] += s.cancel_qty;
            REQUIRE(new_fill_qty == 0);
        } else {
            REQUIRE(new_fill_qty == 0);
        }
    }

    // 約定の買い合計 = 売り合計（各約定は買い手 1 と売り手 1 を持つ）: 記録された約定から独立に集計する
    Qty         total = 0, taker_buy = 0, taker_sell = 0;
    std::size_t bad_fills = 0;
    for (const Fill& f : fills) {
        total += f.qty;
        (f.taker_side == Side::Bid ? taker_buy : taker_sell) += f.qty;
        if (f.qty == 0 || f.maker == 0 || f.taker == 0 || f.maker == f.taker) ++bad_fills;
    }
    CHECK(bad_fills == 0);
    CHECK(taker_buy > 0);  // 両側がテイカーになる経路を踏んだ
    CHECK(taker_sell > 0);
    CHECK(engine.filled_qty(Side::Bid) == total);
    CHECK(engine.filled_qty(Side::Ask) == total);
    CHECK(engine.filled_qty(Side::Bid) == engine.filled_qty(Side::Ask));
    CHECK(fills.size() == engine.next_seq() - 1);
    for (std::size_t k = 1; k < fills.size(); ++k) CHECK(fills[k].seq == fills[k - 1].seq + 1);

    for (Side side : {Side::Bid, Side::Ask}) {
        const auto si = static_cast<std::size_t>(side);
        INFO("side=" << si);
        CHECK(engine.submitted_qty(side) == submitted[si]);
        CHECK(engine.cancelled_qty(side) == cancelled[si]);
        CHECK(engine.discarded_qty(side) == discarded[si]);
        CHECK(engine.submitted_qty(side) == engine.filled_qty(side) + engine.cancelled_qty(side) +
                                                engine.discarded_qty(side) + engine.book().total_qty(side));
    }
    CHECK(engine.discarded_qty(Side::Bid) + engine.discarded_qty(Side::Ask) > 0);  // 経路を踏んだ
    CHECK(engine.book().check_invariants());
}

TEST_CASE("LOB-09: a market order sweeps multiple levels", "[lob][unit]") {
    SmallEngine engine(kBase);
    FillBuf     buf;
    const OrderId a1 = engine.submit_limit(Side::Ask, 1001, 10, buf.span(), buf.n);
    const OrderId a2 = engine.submit_limit(Side::Ask, 1002, 10, buf.span(), buf.n);
    const OrderId a3 = engine.submit_limit(Side::Ask, 1003, 10, buf.span(), buf.n);
    REQUIRE((a1 != 0 && a2 != 0 && a3 != 0));

    CHECK(engine.submit_market(Side::Bid, 25, buf.span(), buf.n) == 25);
    REQUIRE(buf.n == 3);
    CHECK(buf.fills[0].price == 1001);
    CHECK(buf.fills[0].qty == 10);
    CHECK(buf.fills[1].price == 1002);
    CHECK(buf.fills[1].qty == 10);
    CHECK(buf.fills[2].price == 1003);
    CHECK(buf.fills[2].qty == 5);
    CHECK(buf.fills[2].maker == a3);
    CHECK(*engine.book().best_ask() == 1003);
    CHECK(engine.book().level_qty(Side::Ask, 1003) == 5);
    CHECK(engine.book().order_count() == 1);
    CHECK(engine.discarded_qty(Side::Bid) == 0);
    CHECK(engine.book().check_invariants());

    // 板が尽きたら残りは捨てる（返り値 = 約定数量、discarded に計上）
    buf.n = 0;
    CHECK(engine.submit_market(Side::Bid, 100, buf.span(), buf.n) == 5);
    REQUIRE(buf.n == 1);
    CHECK(buf.fills[0].qty == 5);
    CHECK_FALSE(engine.book().best_ask().has_value());
    CHECK(engine.discarded_qty(Side::Bid) == 95);
    CHECK(engine.submitted_qty(Side::Bid) == 125);
    CHECK(engine.filled_qty(Side::Bid) == 30);
    // 空の板への成行は全量捨て
    buf.n = 0;
    CHECK(engine.submit_market(Side::Ask, 7, buf.span(), buf.n) == 0);
    CHECK(buf.n == 0);
    CHECK(engine.discarded_qty(Side::Ask) == 7);
    CHECK(engine.book().check_invariants());
}

TEST_CASE("LOB-10: a limit crossing the spread fills aggressively and the remainder rests", "[lob][unit]") {
    SmallEngine engine(kBase);
    FillBuf     buf;
    REQUIRE(engine.submit_limit(Side::Ask, 1001, 10, buf.span(), buf.n) != 0);
    REQUIRE(engine.submit_limit(Side::Ask, 1002, 10, buf.span(), buf.n) != 0);
    REQUIRE(engine.submit_limit(Side::Ask, 1004, 10, buf.span(), buf.n) != 0);

    SECTION("crossing two levels, remainder rests at its limit") {
        const OrderId bid = engine.submit_limit(Side::Bid, 1002, 25, buf.span(), buf.n);
        REQUIRE(bid != 0);
        REQUIRE(buf.n == 2);
        CHECK(buf.fills[0].price == 1001);  // 約定価格はメイカーの価格（価格改善）
        CHECK(buf.fills[0].qty == 10);
        CHECK(buf.fills[0].taker == bid);
        CHECK(buf.fills[1].price == 1002);
        CHECK(buf.fills[1].qty == 10);
        CHECK(*engine.book().best_bid() == 1002);
        CHECK(engine.book().level_qty(Side::Bid, 1002) == 5);
        CHECK(*engine.book().best_ask() == 1004);
        CHECK(engine.book().find(bid)->qty == 5);
    }
    SECTION("only levels at or better than the limit are taken") {
        const OrderId bid = engine.submit_limit(Side::Bid, 1001, 25, buf.span(), buf.n);
        REQUIRE(bid != 0);
        REQUIRE(buf.n == 1);
        CHECK(buf.fills[0].price == 1001);
        CHECK(buf.fills[0].qty == 10);
        CHECK(*engine.book().best_bid() == 1001);
        CHECK(engine.book().level_qty(Side::Bid, 1001) == 15);
        CHECK(*engine.book().best_ask() == 1002);
    }
    SECTION("a limit that sweeps everything rests alone") {
        const OrderId bid = engine.submit_limit(Side::Bid, 1010, 31, buf.span(), buf.n);
        REQUIRE(bid != 0);
        REQUIRE(buf.n == 3);
        CHECK_FALSE(engine.book().best_ask().has_value());
        CHECK(*engine.book().best_bid() == 1010);
        CHECK(engine.book().level_qty(Side::Bid, 1010) == 1);
    }
    SECTION("symmetric: a crossing ask takes bids from the top down") {
        REQUIRE(engine.submit_limit(Side::Bid, 999, 10, buf.span(), buf.n) != 0);
        REQUIRE(engine.submit_limit(Side::Bid, 998, 10, buf.span(), buf.n) != 0);
        buf.n = 0;
        const OrderId ask = engine.submit_limit(Side::Ask, 998, 15, buf.span(), buf.n);
        REQUIRE(ask != 0);
        REQUIRE(buf.n == 2);
        CHECK(buf.fills[0].price == 999);
        CHECK(buf.fills[0].taker_side == Side::Ask);
        CHECK(buf.fills[1].price == 998);
        CHECK(buf.fills[1].qty == 5);
        CHECK(*engine.book().best_bid() == 998);
        CHECK(engine.book().level_qty(Side::Bid, 998) == 5);
        CHECK(*engine.book().best_ask() == 1001);
    }
    CHECK(engine.book().check_invariants());
}

TEST_CASE("LOB-11: no self-crossing: the book never holds bid >= ask", "[lob][property]") {
    SECTION("full fills buffer discards the still-crossing remainder instead of resting it") {
        SmallEngine engine(kBase);
        FillBuf     buf;
        REQUIRE(engine.submit_limit(Side::Ask, 1001, 10, buf.span(), buf.n) != 0);
        REQUIRE(engine.submit_limit(Side::Ask, 1002, 10, buf.span(), buf.n) != 0);

        std::array<Fill, 1> one{};
        std::size_t         n  = 0;
        const OrderId       id = engine.submit_limit(Side::Bid, 1002, 25, one, n);
        REQUIRE(id != 0);
        CHECK(n == 1);
        CHECK(one[0].price == 1001);
        CHECK(one[0].qty == 10);
        CHECK_FALSE(engine.book().find(id).has_value());  // 残り 15 は 1002 と交差するので捨てる（id ≠ 0 でも載らない）
        CHECK_FALSE(engine.book().best_bid().has_value());
        CHECK(*engine.book().best_ask() == 1002);
        CHECK(engine.discarded_qty(Side::Bid) == 15);
        CHECK(engine.orders_truncated() == 1);
        CHECK(engine.book().check_invariants());

        // バッファが足りれば全部約定して残りが載る
        std::array<Fill, 2> two{};
        n                      = 0;
        const OrderId id2      = engine.submit_limit(Side::Bid, 1002, 15, two, n);
        REQUIRE(id2 != 0);
        CHECK(n == 1);
        CHECK(engine.book().find(id2)->qty == 5);
        CHECK(*engine.book().best_bid() == 1002);
        CHECK_FALSE(engine.book().best_ask().has_value());
        CHECK(engine.orders_truncated() == 1);
        CHECK(engine.book().check_invariants());

        // 成行も同じ: バッファ 1 件で 2 レベル分の流動性があれば打ち切り
        REQUIRE(engine.submit_limit(Side::Ask, 1003, 10, two, n) != 0);
        REQUIRE(engine.submit_limit(Side::Ask, 1004, 10, two, n) != 0);
        n = 0;
        CHECK(engine.submit_market(Side::Bid, 15, one, n) == 10);
        CHECK(n == 1);
        CHECK(engine.orders_truncated() == 2);
        CHECK(*engine.book().best_ask() == 1004);
    }
    SECTION("OrderBook::add itself rejects a crossing price (unreachable through the engine)") {
        SmallBook book(kBase);
        REQUIRE(book.add(Side::Ask, 1005, 10) != 0);
        CHECK(book.add(Side::Bid, 1005, 1) == 0);  // 等しい価格も交差
        CHECK(book.add(Side::Bid, 1006, 1) == 0);
        CHECK(book.add(Side::Bid, 1004, 1) != 0);
        CHECK(book.add(Side::Ask, 1004, 1) == 0);
        CHECK(book.add(Side::Ask, 1003, 1) == 0);
        CHECK(book.add(Side::Ask, 1005, 1) != 0);
        CHECK(book.add(Side::Bid, 1004, 0) == 0);       // qty 0
        CHECK(book.add(Side::Bid, kBase - 1, 1) == 0);  // 範囲外
        CHECK(book.order_count() == 3);
        CHECK(*book.best_bid() == 1004);
        CHECK(*book.best_ask() == 1005);
        CHECK(book.check_invariants());
    }
    SECTION("random flow with a tiny fills buffer never crosses") {
        FlowEngine           engine(kFlowBase);
        Rng                  rng(9973);
        std::vector<OrderId> live;
        std::array<Fill, 2>  tiny{};
        int                  crossed = 0, discards = 0;
        for (int i = 0; i < 20'000; ++i) {
            std::size_t  n    = 0;
            const double u    = rng.uniform();
            const Side   side = rng.uniform() < 0.5 ? Side::Bid : Side::Ask;
            const Qty    qty  = 1 + static_cast<Qty>(rng.uniform() * 100.0);
            const Price  px   = kFlowAnchor + static_cast<Price>(rng.uniform() * 41.0) - 20;
            const Qty    d0   = engine.discarded_qty(side);
            if (u < 0.7) {
                const OrderId id = engine.submit_limit(side, px, qty, tiny, n);
                if (id != 0) live.push_back(id);
            } else if (u < 0.8) {
                engine.submit_market(side, qty, tiny, n);
            } else if (!live.empty()) {
                const auto k = static_cast<std::size_t>(rng.uniform() * static_cast<double>(live.size()));
                engine.cancel(live[k]);
                live[k] = live.back();
                live.pop_back();
            }
            if (engine.discarded_qty(side) != d0) ++discards;
            if (!uncrossed(engine)) ++crossed;
        }
        CHECK(crossed == 0);
        CHECK(discards > 0);
        CHECK(engine.orders_truncated() > 0);
        CHECK(engine.orders_truncated() <= static_cast<std::uint64_t>(discards));
        CHECK(engine.book().check_invariants());
    }
}

TEST_CASE("LOB-12: the same seeded flow produces bit-identical fill sequences on two engines",
          "[lob][determinism]") {
    auto run = [](std::vector<Fill>& fills, std::vector<OrderId>& ids, std::vector<LevelView>& depth_out) {
        FlowEngine           engine(kFlowBase);
        Rng                  rng(31415);
        std::vector<OrderId> live;
        for (int i = 0; i < 100'000; ++i) {
            const FlowStep s = flow_step(engine, rng, live, &fills);
            ids.push_back(s.id);
        }
        depth_out.resize(64);
        const std::size_t n = engine.book().depth(Side::Bid, depth_out);
        std::vector<LevelView> asks(64);
        const std::size_t      m = engine.book().depth(Side::Ask, asks);
        depth_out.resize(n);
        depth_out.insert(depth_out.end(), asks.begin(), asks.begin() + static_cast<std::ptrdiff_t>(m));
        return engine.next_seq();
    };
    std::vector<Fill>      fa, fb;
    std::vector<OrderId>   ia, ib;
    std::vector<LevelView> da, db;
    const std::uint64_t    sa = run(fa, ia, da);
    const std::uint64_t    sb = run(fb, ib, db);

    CHECK(sa == sb);
    REQUIRE(fa.size() == fb.size());
    REQUIRE(fa.size() > 10'000);
    std::size_t mismatched = 0;
    for (std::size_t k = 0; k < fa.size(); ++k) {
        const bool same = fa[k].maker == fb[k].maker && fa[k].taker == fb[k].taker &&
                          fa[k].taker_side == fb[k].taker_side && fa[k].price == fb[k].price &&
                          fa[k].qty == fb[k].qty && fa[k].seq == fb[k].seq;
        if (!same) ++mismatched;
    }
    CHECK(mismatched == 0);
    CHECK(ia == ib);
    REQUIRE(da.size() == db.size());
    for (std::size_t k = 0; k < da.size(); ++k) {
        CHECK(da[k].price == db[k].price);
        CHECK(da[k].qty == db[k].qty);
        CHECK(da[k].orders == db[k].orders);
    }
}

TEST_CASE("LOB-13: depth(N) returns the top N levels as (price, qty, orders)", "[lob][unit]") {
    SmallEngine engine(kBase);
    FillBuf     buf;
    // bid: 1005 x2 (10+20), 1003 x1 (5), 1000 x3 (1+1+1) / ask: 1010 (7), 1020 (8)
    REQUIRE(engine.submit_limit(Side::Bid, 1005, 10, buf.span(), buf.n) != 0);
    REQUIRE(engine.submit_limit(Side::Bid, 1003, 5, buf.span(), buf.n) != 0);
    REQUIRE(engine.submit_limit(Side::Bid, 1005, 20, buf.span(), buf.n) != 0);
    for (int i = 0; i < 3; ++i) REQUIRE(engine.submit_limit(Side::Bid, 1000, 1, buf.span(), buf.n) != 0);
    REQUIRE(engine.submit_limit(Side::Ask, 1020, 8, buf.span(), buf.n) != 0);
    REQUIRE(engine.submit_limit(Side::Ask, 1010, 7, buf.span(), buf.n) != 0);
    REQUIRE(buf.n == 0);

    std::array<LevelView, 2> top2{};
    REQUIRE(engine.book().depth(Side::Bid, top2) == 2);
    CHECK(top2[0].price == 1005);
    CHECK(top2[0].qty == 30);
    CHECK(top2[0].orders == 2);
    CHECK(top2[1].price == 1003);
    CHECK(top2[1].qty == 5);
    CHECK(top2[1].orders == 1);

    std::array<LevelView, 8> top8{};
    REQUIRE(engine.book().depth(Side::Bid, top8) == 3);  // 空レベルは飛ばし、あるだけ返す
    CHECK(top8[2].price == 1000);
    CHECK(top8[2].qty == 3);
    CHECK(top8[2].orders == 3);

    REQUIRE(engine.book().depth(Side::Ask, top8) == 2);
    CHECK(top8[0].price == 1010);
    CHECK(top8[0].qty == 7);
    CHECK(top8[0].orders == 1);
    CHECK(top8[1].price == 1020);
    CHECK(top8[1].qty == 8);

    std::span<LevelView> none;
    CHECK(engine.book().depth(Side::Bid, none) == 0);
    SmallBook empty(kBase);
    CHECK(empty.depth(Side::Ask, top8) == 0);
    CHECK(engine.book().check_invariants());
}

TEST_CASE("LOB-14: the order pool does not allocate after warm-up and rejects when exhausted",
          "[lob][unit]") {
    SmallEngine engine(kBase);  // ウォームアップ = コンストラクタ（プールとレベル配列の確保）
    SmallEngine fresh(kBase);   // reset 後の id 比較用（SECTION 内で作るとその確保を数えてしまう）
    FillBuf     buf;
    const std::size_t allocs_after_ctor = g_alloc_count;

    // 範囲外の価格と qty 0 は拒否（プールは無傷）
    CHECK(engine.submit_limit(Side::Bid, kBase - 1, 1, buf.span(), buf.n) == 0);
    CHECK(engine.submit_limit(Side::Bid, kBase + 32, 1, buf.span(), buf.n) == 0);
    CHECK(engine.submit_limit(Side::Bid, 1000, 0, buf.span(), buf.n) == 0);
    CHECK(engine.orders_rejected() == 3);
    CHECK(engine.book().pool_free() == 64);

    for (int i = 0; i < 64; ++i) {
        REQUIRE(engine.submit_limit(Side::Bid, 1000 + (i % 8), 5, buf.span(), buf.n) != 0);
    }
    CHECK(engine.book().pool_free() == 0);
    CHECK(engine.book().order_count() == 64);

    // 枯渇: 指値はどちら側でも、交差していても、拒否（0）。板は変わらない
    CHECK(engine.submit_limit(Side::Bid, 1001, 5, buf.span(), buf.n) == 0);
    CHECK(engine.submit_limit(Side::Ask, 1020, 5, buf.span(), buf.n) == 0);
    CHECK(engine.submit_limit(Side::Ask, 1000, 5, buf.span(), buf.n) == 0);
    CHECK(buf.n == 0);
    CHECK(engine.book().order_count() == 64);
    CHECK(engine.orders_rejected() == 6);
    CHECK(engine.submitted_qty(Side::Ask) == 0);

    // 成行はスロット不要なので枯渇中でも約定し、プールを空ける
    CHECK(engine.submit_market(Side::Ask, 5, buf.span(), buf.n) == 5);
    CHECK(buf.n == 1);
    CHECK(engine.book().pool_free() == 1);
    CHECK(engine.submit_limit(Side::Ask, 1020, 5, buf.span(), buf.n) != 0);
    CHECK(engine.book().pool_free() == 0);

    // 全約定済みのメイカー id は取り消せない
    CHECK_FALSE(engine.cancel(buf.fills[0].maker));
    CHECK(engine.book().pool_free() == 0);

    // 取消で 1 つ空けば次の指値は通る（最後に載せた ask を取り消す）
    const OrderId last_ask = engine.submit_limit(Side::Ask, 1021, 5, buf.span(), buf.n);
    CHECK(last_ask == 0);  // まだ満杯
    std::array<LevelView, 1> top{};
    REQUIRE(engine.book().depth(Side::Ask, top) == 1);
    CHECK(top[0].price == 1020);
    const auto resting_ask = engine.book().front(Side::Ask);
    REQUIRE(resting_ask.has_value());
    CHECK(engine.cancel(resting_ask->id));
    CHECK(engine.book().pool_free() == 1);
    CHECK(engine.submit_limit(Side::Bid, 1000, 5, buf.span(), buf.n) != 0);
    CHECK(engine.book().pool_free() == 0);
    CHECK(engine.book().check_invariants());

    CHECK(g_alloc_count == allocs_after_ctor);  // ウォームアップ後はアロケーションなし

    SECTION("reset empties the book and statistics without allocating, and restarts the id sequence") {
        REQUIRE(engine.fills() > 0);
        REQUIRE(engine.orders_rejected() > 0);
        REQUIRE(engine.book().order_count() == 64);
        // SECTION に入る際 Catch2 自身が確保する（名前文字列・トラッカ）ので、基準はここで取り直す
        const std::size_t allocs_before_reset = g_alloc_count;
        engine.reset();
        CHECK(engine.book().order_count() == 0);
        CHECK(engine.book().pool_free() == 64);
        CHECK_FALSE(engine.book().best_bid().has_value());
        CHECK_FALSE(engine.book().best_ask().has_value());
        CHECK(engine.next_seq() == 1);
        CHECK(engine.fills() == 0);
        for (Side s : {Side::Bid, Side::Ask}) {
            CHECK(engine.submitted_qty(s) == 0);
            CHECK(engine.filled_qty(s) == 0);
            CHECK(engine.cancelled_qty(s) == 0);
            CHECK(engine.discarded_qty(s) == 0);
            CHECK(engine.book().total_qty(s) == 0);
            CHECK(engine.book().order_count(s) == 0);
        }
        CHECK(engine.orders_submitted() == 0);
        CHECK(engine.orders_cancelled() == 0);
        CHECK(engine.orders_rejected() == 0);
        CHECK(engine.orders_truncated() == 0);
        CHECK(engine.book().check_invariants());

        // id は clear() 間でのみ一意: リセット後の最初の id は新品のエンジンの最初の id と等しい
        //（同 seed 再生の bit 一致のため。呼び手は live-id 集合を reset と一緒に捨てる）
        buf.n              = 0;
        const OrderId a    = engine.submit_limit(Side::Bid, 1000, 1, buf.span(), buf.n);
        const OrderId b    = fresh.submit_limit(Side::Bid, 1000, 1, buf.span(), buf.n);
        CHECK(a != 0);
        CHECK(a == b);
        CHECK(engine.book().order_count() == 1);
        CHECK(g_alloc_count == allocs_before_reset);  // reset もその後の注文もアロケーションしない
    }
}
