#pragma once
// core/micro/order_book.hpp — 固定容量の指値板（価格レベル配列 + intrusive 双方向リスト + 注文プール）。
// マッチングの規則は matching_engine.hpp が持ち、ここはデータ構造・プール・不変条件だけを担う。
//
// メモリ配置（ヒープはコンストラクタでのみ確保、以後 step 経路でアロケーションなし）:
//   * 注文プール pool_[MaxOrders]（40 B / 注文）: 空きスロットは next で単方向の free list に繋ぐ
//   * 価格レベル levels_[side][MaxLevels]（24 B / レベル）: 価格 p はレベル添字 p − base_price に写す。
//     範囲 [base_price, base_price + MaxLevels) の外は拒否（add が 0 を返す）
//   * 各レベルは head/tail（スロット添字）+ 合計数量 + 件数を持ち、注文はレベル内で FIFO の intrusive
//     双方向リスト（prev/next は 32 bit 添字）。末尾に追加し、先頭から約定する = 価格時間優先
//   * オブジェクト本体は ~128 B（vector ヘッダとカーソル）。OrderBook<65536, 4096> のヒープは
//     65536 × 40 B + 2 × 4096 × 24 B ≈ 2.8 MB
//
// OrderId（64 bit、0 は無効）= (通番 << kSlotBits) | スロット添字。通番は add / take_id ごとに 1 増える
// （単調増加）ので、**clear() と clear() の間では** id は一意で再利用されず、cancel(id) はスロットに保存
// された id と比較するだけで「既に消えた id」を O(1) で拒否できる（false、板は変えない）。clear()
// （= MatchingEngine::reset()）は通番を 1 に戻すので、同 seed の再生が bit 一致する代わりに、リセット前の
// id がリセット後の注文と衝突しうる。呼び手（シーン）は自分の live-id 集合を **reset と同じ操作の中で**
// 捨てること。板に載らないテイカー（成行・全約定した指値）は take_id() でスロット添字 0 の id を受け取る。
// 通番は 2^(64 − kSlotBits) 回で溢れる（MaxOrders = 65536 なら 2^48 ≈ 2.8e14 注文。1e6 注文/秒で約 9 年）。
//
// best bid / ask はレベル添字のカーソルで持つ。追加は O(1)（比較 1 回）。best レベルが空になった
// ときだけ次の非空レベルへ走査する: 最悪 O(連続する空レベル数) ≤ MaxLevels、側が空になれば
// count_ を見て O(1) で kNone にする。depth() も同じ走査（側の全注文を数え終えたら止まる）。
// check_invariants(full): bid < ask、各レベルの qty / count とリストの整合（prev/next 対称、tail、
// 循環なし）、生存注文の id とスロットの対応、側ごとの合計、best カーソルが本当に最良の非空レベルか、
// free_count_ + 生存数 == MaxOrders を見る。full = true（既定）はさらに free list を端まで歩いて長さと
// 各スロットの空きを確かめる — これは占有率に関係なく O(MaxOrders)（65536 で ≈ 150 µs, Debug）なので、
// 毎ステップ呼ぶ側は full = false（O(MaxLevels + 生存数)）を使う。テストは full = true。Debug では
// MatchingEngine が check_every 回の公開操作ごとに full = false で assert する。
//
// 板の不変条件「bid < ask」は add() 自身が守る: 反対側の best と交差する価格は 0 を返す
// （MatchingEngine は交差分を先に約定させるので、この経路には来ない）。

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace quantviz::core::micro {

using Price   = std::int64_t;   ///< ティック整数（外部の double 価格は tick_size で変換する）
using Qty     = std::uint64_t;  ///< 数量
using OrderId = std::uint64_t;  ///< 0 は無効

enum class Side : std::uint8_t { Bid = 0, Ask = 1 };
enum class OrderType : std::uint8_t { Limit, Market };  ///< シーンの Command 配線用に予約（板は使わない）

[[nodiscard]] constexpr Side opposite(Side s) noexcept { return s == Side::Bid ? Side::Ask : Side::Bid; }

/// 約定 1 件。price はメイカー（板に載っていた側）の価格。seq は 1 から始まる通番。
struct Fill {
    OrderId       maker;
    OrderId       taker;
    Side          taker_side;
    Price         price;
    Qty           qty;
    std::uint64_t seq;
};

/// depth() が返す 1 レベル。
struct LevelView {
    Price         price;
    Qty           qty;
    std::uint32_t orders;
};

/// find() / front() / remove() が返す注文の要約。
struct OrderView {
    OrderId id;
    Side    side;
    Price   price;
    Qty     qty;
};

template <std::size_t MaxOrders, std::size_t MaxLevels>
class OrderBook {
    static_assert(MaxOrders >= 2 && MaxOrders <= (std::size_t{1} << 31),
                  "OrderBook: MaxOrders must be in [2, 2^31]");
    static_assert(MaxLevels >= 1 && MaxLevels <= (std::size_t{1} << 31),
                  "OrderBook: MaxLevels must be in [1, 2^31]");

public:
    static constexpr std::size_t kMaxOrders = MaxOrders;
    static constexpr std::size_t kMaxLevels = MaxLevels;
    /// id の下位ビット数（スロット添字を格納する）。
    static constexpr unsigned kSlotBits = static_cast<unsigned>(std::bit_width(MaxOrders - 1));
    static constexpr OrderId  kSlotMask = (OrderId{1} << kSlotBits) - 1;

    /// 価格 [base_price, base_price + MaxLevels) を受け付ける板。ヒープ確保はここだけ。
    explicit OrderBook(Price base_price)
        : base_(base_price),
          pool_(MaxOrders),
          levels_{std::vector<Level>(MaxLevels), std::vector<Level>(MaxLevels)} {
        clear();
    }

    /// 空の板に戻す（id 通番も 1 に戻る = 構築直後と同じ状態）。O(MaxOrders + MaxLevels)。
    void clear() noexcept {
        Order* pool = pool_.data();
        for (std::size_t i = 0; i < MaxOrders; ++i) {
            pool[i].id    = 0;
            pool[i].qty   = 0;
            pool[i].price = 0;
            pool[i].prev  = kNone;
            pool[i].next  = (i + 1 < MaxOrders) ? static_cast<std::uint32_t>(i + 1) : kNone;
            pool[i].side  = Side::Bid;
        }
        free_head_  = 0;
        free_count_ = static_cast<std::uint32_t>(MaxOrders);
        for (auto& lv : levels_)
            for (auto& l : lv) l = Level{kNone, kNone, 0, 0};
        best_      = {kNone, kNone};
        total_qty_ = {0, 0};
        count_     = {0, 0};
        counter_   = 1;
    }

    // ---- 照会 --------------------------------------------------------------------------------

    [[nodiscard]] Price base_price() const noexcept { return base_; }
    [[nodiscard]] bool  in_range(Price p) const noexcept {
        // p ≥ base のとき p − base は [0, 2^64) に収まるので符号なし減算で正確
        return p >= base_ && (static_cast<std::uint64_t>(p) - static_cast<std::uint64_t>(base_)) < MaxLevels;
    }

    [[nodiscard]] std::optional<Price> best(Side s) const noexcept {
        const std::uint32_t li = best_[idx(s)];
        if (li == kNone) return std::nullopt;
        return price_at(li);
    }
    [[nodiscard]] std::optional<Price> best_bid() const noexcept { return best(Side::Bid); }
    [[nodiscard]] std::optional<Price> best_ask() const noexcept { return best(Side::Ask); }

    /// レベルの合計数量 / 件数。範囲外の価格は 0。
    [[nodiscard]] Qty level_qty(Side s, Price p) const noexcept {
        return in_range(p) ? levels_[idx(s)][lidx(p)].qty : 0;
    }
    [[nodiscard]] std::uint32_t level_orders(Side s, Price p) const noexcept {
        return in_range(p) ? levels_[idx(s)][lidx(p)].count : 0;
    }

    /// best から out.size() レベルまで（空レベルは飛ばす）。bid は価格降順、ask は昇順。書いた数を返す。
    /// 側の全注文を数え終えた時点で止まるので、走査は「非空レベルまでの空レベル」ぶんだけ。
    [[nodiscard]] std::size_t depth(Side s, std::span<LevelView> out) const noexcept {
        const std::size_t si = idx(s);
        if (best_[si] == kNone || out.empty()) return 0;
        const Level*       lv        = levels_[si].data();
        std::uint32_t      remaining = count_[si];
        std::size_t        n         = 0;
        const std::int64_t step      = (s == Side::Bid) ? -1 : 1;
        const std::int64_t end       = (s == Side::Bid) ? -1 : static_cast<std::int64_t>(MaxLevels);
        std::int64_t i = static_cast<std::int64_t>(best_[si]);
        for (; i != end && n < out.size() && remaining != 0; i += step) {
            const Level& l = lv[static_cast<std::size_t>(i)];
            if (l.count == 0) continue;
            out[n++] = LevelView{price_at(static_cast<std::uint32_t>(i)), l.qty, l.count};
            remaining -= l.count;
        }
        return n;
    }

    [[nodiscard]] Qty         total_qty(Side s) const noexcept { return total_qty_[idx(s)]; }
    [[nodiscard]] std::size_t order_count() const noexcept { return count_[0] + count_[1]; }
    [[nodiscard]] std::size_t order_count(Side s) const noexcept { return count_[idx(s)]; }
    [[nodiscard]] std::size_t pool_free() const noexcept { return free_count_; }

    /// id の注文（板に載っているもの）。無ければ nullopt。O(1)。
    [[nodiscard]] std::optional<OrderView> find(OrderId id) const noexcept {
        const Order* o = lookup(id);
        if (o == nullptr) return std::nullopt;
        return OrderView{o->id, o->side, o->price, o->qty};
    }

    /// 側の best レベルの先頭（最も古い）注文。側が空なら nullopt。
    [[nodiscard]] std::optional<OrderView> front(Side s) const noexcept {
        const std::size_t si = idx(s);
        if (best_[si] == kNone) return std::nullopt;
        const Order& o = pool_[levels_[si][best_[si]].head];
        return OrderView{o.id, o.side, o.price, o.qty};
    }

    // ---- 変更（MatchingEngine が使う低レベル操作） ------------------------------------------

    /// 注文をレベル末尾に載せ、id を返す。qty 0 / 範囲外 / プール空 / 反対側の best と交差 → 0。O(1)。
    OrderId add(Side s, Price p, Qty q) noexcept {
        if (q == 0 || !in_range(p) || free_head_ == kNone || crosses(s, p)) return 0;
        const std::size_t   si   = idx(s);
        const std::uint32_t li   = lidx(p);
        const std::uint32_t slot = free_head_;
        Order&              o    = pool_[slot];
        free_head_               = o.next;
        --free_count_;

        const OrderId id = make_id(counter_++, slot);
        Level&        l  = levels_[si][li];
        o                = Order{id, q, p, l.tail, kNone, s};
        if (l.tail == kNone) l.head = slot;
        else pool_[l.tail].next = slot;
        l.tail = slot;
        ++l.count;
        l.qty += q;
        total_qty_[si] += q;
        ++count_[si];
        if (best_[si] == kNone || (s == Side::Bid ? li > best_[si] : li < best_[si])) best_[si] = li;
        return id;
    }

    /// スロットを消費せず id だけ払い出す（板に載らないテイカー用。スロット添字は 0）。
    [[nodiscard]] OrderId take_id() noexcept { return make_id(counter_++, 0); }

    /// 取消: 板から外してスロットを解放し、外した注文を返す。不明・消えた id は nullopt で板は変えない。
    std::optional<OrderView> remove(OrderId id) noexcept {
        const Order* o = lookup(id);
        if (o == nullptr) return std::nullopt;
        const OrderView v{o->id, o->side, o->price, o->qty};
        unlink_and_free(slot_of(id));
        return v;
    }

    /// 側の best レベルの先頭注文から最大 q を削る（0 になれば注文を外す）。削った量を返し、maker に
    /// その注文の id を書く。側が空なら 0（maker は変えない）。
    Qty reduce_front(Side s, Qty q, OrderId& maker) noexcept {
        const std::size_t si = idx(s);
        if (best_[si] == kNone) return 0;
        Level&              l    = levels_[si][best_[si]];
        const std::uint32_t slot = l.head;
        Order&              o    = pool_[slot];
        const Qty           take = q < o.qty ? q : o.qty;
        maker                    = o.id;
        o.qty -= take;
        l.qty -= take;
        total_qty_[si] -= take;
        if (o.qty == 0) unlink_and_free(slot);
        return take;
    }

    // ---- 検査 --------------------------------------------------------------------------------

    /// 不変条件の検査（ファイル先頭のコメント参照）。壊れていれば false。
    /// full = true: free list も端まで歩く（O(MaxOrders + MaxLevels + 生存数)、占有率に依らず MaxOrders に比例）。
    /// full = false: free list の走査を省き free_count_ + 生存数 == MaxOrders だけ見る（O(MaxLevels + 生存数)）。
    [[nodiscard]] bool check_invariants(bool full = true) const noexcept {
        const Order* pool = pool_.data();
        std::size_t  live = 0;
        for (std::size_t si = 0; si < 2; ++si) {
            const Level*  lv        = levels_[si].data();
            Qty           side_qty  = 0;
            std::uint32_t side_cnt  = 0;
            std::uint32_t best_seen = kNone;
            for (std::uint32_t li = 0; li < MaxLevels; ++li) {
                const Level& l = lv[li];
                if (l.count == 0) {
                    if (l.head != kNone || l.tail != kNone || l.qty != 0) return false;
                    continue;
                }
                std::uint32_t prev = kNone, cur = l.head, n = 0;
                Qty           q = 0;
                while (cur != kNone) {
                    if (cur >= MaxOrders || n >= l.count) return false;  // 範囲外 / 循環 / 件数超過
                    const Order& o = pool[cur];
                    if (o.id == 0 || slot_of(o.id) != cur || o.qty == 0 || o.price != price_at(li) ||
                        idx(o.side) != si || o.prev != prev)
                        return false;
                    q += o.qty;
                    ++n;
                    prev = cur;
                    cur  = o.next;
                }
                if (prev != l.tail || n != l.count || q != l.qty) return false;
                side_qty += q;
                side_cnt += n;
                if (si == idx(Side::Bid)) best_seen = li;  // 最後に見た非空 = 最高値
                else if (best_seen == kNone) best_seen = li;  // 最初に見た非空 = 最安値
            }
            if (side_qty != total_qty_[si] || side_cnt != count_[si] || best_seen != best_[si]) return false;
            live += side_cnt;
        }
        if (best_[0] != kNone && best_[1] != kNone && best_[0] >= best_[1]) return false;  // bid < ask
        if (free_count_ + live != MaxOrders) return false;
        if (!full) return true;

        std::size_t nfree = 0;
        for (std::uint32_t cur = free_head_; cur != kNone; cur = pool[cur].next) {
            if (cur >= MaxOrders || nfree >= MaxOrders || pool[cur].id != 0) return false;  // 範囲外 / 循環
            ++nfree;
        }
        return nfree == free_count_;
    }

private:
    struct Order {
        OrderId       id;     ///< 0 = 空きスロット
        Qty           qty;    ///< 残数量
        Price         price;
        std::uint32_t prev;   ///< 同一レベル内リスト（空きスロットでは kNone）
        std::uint32_t next;   ///< 同一レベル内リスト / 空きスロットでは free list の次
        Side          side;
    };
    struct Level {
        std::uint32_t head, tail;  ///< スロット添字（空なら kNone）
        std::uint32_t count;
        Qty           qty;
    };
    static constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();

    static constexpr std::size_t   idx(Side s) noexcept { return static_cast<std::size_t>(s); }
    static constexpr OrderId       make_id(std::uint64_t counter, std::uint32_t slot) noexcept {
        return (counter << kSlotBits) | slot;
    }
    static constexpr std::uint32_t slot_of(OrderId id) noexcept {
        return static_cast<std::uint32_t>(id & kSlotMask);
    }

    [[nodiscard]] std::uint32_t lidx(Price p) const noexcept {
        return static_cast<std::uint32_t>(static_cast<std::uint64_t>(p) - static_cast<std::uint64_t>(base_));
    }
    [[nodiscard]] Price price_at(std::uint32_t li) const noexcept { return base_ + static_cast<Price>(li); }

    [[nodiscard]] bool crosses(Side s, Price p) const noexcept {
        const std::uint32_t ob = best_[idx(opposite(s))];
        if (ob == kNone) return false;
        const Price q = price_at(ob);
        return s == Side::Bid ? p >= q : p <= q;
    }

    [[nodiscard]] const Order* lookup(OrderId id) const noexcept {
        if (id == 0) return nullptr;
        const std::uint32_t slot = slot_of(id);
        if (slot >= MaxOrders) return nullptr;
        const Order& o = pool_[slot];
        return o.id == id ? &o : nullptr;
    }

    /// 板からリンクを外し、スロットを free list に戻す。best レベルが空になればカーソルを進める。
    void unlink_and_free(std::uint32_t slot) noexcept {
        Order&              o  = pool_[slot];
        const std::size_t   si = idx(o.side);
        const std::uint32_t li = lidx(o.price);
        Level&              l  = levels_[si][li];
        if (o.prev != kNone) pool_[o.prev].next = o.next;
        else l.head = o.next;
        if (o.next != kNone) pool_[o.next].prev = o.prev;
        else l.tail = o.prev;
        --l.count;
        l.qty -= o.qty;
        total_qty_[si] -= o.qty;
        --count_[si];

        o.id   = 0;
        o.qty  = 0;
        o.prev = kNone;
        o.next = free_head_;
        free_head_ = slot;
        ++free_count_;

        if (l.count == 0 && best_[si] == li) advance_best(si, li);
    }

    /// best レベル li が空になった: 次の非空レベルへ（bid は下へ、ask は上へ）。側が空なら O(1)。
    void advance_best(std::size_t si, std::uint32_t li) noexcept {
        if (count_[si] == 0) {
            best_[si] = kNone;
            return;
        }
        const Level* lv = levels_[si].data();
        if (si == idx(Side::Bid)) {
            while (li > 0) {
                --li;
                if (lv[li].count != 0) {
                    best_[si] = li;
                    return;
                }
            }
        } else {
            while (++li < MaxLevels) {
                if (lv[li].count != 0) {
                    best_[si] = li;
                    return;
                }
            }
        }
        best_[si] = kNone;  // count_ > 0 なら到達しない（check_invariants が検出する）
    }

    Price                             base_;
    std::vector<Order>                pool_;
    std::array<std::vector<Level>, 2> levels_;  ///< [Bid, Ask]
    std::uint32_t                     free_head_  = kNone;
    std::uint32_t                     free_count_ = 0;
    std::array<std::uint32_t, 2>      best_{kNone, kNone};  ///< best レベル添字（空なら kNone）
    std::array<Qty, 2>                total_qty_{0, 0};
    std::array<std::uint32_t, 2>      count_{0, 0};  ///< 側ごとの生存注文数
    std::uint64_t                     counter_ = 1;  ///< 次の id 通番
};

}  // namespace quantviz::core::micro
