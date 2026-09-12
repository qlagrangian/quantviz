#pragma once
// core/micro/matching_engine.hpp — 価格時間優先のマッチング。OrderBook を包み、指値・成行・取消と
// 数量保存の統計を提供する。ホットパスはアロケーション・例外・mutex なし。
//
// 規則:
//   * 指値 submit_limit: 反対側の best から順に、指値と交差する間だけ約定（複数レベルを掃く）。
//     約定価格はメイカーの価格。残りは自分の側のレベル末尾に載る（同価格は先着順）。
//     返り値は付与した id（拒否なら 0）。プールが空なら約定させる前に丸ごと拒否する
//     （= 拒否は副作用なし。成行は取消と同様スロットを使わないので枯渇中も動き、プールを空ける）。
//   * 成行 submit_market: 反対側を数量分だけ掃く。板が尽きたら残りは捨てる（discarded_qty に計上）。
//     返り値は約定数量。
//   * fills は呼び手のバッファに **追記** する: n_fills の位置から書き、n_fills を進める（1 ステップ内の
//     複数注文の約定を 1 つのバッファに集められる）。バッファが満杯なのにまだ約定できる（交差している）
//     なら、その注文の残りは捨てて discarded_qty に計上し、板には載せない — 交差する注文を板に残すと
//     bid ≥ ask になるため（LOB-11）。約定は 1 件も失わない。この「打ち切り」は指値・成行とも
//     orders_truncated() で件数が分かる。LOB シーンは ≥ 64 件の span を渡す。
//   * **id ≠ 0 は「板に載った」を意味しない**: 全約定した指値や打ち切られた指値も id を返す（約定の
//     taker として使う）。載ったかどうかは book().find(id) で見る。
//   * cancel(id): 板から外す。不明・既に消えた id は false。
//   * 数量保存（LOB-08）: 側ごとに
//       submitted_qty == filled_qty + cancelled_qty + discarded_qty + book().total_qty
//     が常に成り立つ。filled_qty は各約定を買い側・売り側の両方に足すので filled_qty(Bid) == filled_qty(Ask)。
//     submitted_qty は受理した注文の全数量（拒否した指値は数えない）。
//   * 約定通番 seq は 1 から 1 ずつ増える（0 は「約定なし」に使える）。next_seq() は次に振る値。
//   * 同 seed のフローで約定列が bit 一致する（LOB-12）: 乱数はここにはなく、処理は完全に決定的。
//   * reset(): 板・統計・約定通番・id 通番を構築直後に戻す。id は clear() 間でしか一意でない
//     （order_book.hpp 参照）ので、呼び手は自分の live-id 集合を reset と同じ操作で捨てること。
//
// Debug（NDEBUG 未定義）では check_every 回（既定 1 = 毎回）の公開操作ごとに
// assert(book().check_invariants(false)) を実行する（free list の走査は省く）。set_check_every(0) で
// 止められるので、LOB シーンは止めて 1 ステップに 1 回 check_invariants() を自分で呼ぶ。
// Release では何もしない。

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "quantviz/core/micro/order_book.hpp"

namespace quantviz::core::micro {

template <std::size_t MaxOrders, std::size_t MaxLevels>
class MatchingEngine {
public:
    using Book = OrderBook<MaxOrders, MaxLevels>;

    explicit MatchingEngine(Price base_price) : book_(base_price) {}

    /// 指値。交差分を約定させ（fills に追記）、残りを板に載せる。返り値は id（拒否なら 0）。
    OrderId submit_limit(Side side, Price price, Qty qty, std::span<Fill> fills,
                         std::size_t& n_fills) noexcept {
        if (qty == 0 || !book_.in_range(price) || book_.pool_free() == 0) {
            ++orders_rejected_;
            return 0;
        }
        const std::size_t si = idx(side);
        ++orders_submitted_;
        submitted_qty_[si] += qty;

        const std::size_t first     = n_fills;
        Qty               remaining = qty;
        const bool        truncated = sweep(side, price, true, remaining, fills, n_fills);

        OrderId id = 0;
        if (remaining != 0 && !truncated) {
            id = book_.add(side, price, remaining);
            assert(id != 0);  // sweep 後は交差せず、スロットも確保済みなので失敗しない
            if (id == 0) {    // Release の保険: 残りは捨てて保存則を守る
                discarded_qty_[si] += remaining;
                id = book_.take_id();
            }
        } else {
            if (truncated) {
                discarded_qty_[si] += remaining;
                ++orders_truncated_;
            }
            id = book_.take_id();
        }
        for (std::size_t k = first; k < n_fills; ++k) fills[k].taker = id;
        debug_check();
        return id;
    }

    /// 成行。反対側を qty 分だけ掃き、板が尽きたら残りは捨てる。返り値は約定数量。
    Qty submit_market(Side side, Qty qty, std::span<Fill> fills, std::size_t& n_fills) noexcept {
        if (qty == 0) {
            ++orders_rejected_;
            return 0;
        }
        const std::size_t si = idx(side);
        ++orders_submitted_;
        submitted_qty_[si] += qty;

        const std::size_t first     = n_fills;
        Qty               remaining = qty;
        if (sweep(side, 0, false, remaining, fills, n_fills)) ++orders_truncated_;
        discarded_qty_[si] += remaining;

        const OrderId id = book_.take_id();
        for (std::size_t k = first; k < n_fills; ++k) fills[k].taker = id;
        debug_check();
        return qty - remaining;
    }

    /// 取消。板に無ければ false。
    bool cancel(OrderId id) noexcept {
        const auto removed = book_.remove(id);
        if (!removed) return false;
        cancelled_qty_[idx(removed->side)] += removed->qty;
        ++orders_cancelled_;
        debug_check();
        return true;
    }

    /// 板を空にし、統計・約定通番・id 通番を構築直後に戻す（Reset 用。check_every は保つ）。
    /// 呼び手の live-id 集合はこの操作と一緒に捨てること（id は clear() 間でしか一意でない）。
    void reset() noexcept {
        book_.clear();
        seq_              = 1;
        orders_submitted_ = orders_cancelled_ = orders_rejected_ = orders_truncated_ = 0;
        submitted_qty_ = filled_qty_ = cancelled_qty_ = discarded_qty_ = {0, 0};
        check_countdown_ = check_every_;
        debug_check();
    }

    /// Debug の不変条件検査の頻度: n 回の公開操作ごとに 1 回（既定 1 = 毎回）。0 で止める（呼び手が
    /// 自分で book().check_invariants() を呼ぶ）。Release では効果なし。
    void set_check_every(std::uint32_t n) noexcept {
        check_every_     = n;
        check_countdown_ = n;
    }
    [[nodiscard]] std::uint32_t check_every() const noexcept { return check_every_; }

    [[nodiscard]] const Book&    book() const noexcept { return book_; }
    [[nodiscard]] std::uint64_t  next_seq() const noexcept { return seq_; }
    [[nodiscard]] std::uint64_t  fills() const noexcept { return seq_ - 1; }

    // ---- 統計（数量保存） --------------------------------------------------------------------
    [[nodiscard]] Qty submitted_qty(Side s) const noexcept { return submitted_qty_[idx(s)]; }
    [[nodiscard]] Qty filled_qty(Side s) const noexcept { return filled_qty_[idx(s)]; }
    [[nodiscard]] Qty cancelled_qty(Side s) const noexcept { return cancelled_qty_[idx(s)]; }
    [[nodiscard]] Qty discarded_qty(Side s) const noexcept { return discarded_qty_[idx(s)]; }
    [[nodiscard]] std::uint64_t orders_submitted() const noexcept { return orders_submitted_; }
    [[nodiscard]] std::uint64_t orders_cancelled() const noexcept { return orders_cancelled_; }
    /// 受理前に拒否した件数（qty 0 / 価格範囲外 / プール枯渇）。
    [[nodiscard]] std::uint64_t orders_rejected() const noexcept { return orders_rejected_; }
    /// fills バッファ満杯で残りを捨てた件数（指値・成行とも）。
    [[nodiscard]] std::uint64_t orders_truncated() const noexcept { return orders_truncated_; }

private:
    static constexpr std::size_t idx(Side s) noexcept { return static_cast<std::size_t>(s); }

    /// side の注文（残 remaining）を反対側の best から順に約定させる。is_limit なら price と交差する
    /// 間だけ。fills が満杯なのにまだ約定できるなら true（呼び手は残りを捨てる）。taker は呼び手が
    /// 後から埋める（id は板に載るかどうかで決まるため）。
    bool sweep(Side side, Price price, bool is_limit, Qty& remaining, std::span<Fill> fills,
               std::size_t& n_fills) noexcept {
        const Side opp = opposite(side);
        while (remaining != 0) {
            const auto best = book_.best(opp);
            if (!best) break;
            if (is_limit && (side == Side::Bid ? *best > price : *best < price)) break;
            if (n_fills >= fills.size()) return true;
            OrderId   maker = 0;
            const Qty take  = book_.reduce_front(opp, remaining, maker);
            fills[n_fills++] = Fill{maker, 0, side, *best, take, seq_++};
            remaining -= take;
            filled_qty_[0] += take;
            filled_qty_[1] += take;
        }
        return false;
    }

    /// Debug: check_every_ 回に 1 回、free list の走査を省いた不変条件検査を assert する。
    void debug_check() noexcept {
#ifndef NDEBUG
        if (check_every_ == 0 || --check_countdown_ != 0) return;
        check_countdown_ = check_every_;
        assert(book_.check_invariants(false));
#endif
    }

    Book               book_;
    std::uint64_t      seq_              = 1;
    std::uint64_t      orders_submitted_ = 0;
    std::uint64_t      orders_cancelled_ = 0;
    std::uint64_t      orders_rejected_  = 0;
    std::uint64_t      orders_truncated_ = 0;
    std::uint32_t      check_every_      = 1;
    std::uint32_t      check_countdown_  = 1;
    std::array<Qty, 2> submitted_qty_{0, 0};
    std::array<Qty, 2> filled_qty_{0, 0};
    std::array<Qty, 2> cancelled_qty_{0, 0};
    std::array<Qty, 2> discarded_qty_{0, 0};
};

}  // namespace quantviz::core::micro
