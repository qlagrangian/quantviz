#pragma once
// bridge/triple_buffer.hpp — 単一生産者・単一消費者の「最新 1 枚」交換（グリッド大の状態向け）
//
// リング（SpscRing）は「publish した値を全て順に」渡すための道具。サーフェスのように 1 枚が大きく
// 履歴が要らない状態では、「書き手は常に最新を置き、読み手は常に最新だけを取る」方が正しい。
//
// 設計上の約束:
//   * T は trivially copyable（Surface は POD）→ コピーは memcpy 相当で完結、例外なし
//   * スロットは 3 枚。書き手の back / 公開済みの middle / 読み手の front は常に相異なる添字
//     → 書き手と読み手が同じスロットに触ることが無い。よって読み手は裂けた値を観測しない
//   * 3 つの添字と「未読」ビットを 1 語の std::atomic<std::uint32_t> に詰め、CAS で入れ替える
//   * 書き手はブロックしない。publish の CAS が失敗するのは読み手が同時に swap した時だけで、
//     読み手の進行を待つループではない（読み手が止まっていれば CAS は 1 回で成功する）
//   * 「満杯」が無いので落とした数も数えない。読まれなかった面は黙って捨てられる（それが仕様）
//   * スロットは 3 枚を使い回す。back() の中身は「2 回前に publish した面」であって、
//     ゼロ初期化でも前回書いた内容でもない（back() のコメント参照）
//
// ABA について:
//   * 書き手の CAS は ABA を見ない。状態語が元の値に戻るには未読ビットが立ち直す必要があり、
//     それを立てるのは publish だけ＝単一の書き手自身なので、CAS を挟んだ間には起こり得ない
//   * 読み手の CAS は ABA を見うる（書き手が 2 回 publish すると添字の組が一周する）が、無害。
//     状態語が状態の全てなので、「今の middle を front に引き取る」という遷移の意味は変わらない
//
// メモリ順序の要点（どちらの CAS も acq_rel）:
//   * 書き手 → 読み手: publish() の release により、直前まで書いた slots_[back] の内容は、
//     その状態語を acquire で観測した read() の後のコピーから必ず見える（＝古い中身は読めない）。
//   * 読み手 → 書き手: スロットは 3 枚で使い回すので、読み手が読み終えたスロットはいずれ書き手の
//     back に戻ってくる。read() の release と publish() の acquire がその向きの happens-before を
//     作り、「読み手のコピー → 書き手の次の上書き」の順序を保証する（これを落とすと read-after-write
//     ではなく write-after-read のデータ競合になる。TSan が実際に検出した）。
//   * 一方、CAS の失敗時順序と read() の先頭 load は relaxed でも正しい（成功した CAS が必要な辺を
//     全て張る）。現状は読みやすさのために acquire にしてある。ここを締めても緩めても意味は変わらない
//     ので、「最適化」や「安全側に倒す」目的で触らないこと。

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "quantviz/bridge/spsc_ring.hpp"  // kCacheLineSize

namespace quantviz::bridge {

template <class T>
class TripleBuffer {
    static_assert(std::is_trivially_copyable_v<T>, "TripleBuffer<T>: T must be trivially copyable");
    static_assert(std::is_default_constructible_v<T>, "TripleBuffer<T>: T must be default constructible");
    static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
                  "TripleBuffer: the state word must be lock-free");

public:
    using value_type = T;

    static constexpr std::size_t slot_count() noexcept { return kSlots; }

    TripleBuffer()                               = default;
    TripleBuffer(const TripleBuffer&)            = delete;
    TripleBuffer& operator=(const TripleBuffer&) = delete;

    /// 書き手専用。ここへ書き、publish() で公開する。publish() までは読み手から見えない。
    ///
    /// 【中身は空ではない】返るのは使い回しのスロットで、中身は「2 回前に publish した面」
    /// （読み手が読み終えて返してきたスロット）。ゼロ初期化も、前回自分が書いた内容の保持も
    /// 期待してはならない。書き手は毎回**全フィールド**を書くこと。「変わった所だけ書く」実装は
    /// 2 世代前の値が混ざった面を publish する。
    ///
    /// 【publish はちょうど 1 回】back() に書いたら publish() を 1 回だけ呼ぶ。続けて 2 回呼ぶと
    /// back と middle が元に戻り、いま公開したはずの面が未公開に戻って古い面が「最新」になる。
    ///
    /// back の添字を変えるのは書き手自身だけなので、状態語の読みは relaxed で足りる。
    T& back() noexcept { return slots_[field(state_.load(std::memory_order_relaxed), kBackShift)].value; }

    /// 書き手専用。back を「最新」として公開し、直前の middle を次の back として確保する。
    /// 読み手を待たない（CAS が失敗するのは読み手が同時に swap した時だけ。再試行で必ず進む）。
    /// release = 書いた内容の公開、acquire = 読み手が手放したスロットの回収（冒頭のコメント参照）。
    void publish() noexcept {
        std::uint32_t s = state_.load(std::memory_order_relaxed);
        std::uint32_t next;
        do {
            next = swap(s, kBackShift, kMiddleShift) | kNewBit;
        } while (!state_.compare_exchange_weak(s, next, std::memory_order_acq_rel,
                                               std::memory_order_acquire));
    }

    /// 読み手専用。未読の面があれば front として引き取り out へコピーして true。無ければ false。
    /// acquire = publish 前に書かれた内容の取得、release = 前回読み終えた front の返却。
    /// front を書き換えるのは読み手だけ、書き手は back にしか書かないので、コピー中に裂けない。
    bool read(T& out) noexcept {
        std::uint32_t s = state_.load(std::memory_order_acquire);
        if ((s & kNewBit) == 0) return false;  // 未読フラグを下ろすのは読み手だけ
        std::uint32_t next;
        do {
            next = swap(s, kMiddleShift, kFrontShift) & ~kNewBit;
        } while (!state_.compare_exchange_weak(s, next, std::memory_order_acq_rel,
                                               std::memory_order_acquire));
        const std::size_t front = field(next, kFrontShift);
        assert(front < kSlots);  // 添字の組は常に {0,1,2} の順列（Debug のみ）
        out = slots_[front].value;
        return true;
    }

    /// どちらのスレッドからも呼べる。未読の面があるか。
    bool has_new() const noexcept { return (state_.load(std::memory_order_acquire) & kNewBit) != 0; }

private:
    static constexpr std::size_t   kSlots       = 3;
    static constexpr std::uint32_t kIndexMask   = 0x3u;
    static constexpr unsigned      kBackShift   = 0;       ///< 状態語 bit 1:0 — 書き手が書くスロット
    static constexpr unsigned      kMiddleShift = 2;       ///< 状態語 bit 3:2 — 公開済みの最新スロット
    static constexpr unsigned      kFrontShift  = 4;       ///< 状態語 bit 5:4 — 読み手が読むスロット
    static constexpr std::uint32_t kNewBit      = 1u << 6; ///< 状態語 bit 6 — 未読フラグ
    /// 初期状態: back = 0, middle = 1, front = 2, 未読なし
    static constexpr std::uint32_t kInitial = (0u << kBackShift) | (1u << kMiddleShift) | (2u << kFrontShift);

    static constexpr std::size_t field(std::uint32_t s, unsigned shift) noexcept {
        return static_cast<std::size_t>((s >> shift) & kIndexMask);
    }
    /// 状態語の 2 つの添字フィールドを入れ替える（残りのビットはそのまま）。
    static constexpr std::uint32_t swap(std::uint32_t s, unsigned a, unsigned b) noexcept {
        const std::uint32_t va   = (s >> a) & kIndexMask;
        const std::uint32_t vb   = (s >> b) & kIndexMask;
        const std::uint32_t rest = s & ~((kIndexMask << a) | (kIndexMask << b));
        return rest | (vb << a) | (va << b);
    }

    /// 各スロットを別キャッシュラインに置く（書き手の back と読み手の front で false sharing しない）。
    struct alignas(kCacheLineSize) Slot {
        T value{};
    };

    alignas(kCacheLineSize) std::atomic<std::uint32_t> state_{kInitial};
    std::array<Slot, kSlots> slots_{};  // Slot 自体が 64B 境界 → 状態語とも別ライン
};

}  // namespace quantviz::bridge
