// TRIPLE-xx — bridge/triple_buffer.hpp の仕様テスト（docs/03_tdd_spec.md §5.3）。
// 「最新 1 枚」交換の契約: 読み手は最後に publish された値だけを見る／書き手は読み手を待たない／
// 読み手は裂けた値（torn read）を観測しない。並行ケース（02, 03）は TSan でも回す。
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <type_traits>

#include "quantviz/bridge/triple_buffer.hpp"

using quantviz::bridge::kCacheLineSize;
using quantviz::bridge::TripleBuffer;

namespace {

/// 裂け検出用のペイロード: 連番 + 64 要素の配列 + 全体のチェックサム。
/// 書き手はスロットを 1 要素ずつ埋めるので、publish の途中を読めば必ず不整合になる。
struct Payload {
    static constexpr std::size_t kN = 64;

    std::uint64_t                 seq = 0;
    std::array<std::uint64_t, kN> data{};
    std::uint64_t                 checksum = 0;

    void fill(std::uint64_t s) noexcept {
        seq               = s;
        std::uint64_t sum = s;
        for (std::size_t i = 0; i < kN; ++i) {
            data[i] = cell(s, i);
            sum ^= data[i] * 0x9e3779b97f4a7c15ULL;
        }
        checksum = sum;
    }

    bool consistent() const noexcept {
        std::uint64_t sum = seq;
        for (std::size_t i = 0; i < kN; ++i) {
            if (data[i] != cell(seq, i)) return false;
            sum ^= data[i] * 0x9e3779b97f4a7c15ULL;
        }
        return sum == checksum;
    }

    static std::uint64_t cell(std::uint64_t s, std::size_t i) noexcept {
        return s * 1'000'003ULL + static_cast<std::uint64_t>(i);
    }
};

}  // namespace

TEST_CASE("TRIPLE-01: the reader always sees the value published last", "[triple][unit]") {
    TripleBuffer<int> tb;
    int               out = -1;

    tb.back() = 1;
    tb.publish();
    REQUIRE(tb.read(out));
    CHECK(out == 1);

    tb.back() = 2;
    tb.publish();
    tb.back() = 3;
    tb.publish();
    tb.back() = 4;
    tb.publish();
    REQUIRE(tb.read(out));
    CHECK(out == 4);  // 途中の 2, 3 は捨てられる（履歴は持たない）

    for (int i = 5; i <= 1000; ++i) {
        tb.back() = i;
        tb.publish();
        if (i % 7 == 0) {  // 読みを挟んでも常に「今の最新」が返る
            REQUIRE(tb.read(out));
            CHECK(out == i);
        }
    }
    REQUIRE(tb.read(out));
    CHECK(out == 1000);
}

TEST_CASE("TRIPLE-02: the writer never blocks, even when the reader sleeps instead of reading",
          "[triple][concurrency]") {
    // ブロックする実装ならここでハングする。上限 2 s は「読み手の 50 ms 周期に引きずられない」こと
    // を示すための粗い上限（実測: Release 0.006 s / Debug 0.020 s / ASan 0.037 s / TSan 0.125 s）。
    constexpr std::uint64_t kN = 1'000'000;

    TripleBuffer<std::uint64_t> tb;
    std::atomic<bool>           done{false};
    std::atomic<std::uint64_t>  reads{0};

    std::uint64_t last_seen = 0;  // 読み手スレッドだけが書き、join 後に main が読む

    std::thread slow_reader([&] {
        std::uint64_t v = 0;
        while (!done.load(std::memory_order_acquire)) {
            if (tb.read(v)) {
                reads.fetch_add(1, std::memory_order_relaxed);
                last_seen = v;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    const auto t0 = std::chrono::steady_clock::now();
    for (std::uint64_t i = 1; i <= kN; ++i) {
        tb.back() = i;
        tb.publish();
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    done.store(true, std::memory_order_release);
    slow_reader.join();

    CHECK(elapsed < 2.0);
    CHECK(reads.load() < kN);  // 読み手は当然ほとんど取りこぼしている

    // 最後の 1 枚は必ずどちらかで観測される: まだ未読で残っているか、読み手が done を見る直前の
    // read() で既に拾ったか。どちらも正しいので両方許す（片方だけを要求するとフレークする）。
    std::uint64_t last = 0;
    if (tb.read(last)) {
        CHECK(last == kN);
    } else {
        CHECK(last_seen == kN);
    }
}

TEST_CASE("TRIPLE-03: the reader never observes a torn value over 1M publishes", "[triple][concurrency]") {
    constexpr std::uint64_t kN = 1'000'000;

    TripleBuffer<Payload> tb;
    std::atomic<bool>     writer_done{false};

    std::thread writer([&] {
        for (std::uint64_t i = 1; i <= kN; ++i) {
            tb.back().fill(i);
            tb.publish();
        }
        writer_done.store(true, std::memory_order_release);
    });

    // ホットループでは Catch2 のアサーションを使わない（遅すぎる）。フラグに畳んで後で判定する。
    std::uint64_t observed = 0;
    std::uint64_t last_seq = 0;
    bool          torn     = false;
    bool          reorder  = false;
    Payload       p;
    for (;;) {
        if (tb.read(p)) {
            ++observed;
            if (!p.consistent()) {
                torn = true;
                break;
            }
            if (p.seq < last_seq) {
                reorder = true;
                break;
            }
            last_seq = p.seq;
        } else if (writer_done.load(std::memory_order_acquire) && !tb.has_new()) {
            break;  // 書き手が終わり、未読も無い → 全て消化した
        }
    }
    writer.join();

    CHECK_FALSE(torn);
    CHECK_FALSE(reorder);
    CHECK(observed > 0);
    CHECK(observed <= kN);
    CHECK(last_seq == kN);  // 最後に publish された 1 枚は必ず読める
}

TEST_CASE("TRIPLE-04: read returns false and leaves the output untouched when nothing is new",
          "[triple][unit]") {
    TripleBuffer<int> tb;
    int               out = -7;

    CHECK_FALSE(tb.has_new());
    CHECK_FALSE(tb.read(out));
    CHECK(out == -7);

    tb.back() = 5;
    CHECK_FALSE(tb.has_new());  // publish() するまでは「新しい」ではない
    CHECK_FALSE(tb.read(out));
    CHECK(out == -7);

    tb.publish();
    CHECK(tb.has_new());
    REQUIRE(tb.read(out));
    CHECK(out == 5);

    CHECK_FALSE(tb.has_new());
    CHECK_FALSE(tb.read(out));
    CHECK(out == 5);  // 2 回目は false で out を書き換えない
}

TEST_CASE("TRIPLE-05: T is trivially copyable and the slots sit off the state word's cache line",
          "[triple][contract]") {
    STATIC_REQUIRE(std::is_trivially_copyable_v<Payload>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<TripleBuffer<Payload>::value_type>);
    // 状態語と 3 スロットがそれぞれ 64B 境界に置かれる → 全体は 4 ライン以上、アラインは 64
    STATIC_REQUIRE(alignof(TripleBuffer<char>) == kCacheLineSize);
    STATIC_REQUIRE(sizeof(TripleBuffer<char>) == 4 * kCacheLineSize);  // 状態語 1 ライン + スロット 3 ライン
    STATIC_REQUIRE(TripleBuffer<Payload>::slot_count() == 3);
    // 非 trivially-copyable な T は static_assert でコンパイルエラーになる（仕様）
}
