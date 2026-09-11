#pragma once
// bridge/spsc_ring.hpp — 単一生産者・単一消費者のロックフリー・リングバッファ
//
// 設計上の約束:
//   * T は trivially copyable（Snapshot / Command は POD）→ memcpy 相当で完結、例外なし
//   * Capacity は 2 の冪 → インデックス計算がマスク 1 命令
//   * head/tail と両者のキャッシュを別キャッシュラインに置き false sharing を避ける
//   * 生産者は head だけを書き、消費者は tail だけを書く。互いの値は acquire で読む
//   * バッファが満杯なら push は false（ブロックしない）。落とした数は呼び手が数える

#include <array>
#include <atomic>
#include <cstddef>
#include <type_traits>

namespace quantviz::bridge {

inline constexpr std::size_t kCacheLineSize = 64;

template <class T, std::size_t Capacity>
class SpscRing {
    static_assert(std::is_trivially_copyable_v<T>, "SpscRing<T>: T must be trivially copyable");
    static_assert(std::is_default_constructible_v<T>, "SpscRing<T>: T must be default constructible");
    static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0,
                  "SpscRing: Capacity must be a power of two (>= 2)");

public:
    static constexpr std::size_t capacity() noexcept { return Capacity; }

    SpscRing() = default;
    SpscRing(const SpscRing&)            = delete;
    SpscRing& operator=(const SpscRing&) = delete;

    /// 生産者スレッド専用。満杯なら false。
    bool try_push(const T& value) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        if (head - tail_cache_ == Capacity) {
            tail_cache_ = tail_.load(std::memory_order_acquire);
            if (head - tail_cache_ == Capacity) return false;
        }
        buffer_[head & kMask] = value;
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    /// 消費者スレッド専用。空なら false。
    bool try_pop(T& out) noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (head_cache_ == tail) {
            head_cache_ = head_.load(std::memory_order_acquire);
            if (head_cache_ == tail) return false;
        }
        out = buffer_[tail & kMask];
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    /// どちらのスレッドからも呼べる近似値（監視・UI 表示用）。
    std::size_t size_approx() const noexcept {
        return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire);
    }
    bool empty_approx() const noexcept { return size_approx() == 0; }

private:
    static constexpr std::size_t kMask = Capacity - 1;

    alignas(kCacheLineSize) std::atomic<std::size_t> head_{0};   // 生産者が書く
    alignas(kCacheLineSize) std::atomic<std::size_t> tail_{0};   // 消費者が書く
    alignas(kCacheLineSize) std::size_t tail_cache_{0};          // 生産者だけが触る
    alignas(kCacheLineSize) std::size_t head_cache_{0};          // 消費者だけが触る
    alignas(kCacheLineSize) std::array<T, Capacity> buffer_{};
};

}  // namespace quantviz::bridge
