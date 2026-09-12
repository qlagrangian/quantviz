// BENCH-05 — 既定では実行されない（[!benchmark] タグ）。
//   ./quantviz_tests "[!benchmark][lob]" で実行。
// MatchingEngine<65536, 4096> に 1e6 件の合成フロー（指値 70 % / 成行 10 % / 取消 20 %、価格は
// 生きている mid ± 20 ティック、数量 1〜100、core::Rng seed 固定）を流し、注文/秒を報告する。
// 目標 ≥ 1e6 注文/秒。乱数生成（uniform 4 回 ≈ 20 ns）とテスト側の live id 配列の管理も計測に含む。
// 「dropped 0」の対応物として拒否（プール枯渇・範囲外）が 0 件であることも確認する。
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "quantviz/core/micro/matching_engine.hpp"
#include "quantviz/core/rng.hpp"

using quantviz::core::Rng;
using quantviz::core::micro::Fill;
using quantviz::core::micro::MatchingEngine;
using quantviz::core::micro::OrderId;
using quantviz::core::micro::Price;
using quantviz::core::micro::Qty;
using quantviz::core::micro::Side;

namespace {

using Engine                   = MatchingEngine<65536, 4096>;
constexpr Price kMid0          = 100'000;
constexpr Price kBase          = kMid0 - 2048;
constexpr Price kMidMargin     = 64;  // mid は [base + margin, base + MaxLevels − margin] に留める
constexpr std::size_t kFillCap = 64;  // LOB シーンが渡す span と同じ大きさ

struct Flow {
    Engine               engine{kBase};
    Rng                  rng{12345};
    std::vector<OrderId> live;
    Price                mid = kMid0;
    std::array<Fill, kFillCap> fills{};
    std::uint64_t        n_fills_total = 0;

    Flow() { live.reserve(std::size_t{1} << 21); }

    void one() {
        std::size_t  n    = 0;
        const double u    = rng.uniform();
        const Side   side = rng.uniform() < 0.5 ? Side::Bid : Side::Ask;
        const Qty    qty  = 1 + static_cast<Qty>(rng.uniform() * 100.0);
        if (u < 0.7) {
            const auto bb = engine.book().best_bid();
            const auto ba = engine.book().best_ask();
            if (bb && ba) mid = (*bb + *ba) / 2;
            if (mid < kBase + kMidMargin) mid = kBase + kMidMargin;
            if (mid > kBase + static_cast<Price>(Engine::Book::kMaxLevels) - kMidMargin)
                mid = kBase + static_cast<Price>(Engine::Book::kMaxLevels) - kMidMargin;
            const Price   px = mid + static_cast<Price>(rng.uniform() * 41.0) - 20;
            const OrderId id = engine.submit_limit(side, px, qty, fills, n);
            if (id != 0) live.push_back(id);
        } else if (u < 0.8) {
            engine.submit_market(side, qty, fills, n);
        } else if (!live.empty()) {
            const std::size_t k = static_cast<std::size_t>(rng.uniform() * static_cast<double>(live.size()));
            engine.cancel(live[k]);
            live[k] = live.back();
            live.pop_back();
        }
        n_fills_total += n;
    }
};

}  // namespace

TEST_CASE("BENCH-05: matching engine 1e6 mixed orders (70% limit / 10% market / 20% cancel)",
          "[!benchmark][lob]") {
    constexpr std::uint64_t kN = 1'000'000;

    // 1 回の 1e6 件フル走行（仕様の計測）。
    {
        Flow       flow;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::uint64_t i = 0; i < kN; ++i) flow.one();
        const auto   t1      = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        const double rate    = static_cast<double>(kN) / seconds;
        const Qty discarded = flow.engine.discarded_qty(Side::Bid) + flow.engine.discarded_qty(Side::Ask);
        WARN("BENCH-05: " << kN << " orders in " << seconds * 1e3 << " ms = " << rate / 1e6
                          << " M orders/s (target >= 1.0); fills=" << flow.engine.fills()
                          << " resting=" << flow.engine.book().order_count()
                          << " pool_free=" << flow.engine.book().pool_free()
                          << " rejected=" << flow.engine.orders_rejected() << " discarded_qty=" << discarded
                          << " sizeof(OrderBook<65536,4096>)=" << sizeof(Engine::Book)
                          << " heap~=" << (65536 * 40 + 2 * 4096 * 24) / 1024 << " KiB");
        CHECK(flow.engine.orders_rejected() == 0);  // 「dropped 0」の対応物
        CHECK(flow.engine.book().check_invariants());
    }

    // Catch2 の統計付き計測（10k 件のチャンク。板の状態はサンプル間で引き継ぐ）。
    Flow flow;
    BENCHMARK("10k mixed orders") {
        for (int i = 0; i < 10'000; ++i) flow.one();
        return flow.engine.next_seq();
    };
}
