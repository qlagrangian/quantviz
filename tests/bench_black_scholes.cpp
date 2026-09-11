// BENCH-03 — 既定では実行されない（[!benchmark] タグ）。
//   ./quantviz_tests "[!benchmark][bs]" で実行。
// N = 1024 のストライク・ストリップを、素直なスカラループ（ストライクごとに bs_price）と
// SIMD 版（ループ不変量を外に出し、除算・乗算をレーン並列化）で比較する。
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <span>
#include <vector>

#include "quantviz/core/pricing/black_scholes.hpp"

using quantviz::core::bs_price_strip;
using quantviz::core::bs_price_strip_scalar;
using quantviz::core::OptionType;

TEST_CASE("BENCH-03: BS strip N=1024 SIMD vs scalar", "[!benchmark][bs]") {
    constexpr std::size_t kN = 1024;
    std::vector<double>   strikes(kN), out(kN);
    for (std::size_t i = 0; i < kN; ++i) strikes[i] = 50.0 + 0.1 * static_cast<double>(i);  // 50 .. 152.3

    const double     s = 100.0, t = 1.0, r = 0.05, sigma = 0.2;
    const OptionType type = OptionType::Call;

    BENCHMARK("scalar loop (N=1024)") {
        bs_price_strip_scalar(s, std::span<const double>(strikes), t, r, sigma, type,
                              std::span<double>(out));
        return out[kN - 1];
    };

    BENCHMARK("simd strip (N=1024)") {
        bs_price_strip(s, std::span<const double>(strikes), t, r, sigma, type, std::span<double>(out));
        return out[kN - 1];
    };
}
