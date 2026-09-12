// TRIDIAG-xx — core/math/tridiag.hpp の仕様テスト
// Thomas 法を密行列 Gauss 消去（テスト内の参照実装）と比較し、単位行列・n=1・
// 大規模な対角優位ランダム系の残差を検証する。
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "quantviz/core/math/tridiag.hpp"
#include "quantviz/core/rng.hpp"

using Catch::Matchers::WithinRel;
using quantviz::core::Rng;
using quantviz::core::tridiag_solve;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// 部分ピボット付き Gauss 消去（テスト内の参照実装）。A は n×n row-major。
std::vector<double> dense_solve(std::vector<double> A, std::vector<double> b, std::size_t n) {
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t p = k;
        for (std::size_t i = k + 1; i < n; ++i)
            if (std::abs(A[i * n + k]) > std::abs(A[p * n + k])) p = i;
        if (p != k) {
            for (std::size_t j = 0; j < n; ++j) std::swap(A[k * n + j], A[p * n + j]);
            std::swap(b[k], b[p]);
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            const double f = A[i * n + k] / A[k * n + k];
            for (std::size_t j = k; j < n; ++j) A[i * n + j] -= f * A[k * n + j];
            b[i] -= f * b[k];
        }
    }
    std::vector<double> x(n);
    for (std::size_t i = n; i-- > 0;) {
        double s = b[i];
        for (std::size_t j = i + 1; j < n; ++j) s -= A[i * n + j] * x[j];
        x[i] = s / A[i * n + i];
    }
    return x;
}

/// 三重対角 (a, b, c) を密行列に展開する。a: 下対角 (n−1), b: 対角 (n), c: 上対角 (n−1)。
std::vector<double> to_dense(std::span<const double> a, std::span<const double> b, std::span<const double> c) {
    const std::size_t   n = b.size();
    std::vector<double> A(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        A[i * n + i] = b[i];
        if (i > 0) A[i * n + i - 1] = a[i - 1];
        if (i + 1 < n) A[i * n + i + 1] = c[i];
    }
    return A;
}

/// 残差 ‖Ax − d‖∞
double residual_inf(std::span<const double> a, std::span<const double> b, std::span<const double> c,
                    std::span<const double> d, std::span<const double> x) {
    const std::size_t n = b.size();
    double            r = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double ax = b[i] * x[i];
        if (i > 0) ax += a[i - 1] * x[i - 1];
        if (i + 1 < n) ax += c[i] * x[i + 1];
        r = std::max(r, std::abs(ax - d[i]));
    }
    return r;
}

}  // namespace

TEST_CASE("TRIDIAG-01: n=5 Thomas solve matches a dense Gaussian elimination to a relative 1e-12",
          "[tridiag][numeric]") {
    // 非対称で、3 行目は対角優位でない（|b_3| = 0.8 < |a_2| + |c_3| = 0.5 + 0.5）が正則な系。
    // 参照は部分ピボット付き Gauss 消去。
    const std::array<double, 4> a{1.0, -2.0, 0.5, 3.0};
    const std::array<double, 5> b{4.0, -5.0, 6.0, 0.8, -7.0};
    const std::array<double, 4> c{-1.0, 2.0, 1.5, -0.5};
    const std::array<double, 5> d{1.0, 2.0, 3.0, 4.0, 5.0};

    const std::vector<double> expect = dense_solve(to_dense(a, b, c), std::vector<double>(d.begin(), d.end()), 5);

    SECTION("separate output buffer") {
        std::array<double, 5> x{}, work{};
        REQUIRE(tridiag_solve(a, b, c, d, x, work));
        for (std::size_t i = 0; i < 5; ++i) {
            INFO("i=" << i);
            // 別実装との一致（03_tdd_spec.md §5.5 は相対 1e-12、§2.4 の 1e-10 より厳しい方を採る）。
            CHECK_THAT(x[i], WithinRel(expect[i], 1e-12));
        }
    }

    SECTION("in-place: x aliases d and gives bit-identical results") {
        std::array<double, 5> x{}, work{};
        REQUIRE(tridiag_solve(a, b, c, d, x, work));
        std::array<double, 5> dx = d;  // 右辺を上書きして解を受け取る
        std::array<double, 5> work2{};
        REQUIRE(tridiag_solve(a, b, c, dx, dx, work2));
        for (std::size_t i = 0; i < 5; ++i) {
            INFO("i=" << i);
            CHECK(dx[i] == x[i]);
        }
    }
}

TEST_CASE("TRIDIAG-02: the identity matrix returns x = d exactly", "[tridiag][unit]") {
    constexpr std::size_t n = 8;
    std::array<double, n - 1> a{}, c{};  // 0
    std::array<double, n>     b{};
    b.fill(1.0);
    std::array<double, n> d{}, x{}, work{};
    for (std::size_t i = 0; i < n; ++i) d[i] = -3.5 + 1.25 * static_cast<double>(i);

    REQUIRE(tridiag_solve(a, b, c, d, x, work));
    for (std::size_t i = 0; i < n; ++i) {
        INFO("i=" << i);
        CHECK(x[i] == d[i]);  // 1 で割り 0 を引くだけなのでビット一致
    }
}

TEST_CASE("TRIDIAG-03: n=1 works, and a zero / non-finite pivot or a short work buffer returns false",
          "[tridiag][unit]") {
    const std::span<const double> none;  // a, c は長さ 0

    SECTION("n=1: b x = d") {
        const std::array<double, 1> b{3.0}, d{6.0};
        std::array<double, 1>       x{}, work{};
        REQUIRE(tridiag_solve(none, b, none, d, x, work));
        CHECK_THAT(x[0], WithinRel(2.0, 1e-15));
    }
    SECTION("n=1: zero pivot fails") {
        const std::array<double, 1> b{0.0}, d{6.0};
        std::array<double, 1>       x{}, work{};
        CHECK_FALSE(tridiag_solve(none, b, none, d, x, work));
    }
    SECTION("n=1: NaN pivot fails") {
        const std::array<double, 1> b{kNaN}, d{6.0};
        std::array<double, 1>       x{}, work{};
        CHECK_FALSE(tridiag_solve(none, b, none, d, x, work));
    }
    SECTION("n=3: a pivot that becomes zero during elimination fails") {
        // 2 行目のピボット b_1 − a_0 c_0 / b_0 = 1 − 1·1/1 = 0
        const std::array<double, 2> a{1.0, 1.0}, c{1.0, 1.0};
        const std::array<double, 3> b{1.0, 1.0, 1.0}, d{1.0, 2.0, 3.0};
        std::array<double, 3>       x{}, work{};
        CHECK_FALSE(tridiag_solve(a, b, c, d, x, work));
    }
    SECTION("work shorter than n fails without touching x") {
        const std::array<double, 1> b{3.0}, d{6.0};
        std::array<double, 1>       x{-1.0};
        std::span<double>           work;  // 長さ 0 < n
        CHECK_FALSE(tridiag_solve(none, b, none, d, x, work));
        CHECK(x[0] == -1.0);
    }
    SECTION("n=0 is vacuously solved") {
        std::span<double> x, work;
        CHECK(tridiag_solve(none, none, none, none, x, work));
    }
}

TEST_CASE("TRIDIAG-04: a diagonally dominant random n=1000 system has residual ||Ax-d||_inf <= 1e-10",
          "[tridiag][numeric]") {
    constexpr std::size_t n = 1000;
    Rng                   rng(20260912);
    std::vector<double>   a(n - 1), b(n), c(n - 1), d(n), x(n), work(n);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        a[i] = 2.0 * rng.uniform() - 1.0;
        c[i] = 2.0 * rng.uniform() - 1.0;
    }
    for (std::size_t i = 0; i < n; ++i) {
        // 厳密な対角優位: |b_i| ≥ |a_{i−1}| + |c_i| + 1（符号はランダム）
        const double off  = (i > 0 ? std::abs(a[i - 1]) : 0.0) + (i + 1 < n ? std::abs(c[i]) : 0.0);
        const double sign = rng.uniform() < 0.5 ? -1.0 : 1.0;
        b[i]              = sign * (off + 1.0 + rng.uniform());
        d[i]              = 10.0 * (2.0 * rng.uniform() - 1.0);
    }

    REQUIRE(tridiag_solve(a, b, c, d, x, work));
    // 対角優位なので Thomas 法は後退安定。|x| ≲ 10、丸めは n·eps·10 ≈ 2e-12 のオーダー。
    // 許容 1e-10 は 03_tdd_spec.md §5.5 の値。
    CHECK(residual_inf(a, b, c, d, x) <= 1e-10);
    for (const double v : x) CHECK(std::isfinite(v));
}
