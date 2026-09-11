// OPTIM-xx — core/stats/optim.hpp の仕様テスト（Nelder–Mead / BFGS / 制約変換）
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include "quantviz/core/stats/optim.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::core::bfgs;
using quantviz::core::bfgs_into;
using quantviz::core::from_positive;
using quantviz::core::from_unit;
using quantviz::core::nelder_mead;
using quantviz::core::nelder_mead_into;
using quantviz::core::OptimOptions;
using quantviz::core::OptimResult;
using quantviz::core::to_positive;
using quantviz::core::to_unit;

namespace {
using Vec2 = std::array<double, 2>;

// 最小値 0 を (1, -0.5) でとる正定値 2 次形式（交差項つき）。
// f* = 0 にしておくと最小点近傍の f の差を倍精度でそのまま分解できる
// （f* ≠ 0 だと丸めのせいで x の分解能が √ε ≈ 1.5e-8 程度に落ち、1e-8 の検証ができない）。
double quadratic(const Vec2& x) {
    const double u = x[0] - 1.0, v = x[1] + 0.5;
    return u * u + u * v + 2.0 * v * v;
}

// Rosenbrock: 最小値 0 を (1, 1) でとる。谷が曲がっているので直線探索系には難しい。
double rosenbrock(const Vec2& x) {
    const double a = 1.0 - x[0], b = x[1] - x[0] * x[0];
    return a * a + 100.0 * b * b;
}

constexpr Vec2 kQuadStart{4.0, -3.0};
constexpr Vec2 kRosenStart{-1.2, 1.0};  // 古典的な開始点
}  // namespace

TEST_CASE("OPTIM-01: Nelder-Mead finds the minimum of a quadratic to 1e-8", "[optim][numeric]") {
    OptimOptions o;
    o.tol_f = 1e-18;  // f* = 0 なので単体の頂点間 f 差は (1e-10)^2 程度まで落ちる
    o.tol_x = 1e-10;
    const auto r = nelder_mead<2>(quadratic, kQuadStart, o);
    CHECK(r.converged);
    CHECK(r.iters < o.max_iter);
    CHECK_THAT(r.x[0], WithinAbs(1.0, 1e-8));
    CHECK_THAT(r.x[1], WithinAbs(-0.5, 1e-8));
    CHECK_THAT(r.f, WithinAbs(0.0, 1e-15));
}

TEST_CASE("OPTIM-02: Nelder-Mead reaches (1,1) on Rosenbrock within 1e-4", "[optim][numeric]") {
    OptimOptions o;
    o.max_iter = 5000;
    o.tol_f    = 1e-14;
    o.tol_x    = 1e-8;
    const auto r = nelder_mead<2>(rosenbrock, kRosenStart, o);
    CHECK(r.converged);
    CHECK_THAT(r.x[0], WithinAbs(1.0, 1e-4));
    CHECK_THAT(r.x[1], WithinAbs(1.0, 1e-4));
    CHECK_THAT(r.f, WithinAbs(0.0, 1e-8));
}

TEST_CASE("OPTIM-03: the best f along the returned path is monotonically non-increasing",
          "[optim][property]") {
    // path は先頭が x0、以後「各反復後の best x」。要素数 = iters + 1、終点 = 返却する x、f は単調非増加。
    const auto check = [](const OptimResult<2>& r) {
        REQUIRE(r.iters > 0);
        REQUIRE(r.path.size() == r.iters + 1);
        CHECK(r.path.front() == kRosenStart);
        CHECK(r.path.back() == r.x);
        double prev = std::numeric_limits<double>::infinity();
        for (const auto& p : r.path) {
            const double f = rosenbrock(p);
            CHECK(f <= prev);
            prev = f;
        }
        CHECK_THAT(r.f, WithinRel(prev, 1e-12));
    };
    SECTION("Nelder-Mead") { check(nelder_mead<2>(rosenbrock, kRosenStart)); }
    SECTION("BFGS") { check(bfgs<2>(rosenbrock, kRosenStart)); }
    SECTION("the _into variants give the same results and reuse the path allocation") {
        // 計算スレッドで毎回 path を確保しないための入口。2 回目以降は capacity が変わらない。
        // 同じ呼び出し口からの 2 回はビット一致（再利用が結果に影響しない）。値を返す版との比較は
        // 相対/絶対 1e-12 で行う: -mfma では別々にインライン展開されたコピーで FMA 縮約が変わりうる。
        const auto near = [](const OptimResult<2>& a, const OptimResult<2>& b) {
            CHECK(a.iters == b.iters);
            CHECK(a.converged == b.converged);
            CHECK_THAT(a.f, WithinRel(b.f, 1e-12) || WithinAbs(b.f, 1e-12));
            for (std::size_t i = 0; i < 2; ++i) {
                CHECK_THAT(a.x[i], WithinRel(b.x[i], 1e-12) || WithinAbs(b.x[i], 1e-12));
            }
        };
        OptimResult<2> out;
        nelder_mead_into(out, rosenbrock, kRosenStart);
        check(out);
        near(out, nelder_mead<2>(rosenbrock, kRosenStart));
        const auto first = out;
        const auto cap   = out.path.capacity();
        REQUIRE(cap >= out.path.size());
        nelder_mead_into(out, rosenbrock, kQuadStart);  // 別の初期値で再利用
        CHECK(out.path.capacity() == cap);
        near(out, nelder_mead<2>(rosenbrock, kQuadStart));
        nelder_mead_into(out, rosenbrock, kRosenStart);  // 最初と同じ入力: ビット一致
        CHECK(out.path.capacity() == cap);
        CHECK(out.x == first.x);
        CHECK(out.f == first.f);
        CHECK(out.path == first.path);
        bfgs_into(out, rosenbrock, kRosenStart);
        check(out);
        CHECK(out.path.capacity() == cap);
        near(out, bfgs<2>(rosenbrock, kRosenStart));
    }
}

TEST_CASE("OPTIM-04: BFGS with a numerical gradient converges on a quadratic to 1e-8 within 20 iterations",
          "[optim][numeric]") {
    // 2 次関数では BFGS のヘッセ近似が数反復で厳密になり、以後は Newton ステップで一発収束する。
    const auto r = bfgs<2>(quadratic, kQuadStart);
    CHECK(r.converged);
    CHECK(r.iters <= 20);
    CHECK_THAT(r.x[0], WithinAbs(1.0, 1e-8));
    CHECK_THAT(r.x[1], WithinAbs(-0.5, 1e-8));
    CHECK_THAT(r.f, WithinAbs(0.0, 1e-15));
}

TEST_CASE("OPTIM-05: max_iter is never exceeded and hitting it reports converged = false", "[optim][unit]") {
    OptimOptions o;
    o.max_iter = 10;
    o.tol_f    = 0.0;  // 到達不能な許容にして必ず max_iter で止める
    o.tol_x    = 0.0;
    SECTION("Nelder-Mead") {
        const auto r = nelder_mead<2>(rosenbrock, kRosenStart, o);
        CHECK(r.iters == 10);
        CHECK_FALSE(r.converged);
        REQUIRE(r.path.size() == 11);  // x0 + 10 反復
        CHECK(r.path.front() == kRosenStart);
        CHECK(r.path.back() == r.x);
    }
    SECTION("BFGS") {
        const auto r = bfgs<2>(rosenbrock, kRosenStart, o);
        CHECK(r.iters == 10);
        CHECK_FALSE(r.converged);
        REQUIRE(r.path.size() == 11);
        CHECK(r.path.front() == kRosenStart);
        CHECK(r.path.back() == r.x);
    }
    SECTION("max_iter = 0 does nothing and returns x0 (path is still seeded with x0)") {
        o.max_iter = 0;
        const auto a = nelder_mead<2>(rosenbrock, kRosenStart, o);
        CHECK(a.iters == 0);
        CHECK_FALSE(a.converged);
        REQUIRE(a.path.size() == 1);
        CHECK(a.path.back() == kRosenStart);
        CHECK(a.x == kRosenStart);
        // f はビット一致ではなく相対 1e-12 で比べる: -mfma では最適化器の中にインライン展開された
        // 目的関数と、ここで評価する目的関数とで x ± y·z の FMA 縮約の有無が変わり、1 ulp ずれうる。
        CHECK_THAT(a.f, WithinRel(rosenbrock(kRosenStart), 1e-12));
        const auto b = bfgs<2>(rosenbrock, kRosenStart, o);
        CHECK(b.iters == 0);
        CHECK_FALSE(b.converged);
        REQUIRE(b.path.size() == 1);
        CHECK(b.path.back() == kRosenStart);
        CHECK(b.x == kRosenStart);
        CHECK_THAT(b.f, WithinRel(rosenbrock(kRosenStart), 1e-12));
    }
    SECTION("a constant objective is a stationary point: converged even with iters = 0, path never empty") {
        const auto flat = [](const Vec2&) { return 1.0; };
        const auto b    = bfgs<2>(flat, kRosenStart);  // 勾配 0 → 直ちに停留点
        CHECK(b.converged);
        CHECK(b.iters == 0);
        REQUIRE(b.path.size() == 1);
        CHECK(b.path.back() == b.x);
        CHECK(b.x == kRosenStart);
        const auto a = nelder_mead<2>(flat, kRosenStart);  // 単体が tol_x まで縮んで収束
        CHECK(a.converged);
        REQUIRE_FALSE(a.path.empty());
        CHECK(a.path.back() == a.x);
    }
    SECTION("a non-finite objective never reports convergence") {
        // f(x0) が有限でなければ何もせず converged = false、x = x0、f = f(x0)（NaN / +inf のまま）で返す。
        const auto fnan = [](const Vec2&) { return std::numeric_limits<double>::quiet_NaN(); };
        const auto finf = [](const Vec2&) { return std::numeric_limits<double>::infinity(); };
        for (const auto& r : {nelder_mead<2>(fnan, kRosenStart, o), bfgs<2>(fnan, kRosenStart, o)}) {
            CHECK_FALSE(r.converged);
            CHECK(r.iters == 0);
            CHECK(std::isnan(r.f));
            REQUIRE(r.path.size() == 1);
            CHECK(r.path.back() == r.x);
            CHECK(r.x == kRosenStart);
        }
        for (const auto& r : {nelder_mead<2>(finf, kRosenStart, o), bfgs<2>(finf, kRosenStart, o)}) {
            CHECK_FALSE(r.converged);
            CHECK(r.iters == 0);
            CHECK(r.f == std::numeric_limits<double>::infinity());
            REQUIRE(r.path.size() == 1);
            CHECK(r.path.back() == r.x);
        }
        // 途中から非有限になる場合（x0 では有限）: 収束を報告しない
        const auto cliff = [](const Vec2& x) {
            return x[0] < -1.0 ? rosenbrock(x) : std::numeric_limits<double>::quiet_NaN();
        };
        for (const auto& r : {nelder_mead<2>(cliff, kRosenStart), bfgs<2>(cliff, kRosenStart)}) {
            CHECK(r.iters <= 1000);
            REQUIRE_FALSE(r.path.empty());
            CHECK(r.path.back() == r.x);
            CHECK((r.converged ? std::isfinite(r.f) : true));  // converged なら f は有限
        }
    }
}

TEST_CASE("OPTIM-06: constraint transforms (logit / softplus) round-trip to identity (rel 1e-12)",
          "[optim][numeric]") {
    // 往復の相対誤差 1e-12 は、|x| ≤ 10 程度なら 1 − sigmoid(x) ≥ 4.5e-5 の相殺誤差
    // （絶対 1.1e-16 → 相対 2.5e-12 → logit の絶対誤差 2.5e-12）に収まることによる。
    SECTION("R -> (0,1) -> R (sigmoid then logit)") {
        for (const double x : {-8.0, -3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0, 8.0}) {
            const double u = to_unit(x);
            CHECK(u > 0.0);
            CHECK(u < 1.0);
            CHECK_THAT(from_unit(u), WithinRel(x, 1e-12) || WithinAbs(x, 1e-12));
        }
    }
    SECTION("(0,1) -> R -> (0,1) (logit then sigmoid)") {
        for (const double u : {1e-9, 1e-3, 0.25, 0.5, 0.75, 0.999, 1.0 - 1e-9}) {
            CHECK_THAT(to_unit(from_unit(u)), WithinRel(u, 1e-12));
        }
    }
    SECTION("R -> (0,inf) -> R (softplus then inverse)") {
        for (const double x : {-30.0, -5.0, -1.0, 0.0, 0.5, 3.0, 40.0, 700.0}) {
            const double p = to_positive(x);
            CHECK(p > 0.0);
            CHECK_THAT(from_positive(p), WithinRel(x, 1e-12) || WithinAbs(x, 1e-12));
        }
    }
    SECTION("(0,inf) -> R -> (0,inf) (inverse softplus then softplus)") {
        for (const double p : {1e-12, 1e-6, 0.01, 1.0, 5.0, 100.0, 1e4}) {
            CHECK_THAT(to_positive(from_positive(p)), WithinRel(p, 1e-12));
        }
    }
    SECTION("numerically stable for large |x|: no overflow, no NaN") {
        CHECK(to_unit(-800.0) >= 0.0);
        CHECK(to_unit(-800.0) < 1e-300);
        CHECK(to_unit(800.0) <= 1.0);
        CHECK(to_unit(800.0) > 1.0 - 1e-15);
        CHECK(std::isfinite(to_positive(-800.0)));
        CHECK(to_positive(-800.0) >= 0.0);
        CHECK_THAT(to_positive(800.0), WithinRel(800.0, 1e-12));
        CHECK_THAT(from_positive(800.0), WithinRel(800.0, 1e-12));
        CHECK(std::isfinite(from_unit(1e-300)));
    }
}
