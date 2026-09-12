// AAD-01..08 — core/aad/tape.hpp（テープ式随伴自動微分）の仕様テスト
// 四則の偏微分（AAD-01）、初等関数の微分（AAD-02）、連鎖律と BS 形の式（AAD-03）、1 回の逆伝播で
// 得る全勾配と propagate の契約（AAD-04）、rewind と再利用（AAD-05）、テープ長の勘定（AAD-06）、
// seed 固定のランダム式 vs 中心差分（AAD-07）、reserve 後の無確保と容量超過の安全性（AAD-08）。
//
// アロケーション計数について（AAD-08）: グローバル operator new / delete の置き換えは既に
// test_order_book.cpp（LOB-14）が行っており、1 つのプログラムに 2 つは置けない（多重定義）。
// そこで glibc の mallinfo2() を「ヒープ計」として使い、測る前に既知の確保で計器自体が動くことを
// 自己検査する。計器が使えない環境（非 glibc、あるいは ASan のように malloc を差し替える構成では
// mallinfo2 が 0 を返す）では確保の計数だけを飛ばし、構造的な検査（capacity 不変・値の正しさ・
// 容量超過でも落ちないこと）は常に実行する。
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <span>
#include <vector>

#if defined(__GLIBC__)
#include <features.h>
// __GLIBC_PREREQ は <features.h> が定義する関数マクロ。この 2 行は外側の #if defined(__GLIBC__) +
// <features.h> の入れ子から出してはいけない（未定義のマクロを呼び出す形になり、警告ではなく
// プリプロセッサの hard error になる）。mallinfo2 は glibc 2.33 以降。
#if defined(__GLIBC_PREREQ) && __GLIBC_PREREQ(2, 33)
#define QUANTVIZ_TEST_HEAP_METER 1
#include <malloc.h>
#endif
#endif

#include "quantviz/core/aad/tape.hpp"
#include "quantviz/core/pricing/black_scholes.hpp"
#include "quantviz/core/rng.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::core::Rng;
using quantviz::core::aad::kNoNode;
using quantviz::core::aad::Tape;
using quantviz::core::aad::Var;

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

/// 逆伝播の結果。`g(v)` で Var v の随伴（= dy/dv）を引く。定数・記録失敗の Var は 0。
struct Gradient {
    std::vector<double> adj;

    double operator()(const Var& v) const {
        return (v.node() < adj.size()) ? adj[v.node()] : 0.0;
    }
};

Gradient gradient_of(const Tape& t, const Var& y) {
    Gradient g;
    g.adj.assign(t.size(), 0.0);
    t.propagate(y.node(), g.adj);
    return g;
}

// ---------------------------------------------------------------------------------------------------
// AAD-07 のランダム式。double でも Var でも同じコードを評価できるようにテンプレートにしてある
// （double 版が中心差分の被験者、Var 版が AAD の被験者。同じ演算列であることが比較の前提）。
//
// 定義域の安全性: プール（入力 3 つ + 生成したノード）の値は必ず [0.5, 2] に入るように演算を選んである。
//   0: sqrt(a*b) ∈ [0.5, 2]         1: 0.5(a+b) ∈ [0.5, 2]        2: 0.5 + 0.375(a/b) ∈ [0.59, 2]
//   3: 1.25 + 0.5(a-b) ∈ [0.5, 2]   4: 0.5 + 0.375(a*b) ∈ [0.59, 2]  5: exp(0.25(a-b)) ∈ [0.69, 1.46]
//   6: 1.2 + 0.7 log(a) ∈ [0.72, 1.69]                             7: 0.4 + 0.8 erfc(0.5(a-b)) ∈ [0.63, 1.77]
//   8: 0.5 + 1.5 Phi(a-b) ∈ [0.6, 1.9]                             9: 1/a ∈ [0.5, 2]
// この閉じ方のおかげで log / sqrt の引数は常に正、除数は常に 0.5 以上、exp は必ず有界であり、
// erfc / Phi の引数も |x| ≤ 1.5 なので飽和しない（飽和すると真の微分が 0 に潰れ、中心差分との
// 相対比較が無意味になる）。演算の中身は加減乗除・exp・log・sqrt・erfc・Phi をすべて含む。
// ---------------------------------------------------------------------------------------------------
constexpr int         kNumOpKinds = 10;
constexpr std::size_t kNumInputs  = 3;

struct ExprOp {
    int         kind;
    std::size_t a;
    std::size_t b;
};

using Expr = std::vector<ExprOp>;

/// seed 固定で n_ops 個の演算を合成する。被演算子はそれまでのプール全体から一様に選ぶ。
Expr random_expr(Rng& rng, std::size_t n_ops) {
    Expr e;
    e.reserve(n_ops);
    for (std::size_t i = 0; i < n_ops; ++i) {
        const std::size_t pool = kNumInputs + i;
        ExprOp            op{};
        op.kind = static_cast<int>(rng.uniform() * kNumOpKinds);
        if (op.kind >= kNumOpKinds) op.kind = kNumOpKinds - 1;  // uniform() が 1.0 を返す縁の保険
        op.a = static_cast<std::size_t>(rng.uniform() * static_cast<double>(pool)) % pool;
        op.b = static_cast<std::size_t>(rng.uniform() * static_cast<double>(pool)) % pool;
        e.push_back(op);
    }
    return e;
}

template <class T>
T eval_expr(const Expr& e, std::span<const T> x) {
    using std::erfc;
    using std::exp;
    using std::log;
    using std::sqrt;
    using quantviz::core::norm_cdf;  // double 版。T = Var のときは ADL で aad::norm_cdf が選ばれる

    std::vector<T> pool(x.begin(), x.end());
    pool.reserve(x.size() + e.size());
    for (const ExprOp& op : e) {
        const T a = pool[op.a];
        const T b = pool[op.b];
        T       y{};
        switch (op.kind) {
            case 0: y = sqrt(a * b); break;
            case 1: y = 0.5 * (a + b); break;
            case 2: y = 0.5 + 0.375 * (a / b); break;
            case 3: y = 1.25 + 0.5 * (a - b); break;
            case 4: y = 0.5 + 0.375 * (a * b); break;
            case 5: y = exp(0.25 * (a - b)); break;
            case 6: y = 1.2 + 0.7 * log(a); break;
            case 7: y = 0.4 + 0.8 * erfc(0.5 * (a - b)); break;
            case 8: y = 0.5 + 1.5 * norm_cdf(a - b); break;
            default: y = 1.0 / a; break;
        }
        pool.push_back(y);
    }
    return pool.back();
}

/// AAD-08 のヒープ計。プロセスが malloc から握っているバイト数。
std::size_t heap_in_use() noexcept {
#if defined(QUANTVIZ_TEST_HEAP_METER)
    return mallinfo2().uordblks;
#else
    return 0;
#endif
}

/// 計器の自己検査: 既知の確保でヒープ計が動くか（動かない環境では AAD-08 の計数を飛ばす）。
bool heap_meter_works() {
    // 32 KiB: glibc の mmap しきい値（既定 128 KiB）より小さくないと uordblks に載らない。
    const std::size_t     before = heap_in_use();
    std::vector<double>   probe(std::size_t{1} << 12, 1.0);
    const std::size_t     after = heap_in_use();
    const volatile double sink  = probe[0];
    return after > before && sink == 1.0;
}

/// x を 1 入力として n_ops 回（1 回あたり 2 ノード）記録し、逆伝播して dy/dx を返す。
/// 使うノード数は 1 + 2 n_ops。テープは rewind してから使う。
double record_and_propagate(Tape& t, std::span<double> adj, std::size_t n_ops) noexcept {
    t.rewind();
    Var x(t, 1.0009);
    Var y = x;
    for (std::size_t i = 0; i < n_ops; ++i) y = y * 0.999 + 0.001;
    t.propagate(y.node(), adj);
    return adj[x.node()];
}

}  // namespace

// ===================================================================================================
// AAD-01 — 四則の偏微分
// ===================================================================================================
TEST_CASE("AAD-01: partial derivatives of + - * / match the analytic values", "[aad][numeric]") {
    // 値は非 2 進小数（a/b や 1/s の丸めを避けない）。偏微分の大きさは高々 ~5 なので絶対 1e-15 は数 ulp。
    constexpr double a0 = 1.7;
    constexpr double b0 = 1.25;
    constexpr double s  = 2.5;

    Tape t;
    t.reserve(64);

    auto run = [&](auto&& build) {
        t.rewind();
        Var        a(t, a0);
        Var        b(t, b0);
        const Var  y = build(a, b);
        Gradient   g = gradient_of(t, y);
        return std::array<double, 3>{y.value(), g(a), g(b)};
    };

    SECTION("Var op Var") {
        const auto add = run([](const Var& a, const Var& b) { return a + b; });
        CHECK_THAT(add[0], WithinAbs(a0 + b0, 1e-15));
        CHECK_THAT(add[1], WithinAbs(1.0, 1e-15));
        CHECK_THAT(add[2], WithinAbs(1.0, 1e-15));

        const auto sub = run([](const Var& a, const Var& b) { return a - b; });
        CHECK_THAT(sub[0], WithinAbs(a0 - b0, 1e-15));
        CHECK_THAT(sub[1], WithinAbs(1.0, 1e-15));
        CHECK_THAT(sub[2], WithinAbs(-1.0, 1e-15));

        const auto mul = run([](const Var& a, const Var& b) { return a * b; });
        CHECK_THAT(mul[0], WithinAbs(a0 * b0, 1e-15));
        CHECK_THAT(mul[1], WithinAbs(b0, 1e-15));
        CHECK_THAT(mul[2], WithinAbs(a0, 1e-15));

        const auto div = run([](const Var& a, const Var& b) { return a / b; });
        CHECK_THAT(div[0], WithinAbs(a0 / b0, 1e-15));
        CHECK_THAT(div[1], WithinAbs(1.0 / b0, 1e-15));
        CHECK_THAT(div[2], WithinAbs(-a0 / (b0 * b0), 1e-15));

        // 同じ Var を 2 回使うと随伴は足し合わされる（y = a * a → dy/da = 2a）
        const auto sq = run([](const Var& a, const Var&) { return a * a; });
        CHECK_THAT(sq[0], WithinAbs(a0 * a0, 1e-15));
        CHECK_THAT(sq[1], WithinAbs(2.0 * a0, 1e-15));
    }

    SECTION("Var op scalar / scalar op Var (both operand orders)") {
        const auto p1 = run([](const Var& a, const Var&) { return a + s; });
        CHECK_THAT(p1[0], WithinAbs(a0 + s, 1e-15));
        CHECK_THAT(p1[1], WithinAbs(1.0, 1e-15));
        const auto p2 = run([](const Var& a, const Var&) { return s + a; });
        CHECK_THAT(p2[0], WithinAbs(s + a0, 1e-15));
        CHECK_THAT(p2[1], WithinAbs(1.0, 1e-15));

        const auto m1 = run([](const Var& a, const Var&) { return a - s; });
        CHECK_THAT(m1[0], WithinAbs(a0 - s, 1e-15));
        CHECK_THAT(m1[1], WithinAbs(1.0, 1e-15));
        const auto m2 = run([](const Var& a, const Var&) { return s - a; });
        CHECK_THAT(m2[0], WithinAbs(s - a0, 1e-15));
        CHECK_THAT(m2[1], WithinAbs(-1.0, 1e-15));

        const auto t1 = run([](const Var& a, const Var&) { return a * s; });
        CHECK_THAT(t1[0], WithinAbs(a0 * s, 1e-15));
        CHECK_THAT(t1[1], WithinAbs(s, 1e-15));
        const auto t2 = run([](const Var& a, const Var&) { return s * a; });
        CHECK_THAT(t2[0], WithinAbs(s * a0, 1e-15));
        CHECK_THAT(t2[1], WithinAbs(s, 1e-15));

        const auto d1 = run([](const Var& a, const Var&) { return a / s; });
        CHECK_THAT(d1[0], WithinAbs(a0 / s, 1e-15));
        CHECK_THAT(d1[1], WithinAbs(1.0 / s, 1e-15));
        const auto d2 = run([](const Var& a, const Var&) { return s / a; });
        CHECK_THAT(d2[0], WithinAbs(s / a0, 1e-15));
        CHECK_THAT(d2[1], WithinAbs(-s / (a0 * a0), 1e-15));

        const auto neg = run([](const Var& a, const Var&) { return -a; });
        CHECK_THAT(neg[0], WithinAbs(-a0, 1e-15));
        CHECK_THAT(neg[1], WithinAbs(-1.0, 1e-15));
    }

    SECTION("constants do not contribute to the gradient") {
        t.rewind();
        Var       a(t, a0);
        const Var c = Var::constant(s);
        CHECK(c.is_constant());
        CHECK(c.node() == kNoNode);
        const std::size_t n_before = t.size();
        const Var         y        = a * c + c;  // 定数はノードを作らない → 単項 2 つだけ増える
        CHECK(t.size() == n_before + 2);
        const Gradient g = gradient_of(t, y);
        CHECK_THAT(y.value(), WithinAbs(a0 * s + s, 1e-15));
        CHECK_THAT(g(a), WithinAbs(s, 1e-15));
        CHECK(g(c) == 0.0);

        // 定数どうしの演算は定数のまま（テープは伸びない）
        const std::size_t n = t.size();
        const Var         cc = Var::constant(2.0) * Var::constant(3.0) + 1.0;
        CHECK(cc.is_constant());
        CHECK(cc.value() == 7.0);
        CHECK(t.size() == n);
    }
}

// ===================================================================================================
// AAD-02 — 初等関数の微分
// ===================================================================================================
TEST_CASE("AAD-02: derivatives of exp, log, sqrt, erfc and norm_cdf match the closed forms",
          "[aad][numeric]") {
    Tape t;
    t.reserve(32);

    auto deriv = [&](auto&& f, double x) {
        t.rewind();
        Var            v(t, x);
        const Var      y = f(v);
        const Gradient g = gradient_of(t, y);
        return std::array<double, 2>{y.value(), g(v)};
    };

    SECTION("exp / log / sqrt") {
        for (const double x : {-2.3, -0.4, 0.75, 1.9}) {
            const auto e = deriv([](const Var& v) { return exp(v); }, x);
            CHECK_THAT(e[0], WithinRel(std::exp(x), 1e-14));
            CHECK_THAT(e[1], WithinRel(std::exp(x), 1e-14));
        }
        for (const double x : {0.25, 1.0, 3.7}) {
            const auto l = deriv([](const Var& v) { return log(v); }, x);
            CHECK_THAT(l[0], WithinRel(std::log(x), 1e-14));
            CHECK_THAT(l[1], WithinRel(1.0 / x, 1e-14));

            const auto r = deriv([](const Var& v) { return sqrt(v); }, x);
            CHECK_THAT(r[0], WithinRel(std::sqrt(x), 1e-14));
            CHECK_THAT(r[1], WithinRel(1.0 / (2.0 * std::sqrt(x)), 1e-14));
        }
    }

    SECTION("erfc / norm_cdf") {
        const double two_over_sqrt_pi = 2.0 / std::sqrt(std::numbers::pi);
        for (const double x : {-2.3, -0.4, 0.0, 0.75, 1.9}) {
            const auto c = deriv([](const Var& v) { return erfc(v); }, x);
            CHECK_THAT(c[0], WithinRel(std::erfc(x), 1e-14));
            CHECK_THAT(c[1], WithinRel(-two_over_sqrt_pi * std::exp(-x * x), 1e-14));

            const auto n = deriv([](const Var& v) { return norm_cdf(v); }, x);
            // 値は BS カーネルの norm_cdf とビット一致すること（M5 Task 2 の AadOps の前提）
            CHECK(n[0] == quantviz::core::norm_cdf(x));
            CHECK_THAT(n[1], WithinRel(quantviz::core::norm_pdf(x), 1e-14));
        }
    }

    SECTION("domain edges: log(0) and sqrt(0) give infinite partials, not NaN traps") {
        const auto l = deriv([](const Var& v) { return log(v); }, 0.0);
        CHECK(l[0] == -std::numeric_limits<double>::infinity());
        CHECK(l[1] == std::numeric_limits<double>::infinity());
        const auto r = deriv([](const Var& v) { return sqrt(v); }, 0.0);
        CHECK(r[0] == 0.0);
        CHECK(r[1] == std::numeric_limits<double>::infinity());
    }
}

// ===================================================================================================
// AAD-03 — 連鎖律
// ===================================================================================================
TEST_CASE("AAD-03: the chain rule holds for a composition f(g(h(x)))", "[aad][numeric]") {
    Tape t;
    t.reserve(64);

    SECTION("Phi(sqrt(x^2 + 1))") {
        // h(x) = x^2 + 1, g(u) = sqrt(u), f(w) = Phi(w) → dy/dx = phi(s) * x / s,  s = sqrt(x^2+1)
        for (const double x : {-1.4, -0.3, 0.0, 0.9, 2.6}) {
            t.rewind();
            Var            v(t, x);
            const Var      y = norm_cdf(sqrt(v * v + 1.0));
            const Gradient g = gradient_of(t, y);

            const double s    = std::sqrt(x * x + 1.0);
            const double want = quantviz::core::norm_pdf(s) * x / s;
            CHECK_THAT(y.value(), WithinRel(quantviz::core::norm_cdf(s), 1e-14));
            CHECK_THAT(g(v), WithinAbs(want, 1e-14 * std::max(1.0, std::abs(want))));
        }
    }

    SECTION("exp(-log(1 + x)^2 / 2) mixes exp, log and division") {
        // dy/dx = y * (-log(1+x)) / (1+x)
        for (const double x : {0.1, 1.0, 4.5}) {
            t.rewind();
            Var            v(t, x);
            const Var      u = log(1.0 + v);
            const Var      y = exp(-(u * u) / 2.0);
            const Gradient g = gradient_of(t, y);

            const double lu   = std::log(1.0 + x);
            const double yv   = std::exp(-lu * lu / 2.0);
            const double want = yv * (-lu) / (1.0 + x);
            CHECK_THAT(y.value(), WithinRel(yv, 1e-14));
            CHECK_THAT(g(v), WithinAbs(want, 1e-14 * std::max(1.0, std::abs(want))));
        }
    }

    SECTION("BS-shaped Phi(d1): derivative wrt S (de-risks the AadOps kernel of M5 Task 2)") {
        // d1 = (log(S/K) + drift) / vol_sqrt_t,  y = 0.5 erfc(-d1/sqrt(2)) = Phi(d1)
        // dy/dS = phi(d1) / (S * vol_sqrt_t)
        constexpr double S     = 105.0;
        constexpr double K     = 100.0;
        constexpr double T     = 0.75;
        constexpr double r     = 0.03;
        constexpr double sigma = 0.22;

        const double vol_sqrt_t = sigma * std::sqrt(T);
        const double drift      = (r + 0.5 * sigma * sigma) * T;

        t.rewind();
        Var       spot(t, S);
        const Var d1 = (log(spot / K) + drift) / vol_sqrt_t;
        const Var y  = 0.5 * erfc(-d1 * quantviz::core::kInvSqrt2);

        const Gradient g = gradient_of(t, y);

        const double d1v  = (std::log(S / K) + drift) / vol_sqrt_t;
        const double want = quantviz::core::norm_pdf(d1v) / (S * vol_sqrt_t);
        CHECK_THAT(d1.value(), WithinRel(d1v, 1e-15));
        CHECK(y.value() == quantviz::core::norm_cdf(d1v));  // BS カーネルとビット一致
        CHECK_THAT(g(spot), WithinRel(want, 1e-12));

        // 同じテープで K も入力にすると Phi(d1) の K 微分も同時に出る: -phi(d1)/(K vol_sqrt_t)
        t.rewind();
        Var            s2(t, S);
        Var            k2(t, K);
        const Var      d1b = (log(s2 / k2) + drift) / vol_sqrt_t;
        const Var      y2  = norm_cdf(d1b);
        const Gradient g2  = gradient_of(t, y2);
        CHECK_THAT(g2(s2), WithinRel(want, 1e-12));
        CHECK_THAT(g2(k2), WithinRel(-quantviz::core::norm_pdf(d1v) / (K * vol_sqrt_t), 1e-12));
    }
}

// ===================================================================================================
// AAD-04 — 1 回の逆伝播で全勾配
// ===================================================================================================
TEST_CASE("AAD-04: one propagate yields the full gradient of a four-input function", "[aad][unit]") {
    // f = x0 x1 x2 + x3 / x0 + exp(x1 - x2)
    //   df/dx0 = x1 x2 - x3/x0^2,  df/dx1 = x0 x2 + e,  df/dx2 = x0 x1 - e,  df/dx3 = 1/x0
    constexpr double x0 = 1.3;
    constexpr double x1 = 0.7;
    constexpr double x2 = 2.1;
    constexpr double x3 = 0.45;

    Tape t;
    t.reserve(64);
    Var a(t, x0);
    Var b(t, x1);
    Var c(t, x2);
    Var d(t, x3);

    const Var y = a * b * c + d / a + exp(b - c);
    REQUIRE(t.size() > 4);

    std::vector<double> adj(t.size(), 0.0);
    t.propagate(y.node(), adj);  // 1 回だけ

    const double e = std::exp(x1 - x2);
    CHECK_THAT(y.value(), WithinRel(x0 * x1 * x2 + x3 / x0 + e, 1e-15));
    CHECK_THAT(adj[a.node()], WithinRel(x1 * x2 - x3 / (x0 * x0), 1e-14));
    CHECK_THAT(adj[b.node()], WithinRel(x0 * x2 + e, 1e-14));
    CHECK_THAT(adj[c.node()], WithinRel(x0 * x1 - e, 1e-14));
    CHECK_THAT(adj[d.node()], WithinRel(1.0 / x0, 1e-14));
    CHECK(adj[y.node()] == 1.0);  // 結果ノードの随伴は 1

    SECTION("propagate does nothing when the adjoint buffer is too small") {
        std::vector<double> small(t.size() - 1, -1.0);
        t.propagate(y.node(), small);
        CHECK(std::all_of(small.begin(), small.end(), [](double v) { return v == -1.0; }));
    }

    SECTION("propagate does nothing for an invalid result node") {
        std::vector<double> buf(t.size(), -1.0);
        t.propagate(kNoNode, buf);
        CHECK(std::all_of(buf.begin(), buf.end(), [](double v) { return v == -1.0; }));
        t.propagate(t.size(), buf);
        CHECK(std::all_of(buf.begin(), buf.end(), [](double v) { return v == -1.0; }));
    }

    SECTION("a larger buffer is fine; only the first size() entries are written") {
        std::vector<double> big(t.size() + 3, -1.0);
        t.propagate(y.node(), big);
        CHECK_THAT(big[a.node()], WithinRel(x1 * x2 - x3 / (x0 * x0), 1e-14));
        CHECK(big[t.size()] == -1.0);
    }
}

// ===================================================================================================
// AAD-05 — rewind
// ===================================================================================================
TEST_CASE("AAD-05: rewind empties the tape, keeps the capacity and leaves it reusable", "[aad][unit]") {
    Tape t;
    t.reserve(128);
    const std::size_t cap = t.capacity();
    REQUIRE(cap >= 128);
    REQUIRE(t.size() == 0);

    auto run = [&]() {
        t.rewind();
        Var            a(t, 1.1);
        Var            b(t, 2.3);
        const Var      y = exp(a / b) * (a + b) - log(b);
        const Gradient g = gradient_of(t, y);
        return std::array<double, 4>{y.value(), g(a), g(b), static_cast<double>(t.size())};
    };

    const auto first = run();
    REQUIRE(t.size() > 0);

    t.rewind();
    CHECK(t.size() == 0);
    CHECK(t.capacity() == cap);

    const auto second = run();
    CHECK(t.capacity() == cap);  // 再利用で確保は起きない
    // ビット一致（同じ演算列を同じ順で辿るので決定的）
    CHECK(second[0] == first[0]);
    CHECK(second[1] == first[1]);
    CHECK(second[2] == first[2]);
    CHECK(second[3] == first[3]);

    SECTION("reserve may grow mid-recording: nodes 0..n and the gradient survive") {
        // y = exp(a b) / (a + b) を「途中で reserve して広げたテープ」と「最初から十分なテープ」の
        // 両方で記録し、勾配が解析値と一致し、かつ 2 本のテープでビット一致することを見る。
        constexpr double av = 1.7;
        constexpr double bv = 0.6;

        auto build = [](Tape& tape, bool grow_midway) {
            Var       a(tape, av);
            Var       b(tape, bv);
            const Var u = a * b;
            if (grow_midway) tape.reserve(tape.capacity() + 64);  // 記録の途中で拡張
            const Var      y = exp(u) / (a + b);
            const Gradient g = gradient_of(tape, y);
            return std::array<double, 4>{y.value(), g(a), g(b), static_cast<double>(u.node())};
        };

        Tape grown;
        grown.reserve(3);  // a, b, a*b でちょうど満杯 → 続きは拡張しないと記録できない
        const auto got = build(grown, true);
        CHECK(grown.size() == 6);  // 2 leaves + (a*b, exp, a+b, div)
        CHECK(got[3] == 2.0);      // 拡張しても node 0..2 の番号は変わらない

        Tape roomy;
        roomy.reserve(64);
        const auto want = build(roomy, false);
        CHECK(got[0] == want[0]);  // ビット一致
        CHECK(got[1] == want[1]);
        CHECK(got[2] == want[2]);

        const double eu  = std::exp(av * bv);
        const double sum = av + bv;
        CHECK_THAT(got[0], WithinRel(eu / sum, 1e-14));
        CHECK_THAT(got[1], WithinRel(eu * (bv / sum - 1.0 / (sum * sum)), 1e-14));
        CHECK_THAT(got[2], WithinRel(eu * (av / sum - 1.0 / (sum * sum)), 1e-14));
    }

    SECTION("stale nodes after rewind are simply not propagated") {
        t.rewind();
        Var            x(t, 3.0);
        const Var      y = x * x;
        const Gradient g = gradient_of(t, y);
        CHECK_THAT(g(x), WithinAbs(6.0, 1e-15));

        t.rewind();
        std::vector<double> buf(4, -1.0);
        t.propagate(y.node(), buf);  // size() == 0 なので何もしない
        CHECK(buf[0] == -1.0);
    }
}

// ===================================================================================================
// AAD-06 — テープ長の勘定
// ===================================================================================================
TEST_CASE("AAD-06: the tape length equals the number of leaves plus recorded operations", "[aad][unit]") {
    // 数え方: size() = 入力 leaf の数 + 記録した演算の回数。
    //   * Var(Tape&, double) が 1 ノード（leaf）
    //   * Var どうしの二項演算が 1 ノード（親 2 つ）
    //   * Var とスカラの演算、単項関数（exp/log/sqrt/erfc/norm_cdf）、単項マイナスが 1 ノード（親 1 つ）
    //   * 定数（Var::constant、定数どうしの演算）は 0 ノード
    Tape t;
    t.reserve(64);

    Var x(t, 2.0);  // leaf 1
    Var y(t, 3.0);  // leaf 2
    CHECK(t.size() == 2);

    const Var z = x * y;      // 演算 1
    CHECK(t.size() == 3);
    const Var w = z + 1.0;    // 演算 2（スカラ = 単項）
    CHECK(t.size() == 4);
    const Var u = exp(w);     // 演算 3
    CHECK(t.size() == 5);
    const Var v = u / y;      // 演算 4
    CHECK(t.size() == 6);
    const Var q = -v;         // 演算 5
    CHECK(t.size() == 7);

    // 2 leaves + 5 ops
    CHECK(t.size() == 2 + 5);

    SECTION("a loop of n operations adds exactly n nodes") {
        const std::size_t before = t.size();
        constexpr std::size_t n  = 20;
        Var                   acc = q;
        for (std::size_t i = 0; i < n; ++i) acc = acc * 1.01;  // 単項 n 回
        CHECK(t.size() == before + n);
        CHECK(acc.node() == t.size() - 1);
    }

    SECTION("the node index is the position in the tape") {
        CHECK(x.node() == 0);
        CHECK(y.node() == 1);
        CHECK(z.node() == 2);
        CHECK(w.node() == 3);
        CHECK(u.node() == 4);
        CHECK(v.node() == 5);
        CHECK(q.node() == 6);
    }
}

// ===================================================================================================
// AAD-07 — ランダム式 vs 中心差分
// ===================================================================================================
TEST_CASE("AAD-07: random expressions agree with central differences", "[aad][numeric]") {
    // seed 固定の Rng で 20 演算のランダム式を 50 本作り、3 入力すべての偏微分を
    // 中心差分 (f(x+h) - f(x-h)) / 2h（h = 1e-6 x スケール）と比べる。
    //
    // 許容: 相対 1e-8 + 中心差分自身の丸め下限。中心差分の誤差は ~ eps |f| / h（丸め）+ h^2/6 |f'''|
    // （打ち切り）で、h = 1e-6 では丸め項だけで絶対 ~2e-10 ある。式によっては真の偏微分が 1e-3 まで
    // 小さくなる（各演算の局所微分が 0.3〜0.5 程度で、合成の深さ分だけ掛かるため）ので、純粋な相対
    // 1e-8 は差分側の分解能を下回りうる。そこで各比較は tol = 1e-8 |fd| + 32 eps |f| / h で判定し、
    // 「相対誤差が意味を持つ」部分集合だけを別に集計して 1e-8 で抑える。
    //
    // 部分集合の境目が |fd| > 1e-1 なのは測って決めた。1e-2 だと差分自身の相対分解能
    // eps |f| / (h |f'|) が ~2e-8 に達してまとめの上限 1e-8 を下回る: この生成器の seed を 200 通り
    // 振ると 12 個が worst_rel < 1e-8 を落とし、最悪 1.672e-8 だった（libm が 1 ulp 違う
    // macOS / MSVC では別の seed が同じ目に遭う = 移植先で不安定になる）。|fd| > 1e-1・式 100 本に
    // すると 200 seed での最悪 worst_rel = 2.573e-9（上限まで 3.9 倍）、n_cmp の最小は 106 なので
    // 末尾の CHECK(n_cmp > 50) も空振りしない。式を 50 本のままだと n_cmp の最小が 48 になり今度は
    // そちらが落ちるので、境目の引き上げと本数の倍増は対（許容そのものは緩めていない）。
    constexpr std::size_t n_ops     = 20;
    constexpr std::size_t n_expr    = 100;
    constexpr double      kRelFloor = 1e-1;  // これを超える |fd| でだけ相対誤差を集計する
    const std::array<double, kNumInputs> x0{0.8, 1.3, 1.7};  // すべて [0.5, 2]

    Rng  rng(20260912);
    Tape t;
    // 1 演算は最大 4 ノード（例: 0.4 + 0.8 erfc(0.5(a-b)) は sub / mul / erfc / mul / add で 5）なので
    // 余裕をもって確保する。足りないと記録が落ちて勾配が 0 になる（それ自体は AAD-08 で検査する）。
    t.reserve(kNumInputs + 8 * n_ops);

    double      worst_rel     = 0.0;   // |fd| > kRelFloor の比較だけ（相対が意味を持つ範囲）
    double      worst_rel_all = 0.0;   // 全比較（差分の分解能に負ける領域を含む）
    double      worst_abs     = 0.0;
    double      min_fd        = 1e300;
    std::size_t n_cmp         = 0;

    for (std::size_t k = 0; k < n_expr; ++k) {
        const Expr e = random_expr(rng, n_ops);

        // AAD
        t.rewind();
        std::array<Var, kNumInputs> xv{};
        for (std::size_t i = 0; i < kNumInputs; ++i) xv[i] = Var(t, x0[i]);
        const Var y = eval_expr<Var>(e, std::span<const Var>(xv));
        REQUIRE(std::isfinite(y.value()));
        REQUIRE(y.value() >= 0.49);  // 演算の選び方でプールは [0.5, 2] に閉じている
        REQUIRE(y.value() <= 2.01);
        const Gradient g = gradient_of(t, y);

        // 中心差分
        for (std::size_t i = 0; i < kNumInputs; ++i) {
            const double h = 1e-6 * std::abs(x0[i]);
            std::array<double, kNumInputs> xp = x0;
            std::array<double, kNumInputs> xm = x0;
            xp[i] += h;
            xm[i] -= h;
            const double fp = eval_expr<double>(e, std::span<const double>(xp));
            const double fm = eval_expr<double>(e, std::span<const double>(xm));
            const double fd = (fp - fm) / (2.0 * h);
            const double ad = g(xv[i]);

            const double tol = 1e-8 * std::abs(fd) + 32.0 * kEps * std::abs(y.value()) / h;
            const double err = std::abs(ad - fd);
            INFO("expr " << k << " input " << i << ": aad=" << ad << " fd=" << fd);
            CHECK(err <= tol);

            worst_abs     = std::max(worst_abs, err);
            worst_rel_all = std::max(worst_rel_all, err / std::abs(fd));
            min_fd        = std::min(min_fd, std::abs(fd));
            if (std::abs(fd) > kRelFloor) {
                worst_rel = std::max(worst_rel, err / std::abs(fd));
                ++n_cmp;
            }
        }
        // 値そのものも double 版と一致する（同じ演算列を辿っている証拠）。ビット一致ではなく相対
        // 1e-14 にしてあるのは、a*b + c の形が double 版と Var 版で別々に FMA に縮約されうるため。
        CHECK_THAT(y.value(), WithinRel(eval_expr<double>(e, std::span<const double>(x0)), 1e-14));
    }

    INFO("worst |aad - fd| = " << worst_abs << "; worst relative error over " << n_cmp
                               << " comparisons with |fd| > " << kRelFloor << " = " << worst_rel
                               << "; over all " << (n_expr * kNumInputs) << " = " << worst_rel_all
                               << " (smallest |fd| = " << min_fd << ")");
    CHECK(worst_rel < 1e-8);
    CHECK(n_cmp > 50);  // 十分な数が「相対で意味のある」比較になっていること（200 seed の最小は 106）
}

// ===================================================================================================
// AAD-08 — reserve 後の無確保・容量超過の安全性
// ===================================================================================================
TEST_CASE("AAD-08: after reserve, recording and propagation allocate nothing", "[aad][unit]") {
    constexpr std::size_t kNodes = 4096;
    constexpr std::size_t kOps   = 1000;  // 1 + 2 * 1000 = 2001 ノード

    const bool meter = heap_meter_works();

    Tape t;
    t.reserve(kNodes);
    std::vector<double> adj(kNodes, 0.0);
    const std::size_t   cap = t.capacity();
    REQUIRE(cap >= kNodes);

    // ウォームアップ（初回のページフォルトを測定区間の外で済ませる）
    const double warm = record_and_propagate(t, adj, kOps);
    REQUIRE(std::isfinite(warm));
    REQUIRE(warm != 0.0);

    const std::size_t before = heap_in_use();
    const double      again  = record_and_propagate(t, adj, kOps);
    const std::size_t after  = heap_in_use();

    CHECK(t.size() == 1 + 2 * kOps);
    CHECK(t.capacity() == cap);  // std::vector は一度も伸びていない
    CHECK(again == warm);        // 再利用しても結果はビット一致
    // dy/dx = 0.999^1000
    CHECK_THAT(warm, WithinRel(std::pow(0.999, static_cast<double>(kOps)), 1e-12));
    if (meter) {
        CHECK(after == before);
    } else {
        WARN("heap meter unavailable (non-glibc or malloc replaced); allocation counting skipped");
    }

    SECTION("overflow past capacity neither allocates nor crashes") {
        Tape                small;
        small.reserve(8);
        const std::size_t   small_cap = small.capacity();
        std::vector<double> buf(small_cap, -1.0);

        const std::size_t h0 = heap_in_use();
        Var               x(small, 2.0);
        Var               y = x;
        for (int i = 0; i < 20; ++i) y = y * 1.5 + 1.0;  // 40 演算 >> 容量 8
        const std::size_t h1 = heap_in_use();

        double ref = 2.0;
        for (int i = 0; i < 20; ++i) ref = ref * 1.5 + 1.0;

        // 値は素の double 計算と一致する（記録が落ちても計算は続く）。ref = ref*1.5 + 1.0 は
        // 片方だけ FMA に縮約されうるので相対 1e-14 で比べる。
        CHECK_THAT(y.value(), WithinRel(ref, 1e-14));
        CHECK(y.recording_failed());
        CHECK(y.node() == kNoNode);
        CHECK(small.size() == small_cap);      // 容量ちょうどで止まる
        CHECK(small.capacity() == small_cap);  // 伸びない
        if (meter) CHECK(h1 == h0);

        small.propagate(y.node(), buf);  // 番兵に対する逆伝播は何もしない
        CHECK(std::all_of(buf.begin(), buf.end(), [](double v) { return v == -1.0; }));

        // 記録できた範囲の勾配は取り出せる（テープ末尾のノードから逆伝播）
        std::vector<double> buf2(small.size(), 0.0);
        small.propagate(small.size() - 1, buf2);
        CHECK(buf2[small.size() - 1] == 1.0);
        CHECK(buf2[0] != 0.0);

        // 番兵の混ざり方: 記録失敗 ⊗ 生存、生存 ⊗ 記録失敗、記録失敗 ⊗ 定数 はすべて kNoNode。
        // 値は素の double と一致し、生きている側のノード番号は動かず、テープも伸びない。
        const std::size_t n_before = small.size();
        const Var         m1       = y * x;                   // failed (x) good
        const Var         m2       = x * y;                    // good (x) failed
        const Var         m3       = y * Var::constant(3.0);   // failed (x) constant
        CHECK(m1.node() == kNoNode);
        CHECK(m1.recording_failed());
        CHECK(m1.value() == y.value() * x.value());
        CHECK(m2.node() == kNoNode);
        CHECK(m2.recording_failed());
        CHECK(m2.value() == x.value() * y.value());
        CHECK(m3.node() == kNoNode);
        CHECK(m3.recording_failed());
        CHECK(m3.value() == y.value() * 3.0);
        CHECK(x.node() == 0);                // 生きている側は無傷
        CHECK(small.size() == n_before);     // ノードは 1 つも増えない

        // 番兵が伝播するのは「テープが満杯だから」ではなく「親が番兵だから」。容量を広げても
        // 記録失敗の Var は kNoNode のまま（勾配の鎖が切れているので記録し直しても意味がない）。
        small.reserve(n_before + 16);
        const Var m4 = y + x;
        CHECK(m4.node() == kNoNode);
        CHECK(small.size() == n_before);
        const Var fresh(small, 5.0);  // 新しい leaf は（空きができたので）記録できる
        CHECK(fresh.node() == n_before);

        // propagate は番兵にも size() にも反応せず、バッファを 1 バイトも触らない
        std::vector<double> probe(small.size(), 0.0);
        for (std::size_t i = 0; i < probe.size(); ++i) probe[i] = -1.0 - static_cast<double>(i);
        const std::vector<double> probe_copy = probe;
        const std::size_t         nbytes     = probe.size() * sizeof(double);
        small.propagate(kNoNode, probe);
        CHECK(std::memcmp(probe.data(), probe_copy.data(), nbytes) == 0);
        small.propagate(small.size(), probe);
        CHECK(std::memcmp(probe.data(), probe_copy.data(), nbytes) == 0);
    }

    SECTION("timing (informational, not asserted as a benchmark)") {
        constexpr std::size_t kBigOps = 50000;  // 1 + 2 * 50000 = 100001 ノード
        Tape                  big;
        big.reserve(1 + 2 * kBigOps);
        std::vector<double> badj(big.capacity(), 0.0);

        // 一度回してウォームアップ
        (void)record_and_propagate(big, badj, kBigOps);

        const auto t0 = std::chrono::steady_clock::now();
        big.rewind();
        Var x(big, 1.0009);
        Var y = x;
        for (std::size_t i = 0; i < kBigOps; ++i) y = y * 0.999 + 0.001;
        const auto t1 = std::chrono::steady_clock::now();
        big.propagate(y.node(), badj);
        const auto t2 = std::chrono::steady_clock::now();

        const double n       = static_cast<double>(big.size());
        const double rec_ns  = std::chrono::duration<double, std::nano>(t1 - t0).count() / n;
        const double prop_ns = std::chrono::duration<double, std::nano>(t2 - t1).count() / n;

        INFO("tape nodes = " << big.size() << "; record " << rec_ns << " ns/node; propagate " << prop_ns
                             << " ns/node");
        CHECK(big.size() == 1 + 2 * kBigOps);
        CHECK(badj[x.node()] != 0.0);
        CHECK(rec_ns < 1000.0);   // 明らかな退行だけを止める緩い上限
        CHECK(prop_ns < 1000.0);
    }
}
