// HJB-xx — core/exec/hjb_merton.hpp の仕様テスト
// Merton の消費なし最適投資問題（CRRA）の HJB を対数富 y = ln w の格子で後退に解く HjbMerton について、
// 終端条件、価値関数の単調性・凹性、最適比率 π* が全 w で定数（解析解 (μ−r)/(γσ²)）であること、
// 格子細分での収束、時間整合（半区間から再開しても同じ解）を検証する。
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "quantviz/core/exec/hjb_merton.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::core::hjb_sanitize;
using quantviz::core::HjbMerton;
using quantviz::core::HjbParams;
using quantviz::core::merton_fraction;
using quantviz::core::merton_value;

namespace {

// 標準ケース: μ = 0.08, r = 0.03, σ = 0.2, γ = 3, T = 1, w ∈ [0.2, 5]。
// 解析解 π* = (μ−r)/(γσ²) = 0.05 / 0.12 = 0.41667。
constexpr double kMu    = 0.08;
constexpr double kR     = 0.03;
constexpr double kSigma = 0.2;
constexpr double kGamma = 3.0;
constexpr double kT     = 1.0;
constexpr double kWmin  = 0.2;
constexpr double kWmax  = 5.0;

HjbParams params(std::size_t n_w, std::size_t n_t, double gamma = kGamma) {
    return HjbParams{kMu, kR, kSigma, gamma, kT, kWmin, kWmax, n_w, n_t};
}

/// CRRA 効用 U(w) = w^{1−γ}/(1−γ)（γ = 1 は log）。
double utility(double w, double gamma) {
    return gamma == 1.0 ? std::log(w) : std::pow(w, 1.0 - gamma) / (1.0 - gamma);
}

/// init して満期から t = 0 まで全ステップ回す。
HjbMerton solve_all(const HjbParams& p) {
    HjbMerton h;
    h.init(p);
    while (h.step_backward()) {
    }
    return h;
}

/// t = 0 の数値解と閉形式の最大相対誤差（全節点）。
double max_rel_error(const HjbMerton& h, const HjbParams& p) {
    double worst = 0.0;
    for (std::size_t i = 0; i < h.wealth().size(); ++i) {
        const double expect = merton_value(p, h.wealth()[i], 0.0);
        worst               = std::max(worst, std::abs(h.values()[i] - expect) / std::abs(expect));
    }
    return worst;
}

}  // namespace

TEST_CASE("HJB-04: at t = T the value function equals the CRRA utility U(w) at every node; the closed-form "
          "helpers and hjb_sanitize follow their documented formulas and clamps",
          "[hjb][unit]") {
    constexpr std::size_t n_w = 64, n_t = 16;

    SECTION("power utility (gamma = 3): V(w, T) = w^{1-gamma}/(1-gamma) on a log-uniform grid") {
        HjbMerton h;
        h.init(params(n_w, n_t));
        const std::span<const double> w = h.wealth();
        const std::span<const double> v = h.values();
        REQUIRE(w.size() == n_w);
        REQUIRE(v.size() == n_w);
        REQUIRE(h.optimal_fraction().size() == n_w);
        CHECK_THAT(w[0], WithinRel(kWmin, 1e-14));
        CHECK_THAT(w[n_w - 1], WithinRel(kWmax, 1e-14));
        // 内部格子は y = ln w で等間隔（w は幾何級数）
        const double ratio = std::pow(kWmax / kWmin, 1.0 / static_cast<double>(n_w - 1));
        for (std::size_t i = 0; i < n_w; ++i) {
            INFO("i=" << i << " w=" << w[i]);
            CHECK_THAT(w[i], WithinRel(kWmin * std::pow(ratio, static_cast<double>(i)), 1e-12));
            CHECK_THAT(v[i], WithinRel(utility(w[i], kGamma), 1e-14));
            CHECK_THAT(v[i], WithinRel(merton_value(params(n_w, n_t), w[i], kT), 1e-14));
        }
        CHECK(h.remaining() == n_t);
        CHECK(h.time() == kT);
        // value_at は y で補間。節点上では節点値、節点の間では両隣の間、範囲外・NaN は NaN
        CHECK_THAT(h.value_at(w[10]), WithinRel(v[10], 1e-14));
        const double mid = std::sqrt(w[10] * w[11]);  // y の中点
        CHECK(h.value_at(mid) <= std::max(v[10], v[11]));
        CHECK(h.value_at(mid) >= std::min(v[10], v[11]));
        CHECK(std::isnan(h.value_at(kWmin * 0.999)));
        CHECK(std::isnan(h.value_at(kWmax * 1.001)));
        CHECK(std::isnan(h.value_at(std::numeric_limits<double>::quiet_NaN())));
        CHECK_THAT(h.value_at(kWmin), WithinRel(v[0], 1e-14));
        CHECK_THAT(h.value_at(kWmax), WithinRel(v[n_w - 1], 1e-14));
    }

    SECTION("log utility (gamma = 1): V(w, T) = ln w and the closed form is ln w + kappa (T - t)") {
        HjbMerton h;
        h.init(params(n_w, n_t, 1.0));
        for (std::size_t i = 0; i < n_w; ++i) {
            INFO("i=" << i);
            CHECK_THAT(h.values()[i], WithinAbs(std::log(h.wealth()[i]), 1e-14));
        }
        const double kappa = kR + (kMu - kR) * (kMu - kR) / (2.0 * kSigma * kSigma);
        CHECK_THAT(merton_value(params(n_w, n_t, 1.0), 2.0, 0.0),
                   WithinRel(std::log(2.0) + kappa * kT, 1e-14));
        CHECK_THAT(merton_fraction(params(n_w, n_t, 1.0)), WithinRel((kMu - kR) / (kSigma * kSigma), 1e-14));
    }

    SECTION("merton_fraction = (mu-r)/(gamma sigma^2), merton_value = U(w) e^{(1-gamma) kappa tau}") {
        const HjbParams p     = params(n_w, n_t);
        const double    kappa = kR + (kMu - kR) * (kMu - kR) / (2.0 * kGamma * kSigma * kSigma);
        CHECK_THAT(merton_fraction(p), WithinRel(0.05 / 0.12, 1e-14));
        CHECK_THAT(merton_value(p, 2.0, 0.25),
                   WithinRel(utility(2.0, kGamma) * std::exp((1.0 - kGamma) * kappa * (kT - 0.25)), 1e-14));
        CHECK(std::isnan(merton_value(p, 0.0, 0.0)));
        CHECK(std::isnan(merton_value(p, std::numeric_limits<double>::quiet_NaN(), 0.0)));
    }

    SECTION("hjb_sanitize: NaN falls back to the defaults, out-of-range values clamp deterministically") {
        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        const HjbParams  d    = hjb_sanitize(HjbParams{kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, 0, 0});
        CHECK(std::isfinite(d.mu));
        CHECK(std::isfinite(d.r));
        CHECK(d.sigma > 0.0);
        CHECK(d.gamma > 0.0);
        CHECK(d.T > 0.0);
        CHECK(d.w_min > 0.0);
        CHECK(d.w_max > d.w_min);
        CHECK(d.n_w >= 8);
        CHECK(d.n_t >= 1);
        // 同じ入力は同じ出力（決定的）
        const HjbParams d2 = hjb_sanitize(HjbParams{kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, 0, 0});
        CHECK(d2.mu == d.mu);
        CHECK(d2.n_w == d.n_w);

        const HjbParams c = hjb_sanitize(HjbParams{kMu, kR, -1.0, 0.0, -2.0, -1.0, 0.5, 3, 0});
        CHECK(c.sigma > 0.0);
        CHECK(c.gamma > 0.0);
        CHECK(c.T > 0.0);
        CHECK(c.w_min > 0.0);
        CHECK(c.w_max > c.w_min);
        CHECK(c.n_w == 8);
        CHECK(c.n_t == 1);
        // w_max ≤ w_min は w_max 側を押し上げる（w_min はそのまま）
        const HjbParams e = hjb_sanitize(HjbParams{kMu, kR, kSigma, kGamma, kT, 2.0, 1.0, 64, 4});
        CHECK(e.w_min == 2.0);
        CHECK(e.w_max > 2.0);
        // 上限: n_w ≤ 4096, n_t ≤ 100000, |μ|, |r| ≤ 10（exp のオーバーフローと格子の暴走を防ぐ）
        const HjbParams big =
            hjb_sanitize(HjbParams{50.0, -50.0, kSigma, kGamma, kT, kWmin, kWmax, 10000, 1000000});
        CHECK(big.mu == 10.0);
        CHECK(big.r == -10.0);
        CHECK(big.n_w == 4096);
        CHECK(big.n_t == 100000);
        // 正常値はそのまま
        const HjbParams ok = hjb_sanitize(params(64, 4));
        CHECK(ok.mu == kMu);
        CHECK(ok.r == kR);
        CHECK(ok.sigma == kSigma);
        CHECK(ok.gamma == kGamma);
        CHECK(ok.T == kT);
        CHECK(ok.w_min == kWmin);
        CHECK(ok.w_max == kWmax);
        CHECK(ok.n_w == 64);
        CHECK(ok.n_t == 4);
        // init は sanitize 済みのパラメータで格子を組む
        HjbMerton h;
        h.init(HjbParams{kMu, kR, kSigma, kGamma, kT, kWmin, kWmax, 3, 0});
        CHECK(h.wealth().size() == 8);
        CHECK(h.remaining() == 1);
    }
}

TEST_CASE("HJB-01: the numerical optimal fraction pi*(w) is the Merton constant (mu-r)/(gamma sigma^2) "
          "within 1e-3 on the inner 80 % of the grid",
          "[hjb][numeric]") {
    // n_w = n_t = 256, w ∈ [0.2, 5] → h = ln(25)/255 = 0.0126。π* は V_y / (V_y − V_yy) の比なので V の誤差
    // より格子誤差 O(h²) が直接見える。両端 10 % の節点を除くのは、境界を閉形式の Dirichlet 値で固定して
    // いるため離散解の誤差（内部で O(h² + Δτ)）が境界で 0 に押し付けられ、その勾配・曲率が境界近傍に集まる
    // から: V 自体の誤差は小さくても、その y 微分を分子・分母に使う π* は境界の数節点で目に見えて振れる。
    // 実測: 内側 80 % の max|π − 0.41667| は t = T で 1.48e-5（終端 U の中心差分の O(h²)）、t = 0 で 1.48e-5、
    // 全節点だと 1.24e-4（境界の隣）。許容 1e-3 は内側で 70 倍、全節点でも 8 倍の余裕。
    const HjbParams p      = params(256, 256);
    const double    expect = merton_fraction(p);
    REQUIRE_THAT(expect, WithinRel(0.05 / 0.12, 1e-14));

    HjbMerton h;
    h.init(p);
    const std::size_t n  = h.wealth().size();
    const std::size_t lo = n / 10, hi = n - n / 10;

    // t = T: 終端 U(w) の中心差分から作る π* も定数（離散化誤差 O(h²) のみ）
    double worst_T = 0.0;
    for (std::size_t i = lo; i < hi; ++i)
        worst_T = std::max(worst_T, std::abs(h.optimal_fraction()[i] - expect));
    INFO("t=T max|pi - analytic| (inner 80%) = " << worst_T);
    CHECK(worst_T < 1e-3);

    while (h.step_backward()) {
    }
    REQUIRE(h.remaining() == 0);
    double worst_0 = 0.0, worst_all = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double err = std::abs(h.optimal_fraction()[i] - expect);
        worst_all        = std::max(worst_all, err);
        if (i >= lo && i < hi) worst_0 = std::max(worst_0, err);
    }
    INFO("t=0 max|pi - analytic| inner 80% = " << worst_0 << ", all nodes = " << worst_all);
    CHECK(worst_0 < 1e-3);
    CHECK(worst_all < 1e-3);  // 実測 1.24e-4: 除外は説明のためで、全節点でも許容に入る
    for (std::size_t i = lo; i < hi; ++i) {
        INFO("i=" << i << " w=" << h.wealth()[i]);
        CHECK_THAT(h.optimal_fraction()[i], WithinAbs(expect, 1e-3));
    }
    // クランプ [0, kPiMax] の内側にある
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(h.optimal_fraction()[i] >= 0.0);
        CHECK(h.optimal_fraction()[i] <= HjbMerton::kPiMax);
    }

    SECTION("binding lower bound (mu < r): pi* = 0 everywhere, V = U(w) e^{(1-gamma) r tau} (constrained)") {
        // μ = 0.01 < r = 0.03 → 非制約の π* = −0.167 は下限 0 に当たる。制約付きの閉形式は定数方策 π_c = 0 の
        // 解 V = U(w) e^{(1−γ) κ_c τ}, κ_c = r + π_c(μ−r) − ½γπ_c²σ² = r。境界の Dirichlet 値・merton_value・
        // merton_fraction はこの制約付きの形に揃える（非制約の κ = r + (μ−r)²/(2γσ²) を使うと境界が内部と
        // 違う速度で伸び、収束先が別の関数になる: 修正前の実測は境界の相対誤差 3.3e-3 = e^{(1−γ)(κ−κ_c)T} − 1）。
        // 内部の V の誤差は π = 0 で拡散 d = 0 となり移流項が 1 次の風上差分に落ちるため O(h): n_w = 256 で
        // 実測 7.6e-4（速度の相対誤差 kh/2 = 1.26e-2 × |(1−γ)r| = 0.06 に一致）、n_w = 2048 で 1.0e-4。
        HjbParams q = params(256, 256);
        q.mu        = 0.01;
        const double rate = (1.0 - kGamma) * kR;
        CHECK(merton_fraction(q) == 0.0);
        CHECK_THAT(merton_value(q, 1.0, 0.0), WithinRel(utility(1.0, kGamma) * std::exp(rate * kT), 1e-14));

        const HjbMerton c = solve_all(q);
        const std::size_t nn = c.wealth().size();
        double worst_v = 0.0;
        for (std::size_t i = 0; i < nn; ++i) {
            const double ref = utility(c.wealth()[i], kGamma) * std::exp(rate * kT);
            INFO("i=" << i << " w=" << c.wealth()[i] << " V=" << c.values()[i] << " ref=" << ref);
            CHECK_THAT(c.optimal_fraction()[i], WithinAbs(0.0, 1e-3));  // 実測ちょうど 0（クランプ）
            CHECK_THAT(c.values()[i], WithinRel(ref, 2e-3));
            worst_v = std::max(worst_v, std::abs(c.values()[i] - ref) / std::abs(ref));
        }
        INFO("mu<r: max rel |V - constrained closed form| = " << worst_v);
        // 境界は Dirichlet なので制約付き閉形式と一致（修正前は 3.3e-3 ずれていた）
        CHECK_THAT(c.values()[0], WithinRel(utility(kWmin, kGamma) * std::exp(rate * kT), 1e-12));
        CHECK_THAT(c.values()[nn - 1], WithinRel(utility(kWmax, kGamma) * std::exp(rate * kT), 1e-12));
        // 風上の 1 次: n_w を 8 倍にすると内部誤差は約 1/8
        q.n_w                = 2048;
        const HjbMerton fine = solve_all(q);
        double worst_fine    = 0.0;
        for (std::size_t i = 0; i < fine.wealth().size(); ++i) {
            const double ref = utility(fine.wealth()[i], kGamma) * std::exp(rate * kT);
            worst_fine       = std::max(worst_fine, std::abs(fine.values()[i] - ref) / std::abs(ref));
        }
        INFO("mu<r, n_w = 2048: max rel error = " << worst_fine);
        CHECK(worst_fine < 2e-4);
        CHECK(worst_fine < worst_v / 4.0);
    }

    SECTION("binding lower bound with log utility: upwind is exact on the linear V = ln w + r tau") {
        // γ = 1, μ < r → π_c = 0, V = ln w + κ_c τ = ln w + r τ は y で 1 次式なので風上差分も厳密。
        // 境界（加法の分岐 D_b + κ_c τ）を含む全節点で 1e-12 に入る（修正前は境界が 3.2e-3 ずれていた）。
        HjbParams q = params(256, 256, 1.0);
        q.mu        = 0.01;
        CHECK(merton_fraction(q) == 0.0);
        CHECK_THAT(merton_value(q, 1.0, 0.0), WithinAbs(kR * kT, 1e-14));
        const HjbMerton c = solve_all(q);
        for (std::size_t i = 0; i < c.wealth().size(); ++i) {
            INFO("i=" << i << " w=" << c.wealth()[i]);
            CHECK(c.optimal_fraction()[i] == 0.0);
            CHECK_THAT(c.values()[i], WithinAbs(std::log(c.wealth()[i]) + kR * kT, 1e-12));
        }
    }

    SECTION("binding upper bound (gamma = 0.1, unconstrained pi* = 12.5): pi* = kPiMax, V finite, monotone") {
        // π_c = 5 → κ_c = r + 5(μ−r) − ½γ·25σ² = 0.03 + 0.25 − 0.05 = 0.23, (1−γ)κ_c = 0.207。
        // d = ½·25·σ² = 0.5 で中心差分が効くので V の誤差は O(h² + Δτ)（実測 ~1e-4; 修正前は非制約の κ で
        // 1.1e-1 ずれていた）。
        HjbParams    q       = params(256, 256, 0.1);
        const double pi_c    = HjbMerton::kPiMax;
        const double kappa_c = kR + pi_c * (kMu - kR) - 0.5 * 0.1 * pi_c * pi_c * kSigma * kSigma;
        CHECK(merton_fraction(q) == pi_c);
        CHECK_THAT(merton_value(q, 1.0, 0.0),
                   WithinRel(utility(1.0, 0.1) * std::exp(0.9 * kappa_c * kT), 1e-14));
        const HjbMerton c = solve_all(q);
        const std::size_t nn = c.wealth().size();
        double worst_v = 0.0;
        for (std::size_t i = 0; i < nn; ++i) {
            const double ref = utility(c.wealth()[i], 0.1) * std::exp(0.9 * kappa_c * kT);
            INFO("i=" << i << " w=" << c.wealth()[i] << " V=" << c.values()[i] << " ref=" << ref);
            REQUIRE(std::isfinite(c.values()[i]));
            CHECK(c.optimal_fraction()[i] == pi_c);
            CHECK_THAT(c.values()[i], WithinRel(ref, 1e-3));
            worst_v = std::max(worst_v, std::abs(c.values()[i] - ref) / std::abs(ref));
            if (i + 1 < nn) CHECK(c.values()[i + 1] > c.values()[i]);
        }
        INFO("gamma=0.1: max rel |V - constrained closed form| = " << worst_v);
        CHECK(worst_v < 1e-3);
    }
}

TEST_CASE("HJB-02: the value function is concave in w at every time step (divided differences decrease)",
          "[hjb][property]") {
    // w 格子は非等間隔（y で等間隔）なので凹性は割差 s_i = (V_{i+1} − V_i)/(w_{i+1} − w_i) が非増加で判定する。
    // γ ∈ {0.5, 1, 3} で U は凹、閉形式 V = U(w) g(τ)（g > 0）も凹。離散解は M 行列の陰的ステップなので
    // 凹性を壊す振動は出ない。許容は丸め相当の相対 1e-12（|s_{i−1}| に対して）。
    for (const double gamma : {0.5, 1.0, 3.0}) {
        HjbMerton h;
        h.init(params(128, 64, gamma));
        std::size_t step = 0;
        do {
            const std::span<const double> w = h.wealth();
            const std::span<const double> v = h.values();
            double prev = (v[1] - v[0]) / (w[1] - w[0]);
            for (std::size_t i = 1; i + 1 < w.size(); ++i) {
                const double s = (v[i + 1] - v[i]) / (w[i + 1] - w[i]);
                INFO("gamma=" << gamma << " step=" << step << " i=" << i << " s=" << s << " prev=" << prev);
                CHECK(s <= prev + 1e-12 * std::abs(prev));
                prev = s;
            }
            ++step;
        } while (h.step_backward());
        CHECK(step == 65);
    }
}

TEST_CASE("HJB-03: the value function is strictly increasing in w at every time step", "[hjb][property]") {
    for (const double gamma : {0.5, 1.0, 3.0}) {
        HjbMerton h;
        h.init(params(128, 64, gamma));
        std::size_t step = 0;
        do {
            const std::span<const double> v = h.values();
            for (std::size_t i = 0; i + 1 < v.size(); ++i) {
                INFO("gamma=" << gamma << " step=" << step << " i=" << i);
                CHECK(v[i + 1] > v[i]);
            }
            ++step;
        } while (h.step_backward());
        REQUIRE(h.remaining() == 0);
        CHECK_THAT(h.time(), WithinAbs(0.0, 1e-12));
        // t = 0 の値は終端 U(w) から動いている（γ = 3 では g(T) = e^{−2κT} < 1 なので U < V < 0）
        const std::span<const double> v = h.values();
        const std::span<const double> w = h.wealth();
        for (std::size_t i = 1; i + 1 < w.size(); ++i) {
            INFO("gamma=" << gamma << " i=" << i);
            CHECK(v[i] != utility(w[i], gamma));
        }
        // 全ステップ後の step_backward は no-op で false
        CHECK_FALSE(h.step_backward());
        CHECK(h.remaining() == 0);
    }
}

TEST_CASE("HJB-05: doubling (n_w, n_t) shrinks the t = 0 error against the closed form by more than 1.5x",
          "[hjb][numeric]") {
    // 誤差 = 全節点の最大相対誤差 |V_num − V_closed| / |V_closed|（t = 0）。時間は陰的 Euler（1 次）、空間は
    // 中心差分（2 次）。実測（n_w = n_t = N）: N = 64 / 128 / 256 で 1.02e-4 / 1.21e-5 / 3.44e-6、比 8.4 / 3.5。
    // 比が 4 と 2 の間に収まらないのは空間誤差（+、2 次）と時間誤差（−、1 次）の符号が逆で N ≈ 256〜512 で
    // 部分的に打ち消すため（N = 512 は 4.06e-6 で 256 より大きい; 以降は時間誤差が支配して比 → 2）。
    // 同時細分の比 > 1.5 を要求（仕様）に加え、下の SECTION で 2 つの次数を別々に固定する。
    std::vector<double> err;
    for (const std::size_t n : {std::size_t{64}, std::size_t{128}, std::size_t{256}}) {
        const HjbParams p = params(n, n);
        err.push_back(max_rel_error(solve_all(p), p));
    }
    const double r1 = err[0] / err[1];
    const double r2 = err[1] / err[2];
    INFO("e64=" << err[0] << " e128=" << err[1] << " e256=" << err[2] << " ratios " << r1 << ", " << r2);
    CHECK(err[2] < 1e-3);  // §2.4「FDM vs 解析解」相対 1e-3
    CHECK(r1 > 1.5);
    CHECK(r2 > 1.5);

    SECTION("space alone is second order: n_t = 4096 fixed, n_w = 64 -> 128 -> 256 cuts the error by 3-5x") {
        // 実測 1.52e-4 / 3.68e-5 / 8.53e-6、比 4.13 / 4.32（n_t = 4096 の残る時間誤差 −8e-7 が最後の点を
        // 少し押し下げる）。
        std::vector<double> e;
        for (const std::size_t n : {std::size_t{64}, std::size_t{128}, std::size_t{256}}) {
            const HjbParams p = params(n, 4096);
            e.push_back(max_rel_error(solve_all(p), p));
        }
        const double s1 = e[0] / e[1], s2 = e[1] / e[2];
        INFO("space: e64=" << e[0] << " e128=" << e[1] << " e256=" << e[2] << " ratios " << s1 << ", " << s2);
        CHECK(s1 > 3.0);
        CHECK(s1 < 5.0);
        CHECK(s2 > 3.0);
        CHECK(s2 < 5.0);
    }

    SECTION("time alone is first order: n_w = 2048 fixed, n_t = 32 -> 64 -> 128 halves the error (1.7-2.3)") {
        // 実測 1.018e-4 / 5.086e-5 / 2.537e-5、比 2.00 / 2.00。n_w = 2048 の空間誤差 +1e-7 は無視できる。
        std::vector<double> e;
        for (const std::size_t n : {std::size_t{32}, std::size_t{64}, std::size_t{128}}) {
            const HjbParams p = params(2048, n);
            e.push_back(max_rel_error(solve_all(p), p));
        }
        const double t1 = e[0] / e[1], t2 = e[1] / e[2];
        INFO("time: e32=" << e[0] << " e64=" << e[1] << " e128=" << e[2] << " ratios " << t1 << ", " << t2);
        CHECK(t1 > 1.7);
        CHECK(t1 < 2.3);
        CHECK(t2 > 1.7);
        CHECK(t2 < 2.3);
    }
}

TEST_CASE("HJB-06: restarting from V(., T/2) as terminal data reproduces the full solve at t = 0 "
          "(time consistency)",
          "[hjb][numeric]") {
    // 全区間解を 128 ステップ進めて t = T/2 の V を取り、それを終端データに T' = T/2, n_t' = 128 で解き直す
    // （Δτ は同じ）。スキームは 1 段法（V^{m+1} は V^m と時刻だけで決まる）で、境界も終端データの端を
    // 閉形式の半群で伸ばすので同じ物理時刻で同じ値になり、差は exp(a)exp(b) と exp(a+b) の丸めだけ。
    // 実測の最大相対差は 2.0e-15（t = 0、全節点）。許容は相対 1e-12（丸めの 500 倍、スキームの整合性は
    // 1e-8 よりはるかに良い）。
    const HjbParams p = params(256, 256);
    HjbMerton       full;
    full.init(p);
    for (std::size_t i = 0; i < 128; ++i) REQUIRE(full.step_backward());
    REQUIRE_THAT(full.time(), WithinAbs(kT / 2.0, 1e-12));
    const std::vector<double> mid(full.values().begin(), full.values().end());

    HjbParams q = p;
    q.T         = kT / 2.0;
    q.n_t       = 128;
    HjbMerton half;
    half.init(q, mid);
    REQUIRE(half.wealth().size() == 256);
    CHECK(half.remaining() == 128);
    CHECK(half.time() == kT / 2.0);
    for (std::size_t i = 0; i < 256; ++i) {
        INFO("i=" << i);
        CHECK(half.values()[i] == mid[i]);  // 終端データはそのまま置く
        CHECK(half.optimal_fraction()[i] == full.optimal_fraction()[i]);
    }

    while (half.step_backward()) {
    }
    while (full.step_backward()) {
    }
    REQUIRE(half.remaining() == 0);
    REQUIRE(full.remaining() == 0);
    double worst = 0.0;
    for (std::size_t i = 0; i < 256; ++i) {
        worst = std::max(worst, std::abs(half.values()[i] - full.values()[i]) / std::abs(full.values()[i]));
    }
    INFO("max rel |V_restart - V_full| at t = 0: " << worst);
    for (std::size_t i = 0; i < 256; ++i) {
        INFO("i=" << i << " w=" << full.wealth()[i]);
        CHECK_THAT(half.values()[i], WithinRel(full.values()[i], 1e-12));
        CHECK_THAT(half.optimal_fraction()[i], WithinAbs(full.optimal_fraction()[i], 1e-10));
    }
    CHECK_THAT(half.value_at(1.0), WithinRel(full.value_at(1.0), 1e-12));

    SECTION("restart in place: init(q, values()) may alias the solver's own values() span") {
        // 同じ長さなら resize は再確保せず span は有効のまま; 長さが違えば size() しか読まない（ヘッダ参照）。
        HjbMerton alias;
        alias.init(p);
        for (std::size_t i = 0; i < 128; ++i) REQUIRE(alias.step_backward());
        alias.init(q, alias.values());
        CHECK(alias.remaining() == 128);
        for (std::size_t i = 0; i < 256; ++i) {
            INFO("i=" << i);
            CHECK(alias.values()[i] == mid[i]);
        }
        while (alias.step_backward()) {
        }
        for (std::size_t i = 0; i < 256; ++i) {
            INFO("i=" << i);
            CHECK_THAT(alias.values()[i], WithinRel(full.values()[i], 1e-12));
        }
    }

    SECTION("gamma in {0.5, 1}: the additive (log) and gamma < 1 boundary branches are time consistent too") {
        // γ = 1 では V = ln w + κτ が w ≈ e^{−κτ} で 0 を横切るので相対誤差（max_rel_error は |merton_value| で
        // 割る）は使えない。絶対誤差を max|V| でスケールして比べる。
        for (const double gamma : {0.5, 1.0}) {
            HjbParams pg = params(128, 64, gamma);
            HjbMerton f;
            f.init(pg);
            for (std::size_t i = 0; i < 32; ++i) REQUIRE(f.step_backward());
            const std::vector<double> m2(f.values().begin(), f.values().end());
            HjbParams qg = pg;
            qg.T         = kT / 2.0;
            qg.n_t       = 32;
            HjbMerton g;
            g.init(qg, m2);
            while (g.step_backward()) {
            }
            while (f.step_backward()) {
            }
            REQUIRE(g.remaining() == 0);
            REQUIRE(f.remaining() == 0);
            double scale = 0.0;
            for (const double v : f.values()) scale = std::max(scale, std::abs(v));
            for (std::size_t i = 0; i < 128; ++i) {
                INFO("gamma=" << gamma << " i=" << i);
                CHECK_THAT(g.values()[i], WithinAbs(f.values()[i], 1e-12 * scale));
            }
        }
    }

    SECTION("terminal data whose size does not match n_w falls back to U(w)") {
        const std::vector<double> wrong(10, 1.0);
        HjbMerton                 h;
        h.init(p, wrong);
        REQUIRE(h.wealth().size() == 256);
        for (std::size_t i = 0; i < 256; ++i) {
            INFO("i=" << i);
            CHECK(h.values()[i] == utility(h.wealth()[i], kGamma));
        }
    }
}
