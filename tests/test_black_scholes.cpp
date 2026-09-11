// BS-xx — core/pricing/black_scholes.hpp の仕様テスト
// 無裁定境界・プットコールパリティ・数表値・極限（sigma→0, T→0）・解析 Greeks vs 中心差分・
// SIMD ストリップとスカラ版の一致を検証する。
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "quantviz/core/pricing/black_scholes.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::core::bs_greeks;
using quantviz::core::bs_price;
using quantviz::core::bs_price_strip;
using quantviz::core::bs_price_strip_scalar;
using quantviz::core::BsGreeks;
using quantviz::core::OptionType;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct Case {
    double S, K, T, r, sigma;
};

/// 広い格子（243 点）: ITM/ATM/OTM × 超短期〜長期 × 低 vol〜高 vol、r = 0 を含む。
/// 境界・パリティのような「常に成り立つ」性質の検査に使う。
std::vector<Case> wide_grid() {
    std::vector<Case> g;
    for (const double s : {50.0, 100.0, 140.0})
        for (const double k : {60.0, 100.0, 130.0})
            for (const double t : {0.02, 0.5, 2.0})
                for (const double rr : {0.0, 0.03, 0.08})
                    for (const double vol : {0.05, 0.20, 0.80}) g.push_back(Case{s, k, t, rr, vol});
    return g;
}

/// 中庸な格子（108 点）: Phi(d1) が 0/1 に丸められない範囲。開区間や単調性の厳密な検査に使う。
std::vector<Case> moderate_grid() {
    std::vector<Case> g;
    for (const double s : {80.0, 100.0, 120.0})
        for (const double k : {90.0, 100.0, 110.0})
            for (const double t : {0.25, 1.0, 2.0})
                for (const double rr : {0.0, 0.05})
                    for (const double vol : {0.10, 0.30}) g.push_back(Case{s, k, t, rr, vol});
    return g;
}

double px(const Case& c, OptionType type) { return bs_price(c.S, c.K, c.T, c.r, c.sigma, type); }

std::vector<double> run_strip(const Case& c, const std::vector<double>& strikes, OptionType type) {
    std::vector<double> out(strikes.size(), -1.0);
    bs_price_strip(c.S, std::span<const double>(strikes), c.T, c.r, c.sigma, type, std::span<double>(out));
    return out;
}

/// strip の 1 要素がスカラ価格と一致することを確認する。K = NaN のときは価格も NaN が正解
/// （有限入力に対して NaN を返さないという契約は S > 0, K > 0 が前提）なので両方 NaN で一致とみなす。
void check_matches_scalar(double got, double expect) {
    if (std::isnan(expect)) {
        CHECK(std::isnan(got));
    } else {
        CHECK_THAT(got, WithinRel(expect, 1e-15));
    }
}

}  // namespace

TEST_CASE("BS-01: call and put prices stay inside the no-arbitrage bounds", "[bs][property]") {
    for (const Case& c : wide_grid()) {
        const double disc = std::exp(-c.r * c.T);
        const double call = px(c, OptionType::Call);
        const double put  = px(c, OptionType::Put);
        // 許容は丸め由来の逸脱のみ（価格スケール S + K に対する相対 1e-12）。
        const double tol = 1e-12 * (c.S + c.K);
        INFO("S=" << c.S << " K=" << c.K << " T=" << c.T << " r=" << c.r << " sigma=" << c.sigma);
        CHECK(call >= std::max(c.S - c.K * disc, 0.0) - tol);
        CHECK(call <= c.S + tol);
        CHECK(put >= std::max(c.K * disc - c.S, 0.0) - tol);
        CHECK(put <= c.K * disc + tol);
    }
}

TEST_CASE("BS-02: put-call parity C - P = S - K exp(-rT) holds to a relative 1e-12", "[bs][numeric]") {
    for (const Case& c : wide_grid()) {
        const double expect = c.S - c.K * std::exp(-c.r * c.T);
        const double got    = px(c, OptionType::Call) - px(c, OptionType::Put);
        INFO("S=" << c.S << " K=" << c.K << " T=" << c.T << " r=" << c.r << " sigma=" << c.sigma);
        // 閉形式どうしの一致なので相対 1e-12（03_tdd_spec.md §2.4）。S - K e^{-rT} が 0 近傍の
        // 格子点では相対誤差が定義しにくいため、価格スケール S に対する絶対許容を併用する。
        CHECK_THAT(got, WithinRel(expect, 1e-12) || WithinAbs(expect, 1e-12 * c.S));
    }
}

TEST_CASE("BS-03: the textbook values S=K=100, T=1, r=0.05, sigma=0.2 give C=10.4506 and P=5.5735",
          "[bs][numeric]") {
    // 有名な数表値は絶対 1e-4（03_tdd_spec.md §2.4）。
    CHECK_THAT(bs_price(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Call), WithinAbs(10.4506, 1e-4));
    CHECK_THAT(bs_price(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Put), WithinAbs(5.5735, 1e-4));

    const BsGreeks call = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Call);
    CHECK_THAT(call.price, WithinAbs(10.4506, 1e-4));
    CHECK_THAT(call.delta, WithinAbs(0.6368, 1e-4));   // Phi(0.35)
    CHECK_THAT(call.gamma, WithinAbs(0.018762, 1e-6));  // phi(0.35) / (S sigma sqrt(T))
    CHECK_THAT(call.vega, WithinAbs(37.524, 1e-3));     // S phi(0.35) sqrt(T)
}

TEST_CASE("BS-04: as sigma goes to zero the price tends to the discounted deterministic payoff",
          "[bs][numeric]") {
    const double s = 100.0, k = 95.0, t = 1.5, r = 0.04;
    const double disc      = std::exp(-r * t);
    const double call_limit = std::max(s - k * disc, 0.0);
    const double put_limit  = std::max(k * disc - s, 0.0);

    SECTION("sigma -> 0+") {
        CHECK_THAT(bs_price(s, k, t, r, 1e-3, OptionType::Call), WithinAbs(call_limit, 1e-6));
        CHECK_THAT(bs_price(s, k, t, r, 1e-5, OptionType::Call), WithinAbs(call_limit, 1e-12));
        CHECK_THAT(bs_price(s, k, t, r, 1e-5, OptionType::Put), WithinAbs(put_limit, 1e-12));
    }

    SECTION("sigma == 0 returns the limit exactly and keeps every greek finite") {
        for (const double vol : {0.0, -1.0}) {  // 負の vol も 0 として扱う
            const BsGreeks g = bs_greeks(s, k, t, r, vol, OptionType::Call);
            INFO("sigma=" << vol);
            CHECK_THAT(g.price, WithinRel(call_limit, 1e-15));
            CHECK(g.gamma == 0.0);
            CHECK(g.vega == 0.0);
            CHECK(g.delta == 1.0);  // ITM（S > K e^{-rT}）
            CHECK(std::isfinite(g.theta));
            CHECK(std::isfinite(g.rho));
            CHECK_THAT(g.theta, WithinRel(-r * k * disc, 1e-15));
            CHECK_THAT(g.rho, WithinRel(t * k * disc, 1e-15));
        }
        const BsGreeks p = bs_greeks(s, k, t, r, 0.0, OptionType::Put);
        CHECK_THAT(p.price, WithinAbs(put_limit, 1e-15));
        CHECK(p.delta == 0.0);  // OTM
        CHECK(p.gamma == 0.0);
        CHECK(p.vega == 0.0);
        CHECK(std::isfinite(p.theta));
        CHECK(std::isfinite(p.rho));
    }

    SECTION("K <= 0, S <= 0 and sigma = NaN all fall into the degenerate branch and stay finite") {
        // 設計どおりの挙動: sigma = NaN は `sigma > 0` が false になるので退化ブランチに落ち、
        // 決定的な割引ペイオフ（有限値）を返す。NaN は伝播させない。
        CHECK_THAT(bs_price(s, k, t, r, kNaN, OptionType::Call), WithinRel(call_limit, 1e-15));
        CHECK_THAT(bs_price(s, k, t, r, kNaN, OptionType::Put), WithinAbs(put_limit, 1e-15));
        const BsGreeks gn = bs_greeks(s, k, t, r, kNaN, OptionType::Call);
        CHECK(std::isfinite(gn.price));
        CHECK(std::isfinite(gn.delta));
        CHECK(gn.gamma == 0.0);
        CHECK(gn.vega == 0.0);
        CHECK(std::isfinite(gn.theta));
        CHECK(std::isfinite(gn.rho));

        // K <= 0: Call は S - K e^{-rT}（K < 0 なら S より大きい）、Put は 0。
        for (const double bad_k : {0.0, -5.0}) {
            INFO("K=" << bad_k);
            CHECK_THAT(bs_price(s, bad_k, t, r, 0.2, OptionType::Call),
                       WithinRel(s - bad_k * disc, 1e-15));
            CHECK(bs_price(s, bad_k, t, r, 0.2, OptionType::Put) == 0.0);
        }

        // S <= 0: Call は 0、Put は K e^{-rT} - S。
        for (const double bad_s : {0.0, -1.0}) {
            INFO("S=" << bad_s);
            CHECK(bs_price(bad_s, k, t, r, 0.2, OptionType::Call) == 0.0);
            CHECK_THAT(bs_price(bad_s, k, t, r, 0.2, OptionType::Put), WithinRel(k * disc - bad_s, 1e-15));
        }
    }

    SECTION("the degenerate branch never returns -0.0 and breaks the ATM-forward tie towards OTM") {
        // ビューアが "-0.00" と表示しないように 0 は必ず +0.0 で返す。
        const BsGreeks otm_call = bs_greeks(100.0, 200.0, 0.0, 0.05, 0.0, OptionType::Call);
        CHECK(otm_call.price == 0.0);
        CHECK_FALSE(std::signbit(otm_call.price));
        CHECK_FALSE(std::signbit(otm_call.delta));
        CHECK_FALSE(std::signbit(otm_call.theta));
        CHECK_FALSE(std::signbit(otm_call.rho));

        const BsGreeks otm_put = bs_greeks(100.0, 50.0, 1.0, 0.05, 0.0, OptionType::Put);
        CHECK(otm_put.price == 0.0);
        CHECK_FALSE(std::signbit(otm_put.price));
        CHECK_FALSE(std::signbit(otm_put.delta));
        CHECK_FALSE(std::signbit(otm_put.theta));
        CHECK_FALSE(std::signbit(otm_put.rho));

        // S == K e^{-rT} ちょうど（劣微分は [0,1]）のときは OTM 側に倒して Delta = 0 とする。
        const BsGreeks atm = bs_greeks(100.0, 100.0, 0.0, 0.05, 0.0, OptionType::Call);
        CHECK(atm.price == 0.0);
        CHECK(atm.delta == 0.0);
        CHECK_FALSE(std::signbit(atm.delta));
    }
}

TEST_CASE("BS-05: as T goes to zero the call tends to its intrinsic value and ATM gamma grows",
          "[bs][numeric]") {
    SECTION("T -> 0+") {
        CHECK_THAT(bs_price(110.0, 100.0, 1e-8, 0.05, 0.2, OptionType::Call), WithinAbs(10.0, 1e-6));
        CHECK_THAT(bs_price(90.0, 100.0, 1e-8, 0.05, 0.2, OptionType::Call), WithinAbs(0.0, 1e-6));
        CHECK_THAT(bs_price(90.0, 100.0, 1e-8, 0.05, 0.2, OptionType::Put), WithinAbs(10.0, 1e-6));
    }

    SECTION("T <= 0 returns the intrinsic value with finite greeks") {
        for (const double t : {0.0, -1.0}) {
            const BsGreeks g = bs_greeks(110.0, 100.0, t, 0.05, 0.2, OptionType::Call);
            INFO("T=" << t);
            CHECK(g.price == 10.0);
            CHECK(g.delta == 1.0);
            CHECK(g.gamma == 0.0);
            CHECK(g.vega == 0.0);
            CHECK(std::isfinite(g.theta));
            CHECK(std::isfinite(g.rho));
            CHECK(bs_price(90.0, 100.0, t, 0.05, 0.2, OptionType::Call) == 0.0);
            CHECK(bs_price(90.0, 100.0, t, 0.05, 0.2, OptionType::Put) == 10.0);
        }
    }

    SECTION("ATM gamma increases monotonically as T shrinks") {
        double prev = 0.0;
        for (const double t : {2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 1e-3, 1e-4}) {
            const double gamma = bs_greeks(100.0, 100.0, t, 0.05, 0.2, OptionType::Call).gamma;
            INFO("T=" << t << " gamma=" << gamma);
            CHECK(std::isfinite(gamma));
            CHECK(gamma > prev);
            prev = gamma;
        }
    }
}

TEST_CASE("BS-06: delta, gamma and vega satisfy the standard call/put relations", "[bs][property]") {
    for (const Case& c : moderate_grid()) {
        const BsGreeks call = bs_greeks(c.S, c.K, c.T, c.r, c.sigma, OptionType::Call);
        const BsGreeks put  = bs_greeks(c.S, c.K, c.T, c.r, c.sigma, OptionType::Put);
        INFO("S=" << c.S << " K=" << c.K << " T=" << c.T << " r=" << c.r << " sigma=" << c.sigma);
        CHECK(call.delta > 0.0);
        CHECK(call.delta < 1.0);
        CHECK_THAT(put.delta, WithinAbs(call.delta - 1.0, 1e-15));
        CHECK(call.gamma > 0.0);
        CHECK(call.vega > 0.0);
        CHECK(put.gamma == call.gamma);  // 同一式なのでビット一致する
        CHECK(put.vega == call.vega);
    }
}

TEST_CASE("BS-07: the analytic greeks agree with central differences to a relative 1e-6", "[bs][numeric]") {
    // 中心差分の刻みは h = 1e-4 x 変数のスケール、許容は相対 1e-6（03_tdd_spec.md §2.4）。
    // スケールは max(|x|, 1) を使う: r = 0 のとき 1e-4 * r だと h = 0 になり差分が壊れるため。
    const auto step = [](double x) { return 1e-4 * std::max(std::abs(x), 1.0); };
    const std::vector<Case> cases = {{100.0, 100.0, 1.0, 0.05, 0.20},
                                     {90.0, 100.0, 0.5, 0.02, 0.30},
                                     {110.0, 100.0, 2.0, 0.01, 0.15},
                                     {100.0, 120.0, 0.75, 0.05, 0.25},
                                     {100.0, 95.0, 1.0, 0.0, 0.20}};  // r = 0（hr のガード）
    for (const Case& c : cases) {
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            const BsGreeks g = bs_greeks(c.S, c.K, c.T, c.r, c.sigma, type);
            const double hs = step(c.S), hv = step(c.sigma), ht = step(c.T), hr = step(c.r);
            const double base = px(c, type);
            const double s_up = bs_price(c.S + hs, c.K, c.T, c.r, c.sigma, type);
            const double s_dn = bs_price(c.S - hs, c.K, c.T, c.r, c.sigma, type);
            const double v_up = bs_price(c.S, c.K, c.T, c.r, c.sigma + hv, type);
            const double v_dn = bs_price(c.S, c.K, c.T, c.r, c.sigma - hv, type);
            const double t_up = bs_price(c.S, c.K, c.T + ht, c.r, c.sigma, type);
            const double t_dn = bs_price(c.S, c.K, c.T - ht, c.r, c.sigma, type);
            const double r_up = bs_price(c.S, c.K, c.T, c.r + hr, c.sigma, type);
            const double r_dn = bs_price(c.S, c.K, c.T, c.r - hr, c.sigma, type);

            INFO("S=" << c.S << " K=" << c.K << " T=" << c.T << " r=" << c.r << " sigma=" << c.sigma
                      << " call=" << (type == OptionType::Call));
            CHECK_THAT(g.delta, WithinRel((s_up - s_dn) / (2.0 * hs), 1e-6));
            CHECK_THAT(g.gamma, WithinRel((s_up - 2.0 * base + s_dn) / (hs * hs), 1e-6));
            CHECK_THAT(g.vega, WithinRel((v_up - v_dn) / (2.0 * hv), 1e-6));
            CHECK_THAT(g.theta, WithinRel(-(t_up - t_dn) / (2.0 * ht), 1e-6));  // Theta = -dV/dT
            CHECK_THAT(g.rho, WithinRel((r_up - r_dn) / (2.0 * hr), 1e-6));
        }
    }
}

TEST_CASE("BS-08: the call price increases in S, sigma and T and decreases in K", "[bs][property]") {
    for (const Case& c : moderate_grid()) {
        const double base = px(c, OptionType::Call);
        INFO("S=" << c.S << " K=" << c.K << " T=" << c.T << " r=" << c.r << " sigma=" << c.sigma);
        CHECK(bs_price(c.S * 1.01, c.K, c.T, c.r, c.sigma, OptionType::Call) > base);
        CHECK(bs_price(c.S, c.K, c.T, c.r, c.sigma * 1.01, OptionType::Call) > base);
        // r >= 0 の無配当ヨーロピアンコールは満期について単調増加（moderate_grid は r >= 0）。
        CHECK(bs_price(c.S, c.K, c.T * 1.01, c.r, c.sigma, OptionType::Call) > base);
        CHECK(bs_price(c.S, c.K * 1.01, c.T, c.r, c.sigma, OptionType::Call) < base);
    }
}

TEST_CASE("BS-09: the price is homogeneous of degree one in (S, K)", "[bs][property]") {
    for (const Case& c : moderate_grid()) {
        for (const double lambda : {0.5, 2.0, 10.0}) {
            for (const OptionType type : {OptionType::Call, OptionType::Put}) {
                INFO("S=" << c.S << " K=" << c.K << " T=" << c.T << " lambda=" << lambda);
                // 厳密に成り立つ恒等式。log(lambda S / lambda K) の丸めだけが誤差源なので相対 1e-12。
                CHECK_THAT(bs_price(lambda * c.S, lambda * c.K, c.T, c.r, c.sigma, type),
                           WithinRel(lambda * px(c, type), 1e-12));
            }
        }
    }
}

TEST_CASE("BS-10: every element of bs_price_strip matches the scalar price to a relative 1e-15",
          "[bs][numeric]") {
    std::vector<double> strikes;
    for (int i = 0; i < 129; ++i) strikes.push_back(20.0 + 3.0 * static_cast<double>(i));  // 20 .. 404

    const std::vector<Case> cases = {{100.0, 0.0, 1.0, 0.05, 0.20},
                                     {100.0, 0.0, 0.08, 0.01, 0.45},
                                     {100.0, 0.0, 3.0, 0.0, 0.15},
                                     {100.0, 0.0, 0.0, 0.05, 0.20},   // 退化: T = 0
                                     {100.0, 0.0, 1.0, 0.05, 0.0}};    // 退化: sigma = 0
    // 実装は超越関数も算術も両経路で同一の式を評価するので、実測の最大相対誤差は 0
    // （ビット一致）。許容 1e-15 はスカラ版と同じ演算順序を崩す実装を弾くための上限。
    for (const Case& c : cases) {
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            const std::vector<double> simd = run_strip(c, strikes, type);
            for (std::size_t i = 0; i < strikes.size(); ++i) {
                INFO("K=" << strikes[i] << " T=" << c.T << " sigma=" << c.sigma);
                CHECK_THAT(simd[i], WithinRel(bs_price(c.S, strikes[i], c.T, c.r, c.sigma, type), 1e-15));
            }
        }
    }
}

TEST_CASE("BS-11: strip lengths that are not a multiple of the SIMD width keep their edges correct",
          "[bs][unit]") {
    constexpr double kSentinel = -12345.0;

    SECTION("lengths 0, 1, 7 and 65 (not multiples of any SIMD width)") {
        for (const std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{65}}) {
            for (const OptionType type : {OptionType::Call, OptionType::Put}) {
                std::vector<double> strikes(n);
                for (std::size_t i = 0; i < n; ++i) strikes[i] = 60.0 + 1.5 * static_cast<double>(i);
                std::vector<double> out(n + 1, kSentinel);

                bs_price_strip(100.0, std::span<const double>(strikes), 1.0, 0.03, 0.25, type,
                               std::span<double>(out.data(), n));

                INFO("n=" << n << " call=" << (type == OptionType::Call));
                for (std::size_t i = 0; i < n; ++i) {
                    INFO("i=" << i << " K=" << strikes[i]);
                    CHECK_THAT(out[i], WithinRel(bs_price(100.0, strikes[i], 1.0, 0.03, 0.25, type), 1e-15));
                }
                CHECK(out[n] == kSentinel);  // out の範囲外には書かない
            }
        }
    }

    SECTION("invalid strikes (-5, NaN, 0) inside a block fall back to the scalar path") {
        // 不正なストライクを含むブロックは SIMD をやめてスカラで処理する。どのレーン幅でも
        // 先頭のブロックは健全なので、SIMD 経路とフォールバックの両方を通る。
        constexpr std::size_t n = 32;
        std::vector<double>   strikes(n);
        for (std::size_t i = 0; i < n; ++i) strikes[i] = 60.0 + 2.5 * static_cast<double>(i);
        strikes[11] = -5.0;
        strikes[19] = kNaN;
        strikes[26] = 0.0;

        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            std::vector<double> out(n + 1, kSentinel);
            bs_price_strip(100.0, std::span<const double>(strikes), 1.0, 0.03, 0.25, type,
                           std::span<double>(out.data(), n));
            INFO("call=" << (type == OptionType::Call));
            for (std::size_t i = 0; i < n; ++i) {
                INFO("i=" << i << " K=" << strikes[i]);
                check_matches_scalar(out[i], bs_price(100.0, strikes[i], 1.0, 0.03, 0.25, type));
            }
            CHECK(out[n] == kSentinel);
        }
    }

    SECTION("out shorter than strikes writes only out.size() elements") {
        constexpr std::size_t n_k = 20, n_out = 5;
        std::vector<double>   strikes(n_k);
        for (std::size_t i = 0; i < n_k; ++i) strikes[i] = 80.0 + 2.0 * static_cast<double>(i);

        for (const bool use_simd : {true, false}) {
            std::vector<double> out(n_out + 1, kSentinel);
            const std::span<const double> ks(strikes);
            const std::span<double>       os(out.data(), n_out);
            if (use_simd) {
                bs_price_strip(100.0, ks, 1.0, 0.03, 0.25, OptionType::Call, os);
            } else {
                bs_price_strip_scalar(100.0, ks, 1.0, 0.03, 0.25, OptionType::Call, os);
            }
            INFO("simd=" << use_simd);
            for (std::size_t i = 0; i < n_out; ++i) {
                INFO("i=" << i << " K=" << strikes[i]);
                CHECK_THAT(out[i], WithinRel(bs_price(100.0, strikes[i], 1.0, 0.03, 0.25, OptionType::Call),
                                             1e-15));
            }
            CHECK(out[n_out] == kSentinel);  // out.size() を超えて書かない
        }
    }
}
