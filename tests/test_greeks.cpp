// GREEKS-01..05 — scenes/greeks_model.hpp の契約 / 性質 / 決定性テスト（docs/03_tdd_spec.md §4.3）
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/core/models/gbm.hpp"
#include "quantviz/core/pricing/black_scholes.hpp"
#include "quantviz/scenes/greeks_model.hpp"

using Catch::Matchers::WithinRel;
using quantviz::bridge::Command;
using quantviz::scenes::GreeksModel;
using quantviz::scenes::GreeksSnapshot;

namespace {

constexpr double kDt = 1.0 / (252.0 * 390.0);  // 1 分足（年単位）

/// テスト用の設定。GREEKS-04 のために T を短く、r = 0 に取る（下のコメントを参照）。
GreeksModel::Config test_config() {
    GreeksModel::Config cfg;
    cfg.gbm         = {100.0, 0.0, 0.20};  // s0, mu, sigma
    cfg.r           = 0.0;
    cfg.sigma       = 0.20;
    cfg.maturity    = 0.10;
    cfg.strike_span = 0.40;
    cfg.seed        = 20260912;
    return cfg;
}

/// Snapshot の全要素が有限か。
bool all_finite(const GreeksSnapshot& s) {
    auto ok = [](double v) { return std::isfinite(v); };
    if (!(ok(s.t) && ok(s.spot) && ok(s.r) && ok(s.sigma) && ok(s.T))) return false;
    for (std::size_t i = 0; i < GreeksSnapshot::kStrikes; ++i) {
        if (!(ok(s.strikes[i]) && ok(s.price[i]) && ok(s.delta[i]) && ok(s.gamma[i]) && ok(s.vega[i]) &&
              ok(s.theta[i]) && ok(s.rho[i])))
            return false;
    }
    if (!std::all_of(s.grid_s.begin(), s.grid_s.end(), ok)) return false;
    if (!std::all_of(s.grid_t.begin(), s.grid_t.end(), ok)) return false;
    return std::all_of(s.gamma_surface.begin(), s.gamma_surface.end(), ok);
}

/// spot に最も近いストライクの index（ストライク軸は等間隔なので丸めで求まる）。
std::size_t nearest_strike_index(const GreeksSnapshot& s) {
    const double dk  = s.strikes[1] - s.strikes[0];
    const double raw = std::round((s.spot - s.strikes[0]) / dk);
    const double lim = static_cast<double>(GreeksSnapshot::kStrikes - 1);
    return static_cast<std::size_t>(std::clamp(raw, 0.0, lim));
}

std::size_t argmax_gamma(const GreeksSnapshot& s) {
    const auto it = std::max_element(s.gamma.begin(), s.gamma.end());
    return static_cast<std::size_t>(std::distance(s.gamma.begin(), it));
}

}  // namespace

TEST_CASE("GREEKS-01: satisfies the Model contract with a fixed-size POD snapshot", "[greeks][contract]") {
    STATIC_REQUIRE(quantviz::bridge::Model<GreeksModel>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<GreeksSnapshot>);
    STATIC_REQUIRE(std::is_standard_layout_v<GreeksSnapshot>);
    STATIC_REQUIRE(std::is_default_constructible_v<GreeksSnapshot>);
    // M1 の例外（docs の「Snapshot のサイズ方針」）: 固定格子を載せるので 128 B は超える。上限は 32 KiB。
    STATIC_REQUIRE(sizeof(GreeksSnapshot) <= 32 * 1024);
    STATIC_REQUIRE(GreeksSnapshot::kStrikes == 64);
    STATIC_REQUIRE(GreeksSnapshot::kGridS == 48);
    STATIC_REQUIRE(GreeksSnapshot::kGridT == 32);

    const GreeksModel m(test_config());
    const auto        s = m.snapshot();
    CHECK(s.seq == 0);  // 未ステップ
    CHECK(s.spot == 100.0);
    CHECK(s.t == 0.0);
}

TEST_CASE("GREEKS-02: the strike axis is ascending and equally spaced and every snapshot value is finite",
          "[greeks][property]") {
    GreeksModel m(test_config());

    auto check_axis = [](const GreeksSnapshot& s, double expected_span) {
        const double dk = s.strikes[1] - s.strikes[0];
        REQUIRE(dk > 0.0);
        for (std::size_t i = 0; i + 1 < GreeksSnapshot::kStrikes; ++i) {
            REQUIRE(s.strikes[i + 1] > s.strikes[i]);                             // 昇順
            REQUIRE_THAT(s.strikes[i + 1] - s.strikes[i], WithinRel(dk, 1e-12));  // 等間隔
        }
        // 軸は [S0(1-span), S0(1+span)]（S0 = 100）
        CHECK_THAT(s.strikes.front(), WithinRel(100.0 * (1.0 - expected_span), 1e-12));
        CHECK_THAT(s.strikes.back(), WithinRel(100.0 * (1.0 + expected_span), 1e-12));
        // Γ 格子: S ∈ [0.5 S0, 1.5 S0]、T ∈ (0, T_max]
        CHECK_THAT(s.grid_s.front(), WithinRel(50.0, 1e-12));
        CHECK_THAT(s.grid_s.back(), WithinRel(150.0, 1e-12));
        CHECK(s.grid_t.front() > 0.0);
        CHECK_THAT(s.grid_t.back(), WithinRel(s.T, 1e-12));
    };

    SECTION("fresh model") {
        const auto s = m.snapshot();
        check_axis(s, 0.40);
        CHECK(all_finite(s));
    }

    SECTION("after stepping") {
        for (int i = 0; i < 500; ++i) m.step(kDt);
        const auto s = m.snapshot();
        check_axis(s, 0.40);
        CHECK(all_finite(s));
        CHECK(s.seq == 500);
    }

    SECTION("the axis follows the strike span and stays ascending/equally spaced") {
        for (int i = 0; i < 10; ++i) m.step(kDt);
        const auto before = m.snapshot();
        m.apply(Command::set_param(GreeksModel::kStrikeSpan, 0.75));
        m.step(kDt);
        const auto after = m.snapshot();
        check_axis(before, 0.40);
        check_axis(after, 0.75);
        CHECK(all_finite(after));
    }

    SECTION("degenerate parameters are clamped, never producing NaN") {
        m.apply(Command::set_param(GreeksModel::kSigma, -1.0));      // sigma >= 0
        m.apply(Command::set_param(GreeksModel::kMaturity, 0.0));    // T > 0
        m.apply(Command::set_param(GreeksModel::kStrikeSpan, 5.0));  // span <= 0.9
        m.step(kDt);
        const auto s = m.snapshot();
        CHECK(s.sigma >= 0.0);
        CHECK(s.T > 0.0);
        CHECK_THAT(s.strikes.back(), WithinRel(100.0 * 1.9, 1e-12));
        CHECK(all_finite(s));

        // クランプではなく「無視」される入力（apply() だけを通る経路）。kSpotJump の非正・非有限値と
        // kRate の NaN は状態を一切動かさないこと ― スポットが動くのは GBM のステップの分だけ。
        constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
        m.apply(Command::set_param(GreeksModel::kRate, 0.07));  // 既定 0.0 のままだと「NaN → 0 に戻す」誤実装が通る
        m.step(kDt);
        const auto   armed       = m.snapshot();
        const double r_before    = armed.r;
        const double spot_before = armed.spot;
        REQUIRE(r_before == 0.07);
        m.apply(Command::set_param(GreeksModel::kSpotJump, 0.0));
        m.apply(Command::set_param(GreeksModel::kSpotJump, -2.0));
        m.apply(Command::set_param(GreeksModel::kSpotJump, kNan));
        m.apply(Command::set_param(GreeksModel::kRate, kNan));
        m.step(kDt);
        const auto after = m.snapshot();
        CHECK(after.r == r_before);        // NaN の r は捨てられ、直前の値が残る
        CHECK(after.spot != spot_before);  // ショックは効かないが GBM は進む
        CHECK_THAT(after.spot, WithinRel(spot_before, 1e-2));  // 1 ステップの GBM 変化は ~0.06 %。×0 / ×−2 なら破綻する
        CHECK(after.spot > 0.0);
        CHECK(all_finite(after));
    }
}

TEST_CASE("GREEKS-03: SetParam(sigma) is reflected in the snapshot immediately without advancing seq, and persists across steps",
          "[greeks][unit]") {
    GreeksModel m(test_config());
    for (int i = 0; i < 20; ++i) m.step(kDt);
    const auto before = m.snapshot();

    SECTION("apply() alone recomputes the snapshot with the new sigma and keeps seq (self-consistent)") {
        // R10: 一時停止中の SetParam 直後に Runner が Snapshot を再送するので、その Snapshot が
        // 新しい σ で計算し直された配列と真値フィールドを持っていなければならない。
        m.apply(Command::set_param(GreeksModel::kSigma, 0.50));
        const auto s = m.snapshot();
        CHECK(s.seq == before.seq);  // パラメータ変更はステップを進めない
        CHECK(s.sigma == 0.50);
        CHECK(s.spot == before.spot);  // スポットは動かない
        CHECK(s.gamma != before.gamma);
        for (std::size_t i = 0; i < GreeksSnapshot::kStrikes; ++i) {
            const auto g = quantviz::core::bs_greeks(s.spot, s.strikes[i], s.T, s.r, s.sigma,
                                                     quantviz::core::OptionType::Call);
            REQUIRE(s.gamma[i] == g.gamma);  // 配列と真値フィールドが同じパラメータで計算されている
        }
    }

    SECTION("the next step keeps the new sigma and recomputes the strip at the stepped spot") {
        m.apply(Command::set_param(GreeksModel::kSigma, 0.50));
        m.step(kDt);
        const auto s = m.snapshot();
        CHECK(s.seq == before.seq + 1);
        CHECK(s.sigma == 0.50);
        CHECK(s.gamma != before.gamma);
        // 各ストライクの Greeks は core::bs_greeks（コール）と厳密に一致していること。
        // price だけは別実装（ベクトル化された bs_price_strip）なので、core が約束するのは
        // BS-10 の「相対 1e-15」までで、bit 一致は FMA 融合の文脈に依存する。ここも 1e-15 で見る。
        for (std::size_t i = 0; i < GreeksSnapshot::kStrikes; ++i) {
            const auto g = quantviz::core::bs_greeks(s.spot, s.strikes[i], s.T, s.r, s.sigma,
                                                     quantviz::core::OptionType::Call);
            REQUIRE_THAT(s.price[i], WithinRel(g.price, 1e-15));
            REQUIRE(s.delta[i] == g.delta);
            REQUIRE(s.gamma[i] == g.gamma);
            REQUIRE(s.vega[i] == g.vega);
            REQUIRE(s.theta[i] == g.theta);
            REQUIRE(s.rho[i] == g.rho);
        }
    }

    SECTION("rate and maturity behave the same way") {
        m.apply(Command::set_param(GreeksModel::kRate, 0.08));
        m.apply(Command::set_param(GreeksModel::kMaturity, 1.5));
        CHECK(m.snapshot().r == 0.08);   // 即時反映
        CHECK(m.snapshot().T == 1.5);
        CHECK(m.snapshot().seq == before.seq);
        m.step(kDt);
        const auto s = m.snapshot();
        CHECK(s.r == 0.08);
        CHECK(s.T == 1.5);
        CHECK(s.seq == before.seq + 1);
    }

    SECTION("an unknown param_id is ignored") {
        m.apply(Command::set_param(999, 1.0));
        m.step(kDt);
        const auto s = m.snapshot();
        CHECK(s.sigma == before.sigma);
        CHECK(s.r == before.r);
        CHECK(s.T == before.T);
    }

    SECTION("Reset rewinds the path but keeps the parameters") {
        m.apply(Command::set_param(GreeksModel::kSigma, 0.33));
        m.apply(Command::reset());
        const auto s = m.snapshot();
        CHECK(s.seq == 0);
        CHECK(s.spot == 100.0);
        CHECK(s.t == 0.0);
        CHECK(s.sigma == 0.33);
        CHECK(all_finite(s));
    }
}

TEST_CASE("GREEKS-04: the largest gamma sits on the strike nearest to the spot (+/- one grid step)",
          "[greeks][property]") {
    // Gamma(K) = phi(d1)/(S sigma sqrt(T)) は d1 = 0、すなわち K* = S exp((r + sigma^2/2) T) で最大。
    // テスト設定は r = 0, sigma = 0.2, T = 0.1 なので K*/S = exp(0.002) = 1.002、ストライク間隔は
    // 80/63 = 1.27 なので、ずれは 0.2 / 1.27 = 0.16 グリッド（半グリッド未満）。よって argmax は
    // 「S に最も近いストライク」と ±1 以内で一致する。
    GreeksModel m(test_config());
    m.step(kDt);

    const double jumps[] = {1.0, 1.10, 1.10, 0.80, 0.85, 1.25};
    for (const double j : jumps) {
        m.apply(Command::set_param(GreeksModel::kSpotJump, j));
        m.step(kDt);
        const auto s = m.snapshot();

        const std::size_t peak    = argmax_gamma(s);
        const std::size_t nearest = nearest_strike_index(s);
        INFO("spot = " << s.spot << ", peak K = " << s.strikes[peak]
                       << ", nearest K = " << s.strikes[nearest]);
        REQUIRE(s.spot > s.strikes.front());
        REQUIRE(s.spot < s.strikes.back());
        const std::ptrdiff_t diff = static_cast<std::ptrdiff_t>(peak) - static_cast<std::ptrdiff_t>(nearest);
        REQUIRE(std::abs(diff) <= 1);
    }
}

TEST_CASE("GREEKS-05: the spot path is bit-identical to a bare Gbm with the same seed (dual-run)",
          "[greeks][determinism]") {
    const auto          cfg = test_config();
    GreeksModel         m(cfg);
    quantviz::core::Gbm bare(cfg.gbm, cfg.seed);

    for (int i = 0; i < 2000; ++i) {
        m.step(kDt);
        bare.step(kDt);
        const auto s = m.snapshot();
        REQUIRE(s.spot == bare.spot());  // bit 一致
        REQUIRE(s.t == bare.time());
        REQUIRE(s.seq == static_cast<std::uint64_t>(i + 1));
    }

    // Reset(seed) は同じ seed のパスを最初から再生する。
    m.apply(Command::reset(cfg.seed));
    quantviz::core::Gbm replay(cfg.gbm, cfg.seed);
    for (int i = 0; i < 200; ++i) {
        m.step(kDt);
        replay.step(kDt);
        REQUIRE(m.snapshot().spot == replay.spot());
    }
}
