// LSM-xx — core/pricing/lsm.hpp（+ core/math/linsolve.hpp）の仕様テスト
// アンチセティック対のパス生成、American put の価格が CN（FDM-06 の設定）の 4 SE 内に入ること、
// 同パスの European MC 以上、基底数に対する頑健性、seed 決定性、σ = 0 の決定的ペイオフ、SE の 1/√N 則を
// 検証する。正規方程式のソルバ（linsolve）には独立した仕様 ID がないので LSM-01 の SECTION で検査する。
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "quantviz/core/math/linsolve.hpp"
#include "quantviz/core/pricing/black_scholes.hpp"
#include "quantviz/core/pricing/fdm_cn.hpp"
#include "quantviz/core/pricing/lsm.hpp"
#include "quantviz/core/rng.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::core::FdmCn;
using quantviz::core::FdmGrid;
using quantviz::core::FdmParams;
using quantviz::core::kLinsolveMaxN;
using quantviz::core::linsolve;
using quantviz::core::Lsm;
using quantviz::core::LsmBasis;
using quantviz::core::LsmParams;
using quantviz::core::LsmResult;
using quantviz::core::OptionType;
using quantviz::core::Rng;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// 標準ケース: S0 = K = 100, T = 1, r = 0.05, σ = 0.2（FDM-06 と同じ。American put ≈ 6.09、European 5.5735）。
constexpr double kS0    = 100.0;
constexpr double kK     = 100.0;
constexpr double kT     = 1.0;
constexpr double kR     = 0.05;
constexpr double kSigma = 0.2;

LsmParams put_params(std::size_t n_paths, std::size_t n_steps, std::size_t n_basis, std::uint64_t seed,
                     LsmBasis basis = LsmBasis::Laguerre) {
    LsmParams p{};
    p.s0      = kS0;
    p.K       = kK;
    p.T       = kT;
    p.r       = kR;
    p.sigma   = kSigma;
    p.type    = OptionType::Put;
    p.n_paths = n_paths;
    p.n_steps = n_steps;
    p.n_basis = n_basis;
    p.basis   = basis;
    p.seed    = seed;
    return p;
}

/// init して満期から t = 0 まで全時点の後ろ向き回帰を回す。
Lsm solve_all(const LsmParams& p) {
    Lsm l;
    l.init(p);
    while (l.step_backward()) {
    }
    return l;
}

/// LSM-02 の参照価格: FDM-06 の設定（American put, S_max = 4K, N = M = 200）の CN + PSOR。実測 6.0848。
double fdm_american_put() {
    FdmParams p{};
    p.K        = kK;
    p.T        = kT;
    p.r        = kR;
    p.sigma    = kSigma;
    p.q        = 0.0;
    p.type     = OptionType::Put;
    p.american = true;
    FdmCn f;
    f.init(FdmGrid{4.0 * kK, 200, 200}, p);
    while (f.step_backward()) {
    }
    return f.value_at(kS0);
}

}  // namespace

TEST_CASE("LSM-01: antithetic pairs: path 2j uses +Z and path 2j+1 uses -Z, so a pair's increments sum to 0",
          "[lsm][unit]") {
    constexpr std::size_t n_paths = 8, n_steps = 5;
    const LsmParams       p = put_params(n_paths, n_steps, 3, 7);

    SECTION("generate_paths writes Z once per pair and step and builds both paths from it (bit-exact)") {
        std::vector<double> paths(n_paths * (n_steps + 1), kNaN);
        std::vector<double> normals((n_paths / 2) * n_steps, kNaN);
        REQUIRE(Lsm::generate_paths(p, paths, normals));

        const double dt    = kT / static_cast<double>(n_steps);
        const double drift = (kR - 0.5 * kSigma * kSigma) * dt;
        const double vol   = kSigma * std::sqrt(dt);
        for (std::size_t j = 0; j < n_paths / 2; ++j) {
            const double* a = &paths[(2 * j) * (n_steps + 1)];
            const double* b = &paths[(2 * j + 1) * (n_steps + 1)];
            CHECK(a[0] == kS0);
            CHECK(b[0] == kS0);
            for (std::size_t k = 0; k < n_steps; ++k) {
                const double z = normals[j * n_steps + k];
                INFO("pair " << j << " step " << k << " z=" << z);
                REQUIRE(std::isfinite(z));
                // 同じ Z から同じ式で再構成すればビット一致する（+Z / −Z の使い分けの厳密な検査）
                CHECK(a[k + 1] == Lsm::gbm_step(a[k], drift, vol, z));
                CHECK(b[k + 1] == Lsm::gbm_step(b[k], drift, vol, -z));
                // 対数リターンはドリフトを中心に鏡像: lr_a + lr_b = 2 drift（log/exp の丸めだけが残る）
                const double lr_a = std::log(a[k + 1] / a[k]);
                const double lr_b = std::log(b[k + 1] / b[k]);
                CHECK_THAT(lr_a + lr_b, WithinAbs(2.0 * drift, 1e-12));
            }
        }
        // Z は退化していない（全部 0 ではない）
        double sum_sq = 0.0;
        for (const double z : normals) sum_sq += z * z;
        CHECK(sum_sq > 0.0);
    }

    SECTION("generate_paths rejects an odd n_paths or a span that is too small and then writes nothing") {
        std::vector<double> paths(n_paths * (n_steps + 1), kNaN);
        LsmParams           odd = p;
        odd.n_paths             = 7;
        CHECK_FALSE(Lsm::generate_paths(odd, paths, {}));
        std::vector<double> small(n_paths * (n_steps + 1) - 1, kNaN);
        CHECK_FALSE(Lsm::generate_paths(p, small, {}));
        std::vector<double> normals_small((n_paths / 2) * n_steps - 1, kNaN);
        CHECK_FALSE(Lsm::generate_paths(p, paths, normals_small));
        for (const double v : paths) CHECK(std::isnan(v));
        for (const double v : small) CHECK(std::isnan(v));
        // normals は省略できる
        CHECK(Lsm::generate_paths(p, paths, {}));
    }

    SECTION("init uses the generator: path(i) rows and spots_at(k) columns are two layouts of one path set") {
        std::vector<double> paths(n_paths * (n_steps + 1), kNaN);
        REQUIRE(Lsm::generate_paths(p, paths, {}));
        Lsm l;
        l.init(p);
        CHECK(l.remaining() == n_steps);
        CHECK(l.current_step() == n_steps);
        CHECK(l.time() == kT);
        for (std::size_t i = 0; i < n_paths; ++i) {
            const std::span<const double> pi = l.path(i);
            REQUIRE(pi.size() == n_steps + 1);
            for (std::size_t k = 0; k <= n_steps; ++k) {
                INFO("i=" << i << " k=" << k);
                CHECK(pi[k] == paths[i * (n_steps + 1) + k]);
                REQUIRE(l.spots_at(k).size() == n_paths);
                CHECK(l.spots_at(k)[i] == pi[k]);
            }
        }
        CHECK(l.path(n_paths).empty());
        CHECK(l.spots_at(n_steps + 1).empty());
        // 満期では未行使: exercise_step は n_steps、継続価値はまだフィットされていない
        for (std::size_t i = 0; i < n_paths; ++i) CHECK(l.exercise_step(i) == n_steps);
        CHECK(std::isnan(l.continuation_value(kS0)));
        REQUIRE(l.continuation_coeffs().size() == 3);
    }

    SECTION("init sanitises: odd n_paths rounds up; n_paths/n_steps/n_basis are clamped to their ranges") {
        Lsm       l;
        LsmParams q = p;
        q.n_paths   = 7;
        q.n_basis   = 20;
        q.n_steps   = 0;
        l.init(q);
        CHECK(l.params().n_paths == 8);
        CHECK(l.params().n_basis == Lsm::kMaxBasis);
        CHECK(l.params().n_steps == 1);
        CHECK(l.path(7).size() == 2);
        q.n_basis = 0;
        l.init(q);
        CHECK(l.params().n_basis == 1);
        // 上限（init の確保量 2·N·(M+1)·8 B を有界にする）。片方ずつ最小の相方で試す（両方最大だと 3 GB）。
        q.n_paths = Lsm::kMaxPaths + 2;
        q.n_steps = 1;
        l.init(q);
        CHECK(l.params().n_paths == Lsm::kMaxPaths);
        CHECK(l.spots_at(0).size() == Lsm::kMaxPaths);
        q.n_paths = 2;
        q.n_steps = Lsm::kMaxSteps + 1;
        l.init(q);
        CHECK(l.params().n_steps == Lsm::kMaxSteps);
        CHECK(l.path(0).size() == Lsm::kMaxSteps + 1);
        CHECK(l.remaining() == Lsm::kMaxSteps);
    }

    SECTION("quantile_at: q = 0 / 1 are the column min / max, q = 0.5 the median; invalid input gives NaN") {
        Lsm l;
        l.init(p);
        for (std::size_t k = 0; k <= n_steps; ++k) {
            const std::span<const double> col = l.spots_at(k);
            std::vector<double>           sorted(col.begin(), col.end());
            std::sort(sorted.begin(), sorted.end());
            INFO("k=" << k);
            CHECK(l.quantile_at(k, 0.0) == sorted.front());
            CHECK(l.quantile_at(k, 1.0) == sorted.back());
            // 最近接順位: index = round(q (N − 1)) = round(3.5) = 4
            CHECK(l.quantile_at(k, 0.5) == sorted[4]);
        }
        CHECK(std::isnan(l.quantile_at(n_steps + 1, 0.5)));
        CHECK(std::isnan(l.quantile_at(0, kNaN)));
        CHECK(std::isnan(l.quantile_at(0, -0.1)));
        CHECK(std::isnan(l.quantile_at(0, 1.1)));
    }

    // 仕様 ID なし: linsolve は LSM の正規方程式のソルバなのでここで検査する（03_tdd_spec.md には独立行がない）。
    SECTION("the normal-equation solver matches a hand-computed 3x3 system and rejects singular matrices") {
        std::array<std::size_t, kLinsolveMaxN> piv{};

        // 2x + y − z = 8, −3x − y + 2z = −11, −2x + y + 2z = −3  →  (2, 3, −1)。先頭ピボットは 2 行目の −3。
        std::array<double, 9> a3{2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0};
        std::array<double, 3> b3{8.0, -11.0, -3.0};
        REQUIRE(linsolve(3, a3, b3, piv));
        CHECK_THAT(b3[0], WithinAbs(2.0, 1e-12));
        CHECK_THAT(b3[1], WithinAbs(3.0, 1e-12));
        CHECK_THAT(b3[2], WithinAbs(-1.0, 1e-12));

        // 対角に 0: 行交換なしでは割れない。[[0,1],[1,0]] x = (1, 2) → x = (2, 1)（交換後は単位行列なので厳密）
        std::array<double, 4> a2{0.0, 1.0, 1.0, 0.0};
        std::array<double, 2> b2{1.0, 2.0};
        REQUIRE(linsolve(2, a2, b2, piv));
        CHECK(b2[0] == 2.0);
        CHECK(b2[1] == 1.0);
        CHECK(piv[0] == 1);

        // 特異（ランク 1）
        std::array<double, 4> sing{1.0, 2.0, 2.0, 4.0};
        std::array<double, 2> bs{1.0, 2.0};
        CHECK_FALSE(linsolve(2, sing, bs, piv));

        // 非有限な要素
        std::array<double, 4> nan_a{1.0, kNaN, 0.0, 1.0};
        std::array<double, 2> nb{1.0, 1.0};
        CHECK_FALSE(linsolve(2, nan_a, nb, piv));
        std::array<double, 4> inf_a{1.0, 0.0, 0.0, std::numeric_limits<double>::infinity()};
        CHECK_FALSE(linsolve(2, inf_a, nb, piv));

        // 丸めで 1e-16 程度のピボットが残る「ほぼランク 1」の正規行列 n φφᵀ（σ = 0 の LSM で実際に現れる形）は
        // rel_tol で特異扱いにできる。rel_tol = 0 では厳密な 0 だけを特異とする。
        const std::array<double, 3> phi{1.0, 0.8, 0.64};
        std::array<double, 9>       rank1{};
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j) rank1[i * 3 + j] = 37.0 * phi[i] * phi[j];
        std::array<double, 3> br{37.0 * 5.0, 37.0 * 4.0, 37.0 * 3.2};
        CHECK_FALSE(linsolve(3, rank1, br, piv, 1e-13));

        // サイズ: n = 0 は自明に真、n > kLinsolveMaxN と足りない span は偽（何も書かない）
        CHECK(linsolve(0, a3, b3, piv));
        std::vector<double>      big((kLinsolveMaxN + 1) * (kLinsolveMaxN + 1), 1.0);
        std::vector<double>      bigb(kLinsolveMaxN + 1, 1.0);
        std::vector<std::size_t> bigp(kLinsolveMaxN + 1, 0);
        CHECK_FALSE(linsolve(kLinsolveMaxN + 1, big, bigb, bigp));
        std::array<double, 8> short_a{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
        std::array<double, 3> short_b{1.0, 2.0, 3.0};
        CHECK_FALSE(linsolve(3, short_a, short_b, piv));
        CHECK(short_b[0] == 1.0);
        std::array<std::size_t, 2> short_piv{};
        CHECK_FALSE(linsolve(3, a3, b3, short_piv));

        // ランダムな 8×8 対角優位系: 残差が相対 1e-12
        constexpr std::size_t n = 8;
        Rng                   rng(2024);
        std::vector<double>   a(n * n), b(n), a_copy, b_copy;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) a[i * n + j] = rng.normal();
            a[i * n + i] += 10.0;
            b[i] = rng.normal();
        }
        a_copy = a;
        b_copy = b;
        REQUIRE(linsolve(n, a, b, piv, 1e-13));
        for (std::size_t i = 0; i < n; ++i) {
            double ax = 0.0, scale = std::abs(b_copy[i]);
            for (std::size_t j = 0; j < n; ++j) {
                ax += a_copy[i * n + j] * b[j];
                scale += std::abs(a_copy[i * n + j] * b[j]);
            }
            INFO("row " << i);
            CHECK_THAT(ax, WithinAbs(b_copy[i], 1e-12 * scale));
        }
    }
}

TEST_CASE("LSM-02: American put, N=20000, M=50, Laguerre 4: price within 4 SE of the CN reference",
          "[lsm][statistical]") {
    // S0 = K = 100, T = 1, r = 0.05, σ = 0.2（FDM-06 の設定）。期待値 = CN + PSOR（N = M = 200）6.0848
    // （真値 ≈ 6.0896、二項木 N = 800 で 6.0899）、SE ≈ 0.030、許容 4 SE（§2.4 のモンテカルロ規約）。
    // 仕様行の「95 % CI」は統合時に「4 SE」へ改める: 95 % CI は無バイアスでも seed の 5 % で落ちる構造で、
    // CI の 3 OS は std::normal_distribution の系列が異なるので seed 固定でも赤になりうる。
    // LSM は 50 行使日の Bermudan を in-sample の回帰方策で評価する。ref との差（400 seed の実測）:
    //   Laguerre 3: 平均 −0.0257（1.96 SE では 358/400 しか通らず、最悪 |z| = 3.95）
    //   Laguerre 4: 平均 −0.0049 = 0.16 SE（最悪 |z| = 2.98 → 4 SE で全 seed 通過）
    // 内訳は行使日の離散化（CRR 木 n = 10000: American 6.0903 vs Bermudan(50) 6.0786 → −0.012）と、方策の損から
    // in-sample（同じパスで方策推定と評価）の上方バイアスを引いた残差。よって回帰子は 4 個 = 定数 + Laguerre
    // 3 本（Longstaff–Schwartz 2001 の put と同じ構成）。方策が壊れていれば price ≈ European（5.58）で ref から
    // 17 SE 離れるので、4 SE でも検出できる（下の premium 判定が同じことを直接見る）。
    // SE はアンチセティック対の対平均を標本単位とする（対内の負相関を織り込む）。方策の推定誤差は SE に
    // 含まないが、400 seed の価格の sd は報告 SE の 1.03 倍で実用上無視できる。
    // 実測（seed 20260912, libstdc++）: price = 6.0648, SE = 0.0304, european = 5.5810（BS 5.5735）,
    // z = −0.66（Laguerre 3 のときは 6.0351, z = −1.63 だった）。
    const double ref = fdm_american_put();
    CHECK_THAT(ref, WithinAbs(6.0848, 1e-3));

    const Lsm       l   = solve_all(put_params(20000, 50, 4, 20260912));
    const LsmResult res = l.result();
    INFO("ref=" << ref << " price=" << res.price << " se=" << res.std_error
                << " european=" << res.european_price << " exercised=" << res.exercised_paths
                << " z=" << (res.price - ref) / res.std_error);
    CHECK(l.remaining() == 0);
    CHECK(res.std_error > 0.0);
    CHECK(res.std_error < 0.1);  // 実測 0.030（対平均の sd ≈ 3.0, 10000 対）
    CHECK_THAT(res.price, WithinAbs(ref, 4.0 * res.std_error));
    // 早期行使プレミアム ≈ 0.50 ≈ 17 SE が出ていること（方策が壊れて European に潰れていれば落ちる）
    CHECK(res.price - res.european_price > 10.0 * res.std_error);
    // 同パスの European MC は BS 5.5735 の 4 SE 内（実測 SE 0.048 → 0.05 を使い、帯 = 4 × 0.05 = 0.20）
    CHECK_THAT(res.european_price,
               WithinAbs(quantviz::core::bs_price(kS0, kK, kT, kR, kSigma, OptionType::Put), 4.0 * 0.05));
}

TEST_CASE("LSM-03: the LSM price is at least the European MC price on the same paths (equal before any step)",
          "[lsm][property]") {
    // 早期行使は満期行使を含む上位集合なので、in-sample の LSM 価格は同パスの European 平均以上になる
    // （方策が悪くても ITM でしか行使せず、行使しない選択が常に残る。差 ≈ 0.5 は SE の 10 倍以上）。
    // 実測（seed 1/2/3）: price 6.036 / 6.093 / 6.037, european 5.559 / 5.556 / 5.517, 行使 7382 / 7644 / 7642 本。
    for (const std::uint64_t seed : {std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{3}}) {
        const LsmParams p = put_params(20000, 50, 3, seed);
        Lsm             l;
        l.init(p);
        const LsmResult at_maturity = l.result();
        CHECK(at_maturity.price == at_maturity.european_price);  // 満期だけ: 同じ式、ビット一致
        CHECK(at_maturity.exercised_paths == 0);

        while (l.step_backward()) {
        }
        const LsmResult res = l.result();
        INFO("seed=" << seed << " price=" << res.price << " european=" << res.european_price
                     << " exercised=" << res.exercised_paths);
        CHECK(res.price >= res.european_price);
        CHECK(res.price > res.european_price + 0.3);  // 早期行使プレミアム ≈ 0.5
        CHECK(res.exercised_paths > 0);
        CHECK(res.exercised_paths < p.n_paths);
        // 行使したパスの行使時点は満期より前で、ペイオフはその時点の本源的価値と一致する
        std::size_t counted = 0;
        for (std::size_t i = 0; i < p.n_paths; ++i) {
            const std::size_t k = l.exercise_step(i);
            if (k < p.n_steps) {
                ++counted;
                CHECK(l.path(i)[k] < kK);
            }
        }
        CHECK(counted == res.exercised_paths);
    }
}

TEST_CASE("LSM-04: going from 2 to 5 basis functions moves the price by less than 4 combined SE",
          "[lsm][statistical]") {
    // 同じ seed（同じパス）で n_basis = 2 と 5 の価格を比べる。差は対（paired）なので標本誤差はほとんど
    // 打ち消え、残るのはほぼ純粋な方策バイアス（400 seed の実測: 平均 +0.064、sd 0.020。回帰子 2 個 =
    // 定数 + e^{−x/2} は継続価値のほぼ線形近似で方策がやや悪く価格が低め）。√(SE₂² + SE₅²) は非対の標本誤差
    // であって「対の差の CI」ではないので、ここでは 4 倍（≈ 0.18）を「価格水準に対する頑健性の帯」として
    // 使う（1.96 倍 ≈ 0.087 だと 200 seed 中 175 しか通らない = バイアスが帯と同じ大きさ）。
    // 3 個以上ではほぼ変わらない（Laguerre 3/4/5/8: 6.068 / 6.072 / 6.076 / 6.075）。Laguerre 5 は狭い ITM 区間で
    // 正規方程式が特異に近く、半数の時点でランク 4 に打ち切られる（lsm.hpp 冒頭）。
    // 実測（seed 777）: p2 = 6.0197 (SE 0.0334), p5 = 6.0759 (SE 0.0296), diff = 0.056, 帯 4 × 0.0446 = 0.179。
    const LsmResult r2 = solve_all(put_params(20000, 50, 2, 777)).result();
    const LsmResult r5 = solve_all(put_params(20000, 50, 5, 777)).result();
    const double    combined_se = std::sqrt(r2.std_error * r2.std_error + r5.std_error * r5.std_error);
    INFO("p2=" << r2.price << " se2=" << r2.std_error << " p5=" << r5.price << " se5=" << r5.std_error
               << " diff=" << r5.price - r2.price << " band=" << 4.0 * combined_se);
    CHECK(r2.std_error > 0.0);
    CHECK(r5.std_error > 0.0);
    CHECK_THAT(r2.price, WithinAbs(r5.price, 4.0 * combined_se));
    CHECK(r2.price >= r2.european_price);
    CHECK(r5.price >= r5.european_price);
}

TEST_CASE("LSM-05: same seed and parameters give a bit-identical price, paths and fit; another seed differs",
          "[lsm][determinism]") {
    // フィットの比較は remaining() == 1（時点 k = 1 の回帰直後）で行う。ATM では k = 0 に ITM パスが無く係数が
    // 全て 0 になるので、最終状態での係数比較は空虚（0 == 0）。k = 1 は実測でランク 3、ITM ≈ 1900 本、
    // 係数 ≈ [696, −1138, 648]（O(700) が打ち消し合って O(1) の継続価値になる）。ITM 本数や係数の値そのものは
    // std::normal_distribution の実装（OS ごとに異なる）に依存するので固定しない。
    const LsmParams p = put_params(4000, 20, 3, 99);
    Lsm             a, b;
    a.init(p);
    b.init(p);
    while (a.remaining() > 1) {
        REQUIRE(a.step_backward());
        REQUIRE(b.step_backward());
    }
    REQUIRE(a.remaining() == 1);
    REQUIRE(b.remaining() == 1);
    CHECK(a.last_fit_rank() == 3);
    CHECK(a.last_itm_paths() > 1000);
    CHECK(a.last_fit_rank() == b.last_fit_rank());
    CHECK(a.last_itm_paths() == b.last_itm_paths());
    REQUIRE(a.continuation_coeffs().size() == 3);
    for (std::size_t j = 0; j < 3; ++j) {
        INFO("j=" << j << " coeff=" << a.continuation_coeffs()[j]);
        CHECK(a.continuation_coeffs()[j] != 0.0);
        CHECK(a.continuation_coeffs()[j] == b.continuation_coeffs()[j]);
    }
    CHECK(std::isfinite(a.continuation_value(90.0)));
    CHECK(a.continuation_value(90.0) == b.continuation_value(90.0));

    REQUIRE(a.step_backward());
    REQUIRE(b.step_backward());
    CHECK_FALSE(a.step_backward());
    const LsmResult ra = a.result(), rb = b.result();
    CHECK(ra.price == rb.price);
    CHECK(ra.std_error == rb.std_error);
    CHECK(ra.european_price == rb.european_price);
    CHECK(ra.exercised_paths == rb.exercised_paths);
    for (std::size_t i = 0; i < p.n_paths; ++i) {
        const std::span<const double> pa = a.path(i), pb = b.path(i);
        REQUIRE(pa.size() == pb.size());
        for (std::size_t k = 0; k < pa.size(); ++k) {
            if (pa[k] != pb[k]) {
                FAIL("path " << i << " step " << k << " differs");
            }
        }
        CHECK(a.exercise_step(i) == b.exercise_step(i));
    }
    CHECK(a.quantile_at(10, 0.5) == b.quantile_at(10, 0.5));

    SECTION("re-init on the same object replays the same result") {
        Lsm c;
        c.init(p);
        while (c.step_backward()) {
        }
        c.init(p);
        while (c.step_backward()) {
        }
        CHECK(c.result().price == ra.price);
        CHECK(c.result().std_error == ra.std_error);
    }

    SECTION("a different seed gives a different price") {
        LsmParams q = p;
        q.seed      = 100;
        CHECK(solve_all(q).result().price != ra.price);
    }
}

TEST_CASE("LSM-06: sigma = 0 is deterministic: put S0=80, K=100 is exactly 20, SE 0, all exercised at t=0",
          "[lsm][numeric]") {
    // σ = 0 では全パスが S_k = S0 e^{r k dt} で一致し、回帰の正規方程式はランク 1 になる。ソルバは特異を
    // 返し、継続価値は ITM パスの割引キャッシュフローの平均（定数）にフォールバックする。
    // put（S0 = 80 < K = 100）: 時点 k の本源的価値 h_k = 100 − 80 e^{r k dt} は k が小さいほど大きく、継続価値
    // c_k = h_{k+1} e^{−r dt} = 100 e^{−r dt} − 80 e^{r k dt} < h_k なので毎時点で行使が選ばれ、最後に t = 0 で
    // 全パスが行使される。価格は K − S0 = 20（浮動小数でも厳密）、対平均が全て 20 なので SE は厳密に 0。
    constexpr std::size_t n_paths = 200, n_steps = 10;
    LsmParams             p = put_params(n_paths, n_steps, 3, 5);
    p.s0                    = 80.0;
    p.sigma                 = 0.0;

    SECTION("put: exact 20, SE 0, all exercised at step 0; European = 100 e^{-rT} - 80") {
        Lsm l;
        l.init(p);
        const double dt = kT / static_cast<double>(n_steps);
        for (std::size_t i = 0; i < n_paths; ++i)
            for (std::size_t k = 0; k <= n_steps; ++k) {
                INFO("i=" << i << " k=" << k);
                CHECK_THAT(l.path(i)[k], WithinRel(80.0 * std::exp(kR * dt * static_cast<double>(k)), 1e-12));
            }
        // 各時点で全パスが「その時点」で行使し直す
        while (l.step_backward()) {
            const std::size_t k = l.current_step();
            INFO("k=" << k);
            CHECK(l.last_itm_paths() == n_paths);
            CHECK(l.last_fit_fallback());
            CHECK(l.last_fit_rank() == 1);  // ランク 1 = 定数（平均）。係数は [平均, 0, 0]
            CHECK(l.continuation_coeffs()[1] == 0.0);
            CHECK(l.continuation_coeffs()[2] == 0.0);
            for (std::size_t i = 0; i < n_paths; ++i) CHECK(l.exercise_step(i) == k);
            CHECK_THAT(l.continuation_value(l.spots_at(k)[0]),
                       WithinRel((100.0 - 80.0 * std::exp(kR * dt * static_cast<double>(k + 1))) *
                                     std::exp(-kR * dt),
                                 1e-12));
        }
        const LsmResult res = l.result();
        CHECK(res.price == 20.0);
        CHECK(res.std_error == 0.0);
        CHECK(res.exercised_paths == n_paths);
        CHECK_THAT(res.european_price, WithinRel(100.0 * std::exp(-kR * kT) - 80.0, 1e-12));
    }

    SECTION("call: S0=120 is never exercised early; price = European = 120 - 100 e^{-rT}") {
        // h_k = 120 e^{r k dt} − 100 の割引値 120 − 100 e^{−r k dt} は k について増加するので満期まで待つ。
        // 全パスの値が同一なので SE は丸めの範囲（対平均の和 Σz と n·z の 1 ulp 差 → SE ~ 1e-16）で 0。
        p.s0   = 120.0;
        p.type = OptionType::Call;
        const Lsm       l   = solve_all(p);
        const LsmResult res = l.result();
        CHECK(res.exercised_paths == 0);
        CHECK(res.price == res.european_price);
        CHECK_THAT(res.price, WithinRel(120.0 - 100.0 * std::exp(-kR * kT), 1e-12));
        CHECK_THAT(res.std_error, WithinAbs(0.0, 1e-12));
    }
}

TEST_CASE("LSM-07: the standard error falls like 1/sqrt(N): quadrupling N halves the SE (within 20%)",
          "[lsm][statistical]") {
    // N = 5000 → 20000（M = 50, Laguerre 3）。期待比 2。SE の推定値そのものの相対 sd は ≈ 1/√(2 n_pairs)
    // = 1.4 %（2500 対）なので比の揺らぎは ≈ 2 % で、±20 % の許容には十分な余裕がある。
    // 実測（seed 4242）: se(5000) = 0.0607, se(20000) = 0.0313, 比 1.94。
    const LsmResult small = solve_all(put_params(5000, 50, 3, 4242)).result();
    const LsmResult large = solve_all(put_params(20000, 50, 3, 4242)).result();
    REQUIRE(large.std_error > 0.0);
    const double ratio = small.std_error / large.std_error;
    INFO("se_small=" << small.std_error << " se_large=" << large.std_error << " ratio=" << ratio);
    CHECK(ratio > 2.0 * 0.8);
    CHECK(ratio < 2.0 * 1.2);
}
