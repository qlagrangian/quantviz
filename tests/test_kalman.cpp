// KALMAN-xx — core/math/mat.hpp と core/stats/kalman.hpp の仕様テスト
// （行列の閉形式・カルマンフィルタの数値的性質・統計的性質）
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/core/math/mat.hpp"
#include "quantviz/core/rng.hpp"
#include "quantviz/core/stats/kalman.hpp"
#include "quantviz/scenes/kalman_pair_model.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::bridge::Command;
using quantviz::core::Kalman;
using quantviz::core::Mat;
using quantviz::scenes::KalmanPairModel;
using quantviz::scenes::KalmanPairSnapshot;

namespace {

/// 行列を要素ごとに比較する（テスト内でのみ使う小さなヘルパ）。
template <std::size_t R, std::size_t C>
void check_close(const Mat<R, C>& actual, const Mat<R, C>& expected, double tol) {
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j) CHECK_THAT(actual(i, j), WithinAbs(expected(i, j), tol));
}

/// KALMAN-09..11 が共有するシーン設定。既定値に引きずられないよう全項目を明示する。
KalmanPairModel::Config pair_config() {
    KalmanPairModel::Config c;
    c.x0             = 100.0;
    c.x_vol          = 0.20;
    c.beta_center    = 1.20;
    c.beta_reversion = 0.002;
    c.state_noise    = 0.003;
    c.obs_noise      = 2.0;
    c.beta_prior     = 1.0;
    c.prior_var      = 0.25;
    c.seed           = 20240912;
    return c;
}

constexpr double kDt = 1.0 / 252.0;  // 1 ステップ = 1 取引日

}  // namespace

TEST_CASE("KALMAN-01: Mat product, transpose and 1x1/2x2/3x3 inverse match hand calculations",
          "[kalman][numeric]") {
    using quantviz::core::is_psd;
    using quantviz::core::is_symmetric;

    SECTION("product of a 2x3 and a 3x2 matrix") {
        const Mat<2, 3> a{{1, 2, 3, 4, 5, 6}};
        const Mat<3, 2> b{{7, 8, 9, 10, 11, 12}};
        const Mat<2, 2> expected{{58, 64, 139, 154}};  // 手計算
        check_close(a * b, expected, 1e-12);
    }

    SECTION("transpose swaps the indices") {
        const Mat<2, 3> a{{1, 2, 3, 4, 5, 6}};
        const Mat<3, 2> expected{{1, 4, 2, 5, 3, 6}};
        check_close(a.transpose(), expected, 1e-12);
        check_close(a.transpose().transpose(), a, 1e-12);
    }

    SECTION("identity, addition, subtraction and scalar multiplication") {
        const auto      i3 = Mat<3, 3>::identity();
        const Mat<3, 3> m{{1, 2, 3, 0, 1, 4, 5, 6, 0}};
        check_close(m * i3, m, 1e-12);
        check_close(i3 * m, m, 1e-12);
        check_close(m + m, m * 2.0, 1e-12);
        check_close(2.0 * m - m, m, 1e-12);
        check_close(m - m, Mat<3, 3>{}, 1e-12);
        check_close(-m, m * -1.0, 1e-12);
    }

    SECTION("1x1 inverse is the reciprocal") {
        const Mat<1, 1> m{{4.0}};
        const auto      inv = m.inverse();
        REQUIRE(inv.has_value());
        CHECK_THAT((*inv)(0, 0), WithinRel(0.25, 1e-15));
    }

    SECTION("2x2 inverse: [[4,7],[2,6]] -> [[0.6,-0.7],[-0.2,0.4]] (det = 10)") {
        const Mat<2, 2> m{{4, 7, 2, 6}};
        const auto      inv = m.inverse();
        REQUIRE(inv.has_value());
        check_close(*inv, Mat<2, 2>{{0.6, -0.7, -0.2, 0.4}}, 1e-15);
        check_close(m * *inv, Mat<2, 2>::identity(), 1e-14);
    }

    SECTION("3x3 inverse: [[1,2,3],[0,1,4],[5,6,0]] -> [[-24,18,5],[20,-15,-4],[-5,4,1]] (det = 1)") {
        const Mat<3, 3> m{{1, 2, 3, 0, 1, 4, 5, 6, 0}};
        const auto      inv = m.inverse();
        REQUIRE(inv.has_value());
        check_close(*inv, Mat<3, 3>{{-24, 18, 5, 20, -15, -4, -5, 4, 1}}, 1e-12);
        check_close(m * *inv, Mat<3, 3>::identity(), 1e-12);
    }

    SECTION("a singular matrix yields no inverse instead of throwing") {
        const Mat<2, 2> singular{{1, 2, 2, 4}};  // det = 0
        CHECK_FALSE(singular.inverse().has_value());
        const Mat<3, 3> singular3{{1, 2, 3, 2, 4, 6, 7, 8, 9}};  // 行 1 = 2 × 行 0
        CHECK_FALSE(singular3.inverse().has_value());
        CHECK_FALSE(Mat<1, 1>{{0.0}}.inverse().has_value());
    }

    SECTION("a near-singular matrix is rejected once tol is relative to the determinant scale") {
        // 行が等差数列 → 数学的には特異だが、丸めで det は 0 ではなく ~1e-17 になる。
        // det のスケール（各項の絶対値の和）は 0.024 なので、相対 tol = 1e-12 なら閾値は
        // 2.4e-14 で確実に弾ける。既定 tol = 0 では「厳密に 0 でない」ため通り得る。
        const Mat<3, 3> near{{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}};
        CHECK_FALSE(near.inverse(1e-12).has_value());

        // 逆に、良条件だが絶対値の小さい行列は同じ tol で通らなければならない。
        // det = 1e-14 なので絶対許容 1e-12 だと誤って特異と判定されてしまう。
        const auto small = Mat<2, 2>{{1e-7, 0.0, 0.0, 1e-7}}.inverse(1e-12);
        REQUIRE(small.has_value());
        CHECK_THAT((*small)(0, 0), WithinRel(1e7, 1e-12));
    }

    SECTION("a subnormal determinant is rejected instead of returning an all-infinity matrix") {
        // det は 0 でも非有限でもないが 1/det が overflow する。ガードが無いと
        // has_value() == true のまま全要素 inf の「逆行列」が返る。
        CHECK_FALSE(Mat<1, 1>{{1e-320}}.inverse().has_value());
        CHECK_FALSE((Mat<2, 2>{{1e-160, 0.0, 0.0, 1e-160}}.inverse().has_value()));
    }

    SECTION("is_symmetric and is_psd reject the matrices they must reject") {
        CHECK(is_symmetric(Mat<2, 2>{{1, 2, 2, 1}}, 0.0));
        CHECK_FALSE(is_symmetric(Mat<2, 2>{{1, 2, 3, 1}}, 1e-12));  // 非対称

        CHECK(is_psd(Mat<2, 2>{{2, 1, 1, 2}}, 0.0));                    // 固有値 1, 3
        CHECK(is_psd(Mat<2, 2>{{1, 1, 1, 1}}, 0.0));                    // 半正定値（階数 1）
        CHECK_FALSE(is_psd(Mat<2, 2>{{0, 0, 0, -1}}, 1e-12));           // 首座小行列式では見抜けない
        CHECK_FALSE(is_psd(Mat<2, 2>{{1, 2, 2, 1}}, 1e-12));            // det = −3 で不定符号
        CHECK_FALSE(is_psd(Mat<2, 2>{{1, 2, 3, 1}}, 1e-12));            // そもそも非対称
        CHECK_FALSE(is_psd(Mat<3, 3>{{1, 0, 0, 0, 1, 0, 0, 0, -1}}, 1e-12));

        // 小行列式が overflow して NaN になる場合。比較を `!(x >= -tol·scale)` で書かないと
        // NaN が「合格」して PSD と誤判定される。
        CHECK_FALSE(is_psd(Mat<2, 2>{{1e200, -1e300, -1e300, 1e200}}, 1e-12));
    }

    SECTION("is_psd stays correct when the matrix is far from unit scale (tol must be relative)") {
        // k 次の主小行列式は ‖m‖^k のオーダーなので、絶対許容だと小さい行列で不定符号を
        // 見逃す。以下はどれも「絶対許容 1e-12 なら PSD と誤判定される」本物の非 PSD 行列。
        CHECK_FALSE(is_psd(Mat<2, 2>{{1e-10, 0.0, 0.0, -1e-20}}, 1e-12));  // 1 次: −1e-20 > −1e-12
        // 2 次: 対角は正だが det = −2e-21（絶対値は 1e-12 より小さい）
        CHECK_FALSE(is_psd(Mat<2, 2>{{1e-7, 1.0000001e-7, 1.0000001e-7, 1e-7}}, 1e-12));
        // 3 次: 2 次までは通るが行列式が −2e-21
        CHECK_FALSE(is_psd(Mat<3, 3>{{1e-7, 1.0000001e-7, 0.0, 1.0000001e-7, 1e-7, 0.0, 0.0, 0.0, 1e-7}},
                           1e-12));

        // 逆に、スケールが大きいだけの真っ当な PSD 行列は tol = 0 でも通る。
        CHECK(is_psd(Mat<2, 2>{{1e16, 3e15, 3e15, 9e14}}, 0.0));  // 階数 1（v vᵀ）
    }
}

TEST_CASE("KALMAN-02: with Q = 0, F = 1, H = 1 and a diffuse prior the estimate is the running mean",
          "[kalman][numeric]") {
    // RLS の縮退形。P0 → ∞ で x̂_n = (1/n)Σz_i に一致し、有限の P0 では
    // x̂_n = mean · (1 − 1/(1 + n·P0/R)) と相対バイアス 1/(1 + n·P0/R) が残る。
    // R = 1, P0 = 1e12 なら n = 1 でも相対 1e-12 で、要求する 1e-8 の十分内側。
    const double        p0 = 1e12;
    quantviz::core::Rng rng(4242);
    Kalman<1, 1>        kf(Mat<1, 1>{}, Mat<1, 1>{{p0}});

    const auto f = Mat<1, 1>::identity();
    const auto q = Mat<1, 1>{};  // Q = 0
    const auto h = Mat<1, 1>::identity();
    const auto r = Mat<1, 1>::identity();

    double sum = 0.0;
    for (int n = 1; n <= 200; ++n) {
        const double z = 3.0 + 2.0 * rng.normal();
        sum += z;
        kf.predict(f, q);
        kf.update(h, Mat<1, 1>{{z}}, r);
        const double mean = sum / static_cast<double>(n);
        REQUIRE_THAT(kf.state()(0, 0), WithinRel(mean, 1e-8));
    }
}

TEST_CASE("KALMAN-03: predict grows the covariance to F P F^T + Q; reset and degenerate updates "
          "leave a well-defined state",
          "[kalman][unit]") {
    const Mat<2, 2> p0{{4.0, 1.0, 1.0, 9.0}};
    const Mat<2, 1> x0{{2.0, -3.0}};
    const Mat<2, 2> q{{0.25, 0.1, 0.1, 0.5}};

    SECTION("F = I: the covariance grows by exactly Q and the state is unchanged") {
        Kalman<2, 1> kf(x0, p0);
        kf.predict(Mat<2, 2>::identity(), q);
        check_close(kf.state(), x0, 1e-12);
        check_close(kf.cov(), p0 + q, 1e-12);
    }

    SECTION("general F: x -> F x and P -> F P F^T + Q") {
        const Mat<2, 2> f{{1.0, 0.5, 0.0, 1.0}};
        Kalman<2, 1>    kf(x0, p0);
        kf.predict(f, q);
        check_close(kf.state(), f * x0, 1e-12);
        check_close(kf.cov(), f * p0 * f.transpose() + q, 1e-12);
    }

    SECTION("Q = 0 and F = I leave the covariance untouched") {
        Kalman<2, 1> kf(x0, p0);
        kf.predict(Mat<2, 2>::identity(), Mat<2, 2>{});
        check_close(kf.cov(), p0, 1e-12);
    }

    SECTION("update returns the prior residual z - H x^-, not the posterior one") {
        Kalman<2, 1>    kf(x0, p0);
        const Mat<1, 2> h{{1.0, 2.0}};
        const Mat<1, 1> z{{7.0}};
        kf.predict(Mat<2, 2>::identity(), q);
        const double prior_pred = (h * kf.state())(0, 0);  // = 2 + 2·(−3) = −4
        const double y          = kf.update(h, z, Mat<1, 1>{{0.5}})(0, 0);
        CHECK_THAT(y, WithinRel(7.0 - prior_pred, 1e-12));
        CHECK_THAT(y, WithinRel(11.0, 1e-12));  // 手計算
    }

    SECTION("reset(x0, P0) restores exactly and reset() restores the documented defaults") {
        Kalman<2, 1> kf(x0, p0);
        kf.predict(Mat<2, 2>{{1.0, 0.5, 0.0, 1.0}}, q);
        kf.update(Mat<1, 2>{{1.0, 0.0}}, Mat<1, 1>{{1.0}}, Mat<1, 1>{{0.25}});

        kf.reset(x0, p0);
        check_close(kf.state(), x0, 0.0);
        check_close(kf.cov(), p0, 0.0);
        CHECK(kf.skipped_updates() == 0);

        kf.reset();
        check_close(kf.state(), Mat<2, 1>{}, 0.0);
        check_close(kf.cov(), Mat<2, 2>::diagonal(Kalman<2, 1>::kDefaultPriorVariance), 0.0);
        CHECK(Kalman<2, 1>::kDefaultPriorVariance == 1e6);
    }

    SECTION("an asymmetric P0 comes back symmetrised") {
        const Mat<2, 2> skewed{{4.0, 1.0, 3.0, 9.0}};  // 非対称
        Kalman<2, 1>    kf(x0, skewed);
        CHECK(quantviz::core::is_symmetric(kf.cov(), 0.0));
        check_close(kf.cov(), Mat<2, 2>{{4.0, 2.0, 2.0, 9.0}}, 0.0);  // 非対角は平均の 2.0
    }

    SECTION("a non-finite observation is skipped instead of poisoning the state") {
        Kalman<2, 1>    kf(x0, p0);
        const Mat<1, 2> h{{1.0, 0.0}};
        const Mat<1, 1> r{{0.25}};
        kf.update(h, Mat<1, 1>{{1.0}}, r);
        const Mat<2, 1> good_x = kf.state();
        const Mat<2, 2> good_p = kf.cov();

        const double nan_z = std::numeric_limits<double>::quiet_NaN();
        kf.update(h, Mat<1, 1>{{nan_z}}, r);
        CHECK(kf.skipped_updates() == 1);
        check_close(kf.state(), good_x, 0.0);
        check_close(kf.cov(), good_p, 0.0);

        kf.update(h, Mat<1, 1>{{std::numeric_limits<double>::infinity()}}, r);
        CHECK(kf.skipped_updates() == 2);

        // 良い観測に戻れば、何事も無かったように更新が再開する
        kf.update(h, Mat<1, 1>{{1.0}}, r);
        CHECK(kf.skipped_updates() == 2);
        CHECK(std::isfinite(kf.state()(0, 0)));
        CHECK(kf.state()(0, 0) != good_x(0, 0));
    }
}

TEST_CASE("KALMAN-04: update never increases the covariance (P_prior - P_post is PSD)",
          "[kalman][property]") {
    quantviz::core::Rng rng(7);
    Kalman<2, 1>        kf(Mat<2, 1>{}, Mat<2, 2>{{5.0, 1.5, 1.5, 3.0}});

    const Mat<2, 2> f{{1.0, 0.1, 0.0, 1.0}};
    const Mat<2, 2> q{{1e-4, 0.0, 0.0, 1e-4}};
    const Mat<1, 1> r{{0.09}};

    for (int i = 0; i < 200; ++i) {
        kf.predict(f, q);
        const Mat<2, 2> prior = kf.cov();
        const Mat<1, 2> h{{1.0, 0.3 * rng.normal()}};
        kf.update(h, Mat<1, 1>{{rng.normal()}}, r);
        const Mat<2, 2> post = kf.cov();
        // 差 P⁻ − P⁺ = K S Kᵀ は NZ = 1 なので厳密には階数 1、つまり行列式は 0。
        // 丸めで ±eps 程度ぶれるため相対許容が要る。実測した最悪の相対違反は 2.8e-14
        // （≈ 128 · DBL_EPSILON; -O0/-O2/-ffp-contract=fast で 1.9〜2.8e-14）なので、
        // 35 倍の余裕を取って tol = 1e-12（≈ 4500 · DBL_EPSILON）とする。
        // 本物の PSD 違反はこれより桁違いに大きいので、検出力は落ちない。
        REQUIRE(quantviz::core::is_psd(prior - post, 1e-12));
        // 対角の減少は実測で最低 4.1e-6 あるので、緩衝なしの厳密比較でよい。
        REQUIRE(post(0, 0) <= prior(0, 0));
        REQUIRE(post(1, 1) <= prior(1, 1));
    }
}

TEST_CASE("KALMAN-05: the covariance stays symmetric and PSD after 1000 steps", "[kalman][property]") {
    quantviz::core::Rng rng(20240912);
    Kalman<2, 1>        kf;  // 既定: x = 0, P = I·1e6

    const Mat<2, 2> f{{1.0, 1.0, 0.0, 1.0}};
    const Mat<2, 2> q{{1e-6, 0.0, 0.0, 1e-8}};
    const Mat<1, 1> r{{0.04}};

    for (int i = 0; i < 1000; ++i) {
        kf.predict(f, q);
        kf.update(Mat<1, 2>{{1.0, 0.0}}, Mat<1, 1>{{0.5 * rng.normal()}}, r);
        // 対称化が上三角を下三角へ複製するので、対称性は丸めに依らず厳密（実測違反 0）。
        REQUIRE(quantviz::core::is_symmetric(kf.cov(), 0.0));
        // Q ≻ 0 なので P は厳密に正定値のまま。実測した最小の相対主小行列式は +8.0e-8 で、
        // 丸め（~1e-16）の 8 桁上。したがって tol = 0（相対許容ゼロ）で判定してよく、
        // これが最も強い主張になる。
        REQUIRE(quantviz::core::is_psd(kf.cov(), 0.0));
    }
    CHECK(std::isfinite(kf.cov()(0, 0)));
    CHECK(kf.cov()(0, 0) > 0.0);
}

TEST_CASE("KALMAN-06: the innovation sequence is white (lag-1 autocorrelation within 4 SE of 0)",
          "[kalman][statistical]") {
    // 正しく指定された局所レベルモデル（F = H = 1）なので、イノベーションは平均 0 の白色列。
    // 定常状態に入ってからは分散も一定（S = P⁻ + R が収束する）ので、標本 lag-1 自己相関は
    // 漸近的に N(0, 1/N)。期待値 0、SE = 1/√N = 1/√5000 ≈ 0.01414、許容 = 4 SE ≈ 0.0566。
    const double q_var = 1e-4, r_var = 0.01;
    const int    burn_in = 500, n = 5000;

    quantviz::core::Rng rng(31415);
    double              beta = 0.0;
    Kalman<1, 1>        kf;

    const auto f = Mat<1, 1>::identity();
    const auto h = Mat<1, 1>::identity();
    const auto q = Mat<1, 1>{{q_var}};
    const auto r = Mat<1, 1>{{r_var}};

    std::vector<double> e;
    e.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < burn_in + n; ++i) {
        beta += std::sqrt(q_var) * rng.normal();
        const double z = beta + std::sqrt(r_var) * rng.normal();
        kf.predict(f, q);
        const double prior_state = kf.state()(0, 0);  // 予測後・更新前
        const double innovation  = kf.update(h, Mat<1, 1>{{z}}, r)(0, 0);
        // イノベーションの定義は「事前残差」 y = z − H x⁻（事後残差ではない）
        REQUIRE_THAT(innovation, WithinAbs(z - prior_state, 1e-12));
        if (i >= burn_in) e.push_back(innovation);
    }

    double mean = 0.0;
    for (double v : e) mean += v;
    mean /= static_cast<double>(e.size());

    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i + 1 < e.size(); ++i) num += (e[i] - mean) * (e[i + 1] - mean);
    for (double v : e) den += (v - mean) * (v - mean);
    const double rho = num / den;

    const double se = 1.0 / std::sqrt(static_cast<double>(n));
    CHECK_THAT(rho, WithinAbs(0.0, 4.0 * se));
}

TEST_CASE("KALMAN-07: a random-walk beta is tracked inside the +/-3 sigma band at least 95% of the time",
          "[kalman][statistical]") {
    // 正しく指定されたモデルなので β̂ − β | データ ~ N(0, P) であり、±3√P の被覆率は 0.9973。
    // N = 5000 の標本割合の SE = sqrt(0.9973 · 0.0027 / 5000) ≈ 7.3e-4 なので、
    // 期待値 − 4 SE ≈ 0.9944 > 0.95（仕様のしきい値）。しきい値は 4 SE 以上の余裕を持つ。
    const double q_var = 1e-4, r_var = 2.5e-3;
    const int    n = 5000;

    quantviz::core::Rng rng(2718);
    double              beta = 1.0;
    Kalman<1, 1>        kf(Mat<1, 1>{{1.0}}, Mat<1, 1>{{1.0}});

    const auto f = Mat<1, 1>::identity();
    const auto q = Mat<1, 1>{{q_var}};
    const auto r = Mat<1, 1>{{r_var}};

    int inside = 0;
    for (int i = 0; i < n; ++i) {
        beta += std::sqrt(q_var) * rng.normal();
        const double x = 1.0 + 0.1 * rng.normal();
        const double y = beta * x + std::sqrt(r_var) * rng.normal();

        kf.predict(f, q);
        kf.update(Mat<1, 1>{{x}}, Mat<1, 1>{{y}}, r);

        const double err = std::abs(kf.state()(0, 0) - beta);
        if (err <= 3.0 * std::sqrt(kf.cov()(0, 0))) ++inside;
    }
    const double fraction = static_cast<double>(inside) / static_cast<double>(n);
    CHECK(fraction >= 0.95);
}

TEST_CASE("KALMAN-08: with Q = 0 the regression state converges to the OLS estimate",
          "[kalman][numeric]") {
    // 状態 β、観測行列 H = [x_t]、Q = 0 の再帰最小二乗は
    //   β̂_N = Σxy / (Σx² + R/P0)
    // に厳密に等しい。R = 1, P0 = 1e8, N = 2000（Σx² ≈ 2000）なら相対バイアスは
    // (R/P0)/Σx² ≈ 5e-12 で、要求する相対 1e-6 の十分内側。P0 を有限に取ることの
    // バイアスはここに現れるだけで、逆行列の悪条件化は起きない。
    const double p0 = 1e8, beta_true = 2.0;
    const int    n = 2000;

    quantviz::core::Rng rng(161803);
    Kalman<1, 1>        kf(Mat<1, 1>{}, Mat<1, 1>{{p0}});

    const auto f = Mat<1, 1>::identity();
    const auto q = Mat<1, 1>{};  // Q = 0
    const auto r = Mat<1, 1>::identity();

    double sxy = 0.0, sxx = 0.0;
    for (int i = 0; i < n; ++i) {
        const double x = rng.normal();
        const double y = beta_true * x + 0.5 * rng.normal();
        sxy += x * y;
        sxx += x * x;
        kf.predict(f, q);
        kf.update(Mat<1, 1>{{x}}, Mat<1, 1>{{y}}, r);
    }
    CHECK_THAT(kf.state()(0, 0), WithinRel(sxy / sxx, 1e-6));
}

TEST_CASE("KALMAN-09: the pair scene satisfies the Model contract and publishes a compact POD snapshot",
          "[kalman][contract]") {
    STATIC_REQUIRE(quantviz::bridge::Model<KalmanPairModel>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<KalmanPairSnapshot>);
    STATIC_REQUIRE(std::is_default_constructible_v<KalmanPairSnapshot>);
    STATIC_REQUIRE(sizeof(KalmanPairSnapshot) <= 128);  // 2 キャッシュライン以内

    SECTION("a fresh model reports seq 0, the initial pair and the current parameters") {
        const KalmanPairModel m(pair_config());
        const auto            s = m.snapshot();
        CHECK(s.seq == 0);
        CHECK(s.t == 0.0);
        CHECK(s.x == 100.0);
        CHECK(s.beta_true == 1.20);
        CHECK(s.y == 120.0);  // β·x（観測ノイズはまだ引いていない）
        CHECK(s.obs_noise == 2.0);
        CHECK(s.state_noise == 0.003);
        CHECK(s.beta_hat == 1.0);    // フィルタの事前平均
        CHECK(s.beta_var == 0.25);   // フィルタの事前分散
        CHECK(s.spread == 0.0);
        CHECK(s.innovation == 0.0);
        CHECK(s.skipped == 0);
    }

    SECTION("two models with the same seed produce bit-identical paths") {
        KalmanPairModel a(pair_config()), b(pair_config());
        for (int i = 0; i < 500; ++i) {
            a.step(kDt);
            b.step(kDt);
            const auto sa = a.snapshot(), sb = b.snapshot();
            REQUIRE(sa.x == sb.x);
            REQUIRE(sa.y == sb.y);
            REQUIRE(sa.beta_true == sb.beta_true);
            REQUIRE(sa.beta_hat == sb.beta_hat);
        }
    }

    SECTION("the published estimate equals a bare Kalman<1,1> fed the same observations (dual-run)") {
        // シーンは「生成（Rng）+ core::Kalman」の糊でしかないことを、公開された (x, y) を
        // そのまま素のフィルタに流して bit 一致で確かめる（演算順が同じなので厳密一致する）。
        const auto      cfg = pair_config();
        KalmanPairModel m(cfg);
        Kalman<1, 1>    bare(Mat<1, 1>{{cfg.beta_prior}}, Mat<1, 1>{{cfg.prior_var}});

        const auto f = Mat<1, 1>::identity();
        const auto q = Mat<1, 1>{{cfg.state_noise * cfg.state_noise}};
        const auto r = Mat<1, 1>{{cfg.obs_noise * cfg.obs_noise}};

        for (int i = 0; i < 500; ++i) {
            m.step(kDt);
            const auto s = m.snapshot();
            bare.predict(f, q);
            const double innovation = bare.update(Mat<1, 1>{{s.x}}, Mat<1, 1>{{s.y}}, r)(0, 0);
            REQUIRE(s.seq == static_cast<std::uint64_t>(i + 1));
            REQUIRE(s.beta_hat == bare.state()(0, 0));
            REQUIRE(s.beta_var == bare.cov()(0, 0));
            REQUIRE(s.innovation == innovation);
            // spread = y − β̂x は桁落ち量（|y| ≈ 100 に対し 1e-4）。FMA 縮約の有無で数 ulp ずれるので絶対許容で比較する。
            REQUIRE_THAT(s.spread, WithinAbs(s.y - bare.state()(0, 0) * s.x, 1e-10 * std::max(1.0, std::abs(s.y))));
        }
    }

    SECTION("Reset replays the same path bit-for-bit") {
        // Reset は Rng::reseed 経由で seed を巻き戻す。reseed は std::normal_distribution の
        // Box-Muller キャッシュも捨てるので、「奇数個の normal() を引いた直後の Reset」でも
        // 系列は最初から完全に一致する。ここはその契約を bit 一致で固定する。
        KalmanPairModel m(pair_config());
        for (int i = 0; i < 200; ++i) m.step(kDt);
        const auto first = m.snapshot();

        m.apply(Command::reset());
        REQUIRE(m.snapshot().seq == 0);
        for (int i = 0; i < 200; ++i) m.step(kDt);
        const auto second = m.snapshot();

        REQUIRE(second.seq == first.seq);
        REQUIRE(second.t == first.t);
        REQUIRE(second.x == first.x);
        REQUIRE(second.y == first.y);
        REQUIRE(second.beta_true == first.beta_true);
        REQUIRE(second.beta_hat == first.beta_hat);
        REQUIRE(second.beta_var == first.beta_var);
        REQUIRE(second.spread == first.spread);
        REQUIRE(second.innovation == first.innovation);
    }

    SECTION("Reset rewinds the path and the filter but keeps the parameters") {
        KalmanPairModel m(pair_config());
        for (int i = 0; i < 200; ++i) m.step(kDt);
        m.apply(Command::set_param(KalmanPairModel::kObsNoise, 3.5));
        m.apply(Command::reset());
        const auto s = m.snapshot();
        CHECK(s.seq == 0);
        CHECK(s.t == 0.0);
        CHECK(s.x == 100.0);
        CHECK(s.beta_true == 1.20);
        CHECK(s.beta_hat == 1.0);
        CHECK(s.beta_var == 0.25);
        CHECK(s.obs_noise == 3.5);  // UI で変えた値は Reset で失わない
    }
}

TEST_CASE("KALMAN-10: the generated pair is cointegrated - the spread variance does not grow with N",
          "[kalman][statistical]") {
    // x と y は個別には非定常（μ = 0 の厳密離散化 GBM）だが、y_t = β_t x_t + ε_t の β_t が
    // kBetaTrue を中心とする平均回帰付きランダムウォークなので、ヘッジ後のスプレッド
    // s_t = y_t − β̂_t x_t は定常になる。定常なら前半 N/2 点と後半 N/2 点の標本分散は
    // 同じ母数の推定値であり、その比は 1 の周りに分布する。
    //   期待値 1、正規近似での標本分散の相対 SE = sqrt(2/(n−1)) = sqrt(2/1999) ≈ 3.2 %、
    //   比の相対 SE はその √2 倍 ≈ 4.5 %。4 SE でも ±18 %。
    // 残る系統的なずれは x の水準変化（Var(s) = R²/(x²P⁻+R) が x に緩く依存する）で、
    // 既定パラメータでは x が 2 倍になっても 30 % 程度。よって [0.5, 2] は 4 SE より
    // ずっと緩いしきい値で、seed 固定なら安定して通る。
    const int       n = 4000;
    KalmanPairModel m(pair_config());

    std::vector<double> spread, x;
    spread.reserve(static_cast<std::size_t>(n));
    x.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        m.step(kDt);
        const auto s = m.snapshot();
        spread.push_back(s.spread);
        x.push_back(s.x);
    }

    const auto sample_variance = [](const std::vector<double>& v, std::size_t lo, std::size_t hi) {
        const double count = static_cast<double>(hi - lo);
        double       mean  = 0.0;
        for (std::size_t i = lo; i < hi; ++i) mean += v[i];
        mean /= count;
        double ss = 0.0;
        for (std::size_t i = lo; i < hi; ++i) ss += (v[i] - mean) * (v[i] - mean);
        return ss / (count - 1.0);
    };

    const std::size_t half   = static_cast<std::size_t>(n) / 2;
    const std::size_t all    = static_cast<std::size_t>(n);
    const double      first  = sample_variance(spread, 0, half);
    const double      second = sample_variance(spread, half, all);
    REQUIRE(first > 0.0);
    REQUIRE(second > 0.0);

    const double ratio = first / second;
    INFO("var(first half) = " << first << ", var(second half) = " << second << ", ratio = " << ratio);
    CHECK(ratio >= 0.5);
    CHECK(ratio <= 2.0);

    // 検定が空虚でないこと（スプレッドが「動いていないから定常に見える」だけではないこと）:
    // x は μ = 0・σ = 0.20 の GBM を T = N·dt = 15.87 年ぶん歩くので log x の標準偏差は
    // σ√T = 0.80、水準の標本標準偏差は x0 = 100 のオーダー（この seed では 16.8）。
    // 一方スプレッドは定常で標準偏差 ≈ sqrt(R²/(x²P⁻+R)) ≈ 1.8（σ_ε = 2 の少し内側）。
    // 実測比は約 9 倍なので、しきい値 5 倍は 1.8 倍の余裕がある。
    CHECK(std::sqrt(sample_variance(x, 0, all)) > 5.0 * std::sqrt(sample_variance(spread, 0, all)));
}

TEST_CASE("KALMAN-11: SetParam(observation noise) takes effect from the next step", "[kalman][unit]") {
    SECTION("apply() does not step; the new observation noise is used from the next step") {
        KalmanPairModel m(pair_config()), control(pair_config());
        for (int i = 0; i < 20; ++i) {
            m.step(kDt);
            control.step(kDt);
        }
        const auto before = m.snapshot();
        REQUIRE(before.obs_noise == 2.0);

        m.apply(Command::set_param(KalmanPairModel::kObsNoise, 8.0));
        const auto applied = m.snapshot();
        CHECK(applied.seq == before.seq);             // apply 自体はステップを進めない
        CHECK(applied.obs_noise == 8.0);              // 現在のパラメータとしては即座に見える
        CHECK(applied.beta_hat == before.beta_hat);   // 推定は次のステップまで変わらない
        CHECK(applied.beta_var == before.beta_var);

        m.step(kDt);
        control.step(kDt);
        const auto after = m.snapshot();
        CHECK(after.seq == before.seq + 1);
        CHECK(after.obs_noise == 8.0);
        // R が変わったので、同じ 1 ステップでも共分散の更新結果が対照モデルと異なる
        CHECK(after.beta_var != control.snapshot().beta_var);
    }

    SECTION("the state noise and the true beta are settable as well") {
        // κ = 0.002 の平均回帰は半減期 ln2/κ ≈ 347 ステップなので、3000 ステップあれば
        // β_t は新しい中心 2.0 の定常分布（std = state_noise/sqrt(2κ−κ²) ≈ 0.16）に入る。
        // 許容 0.8 は約 5 σ。seed 固定。
        KalmanPairModel m(pair_config());
        m.apply(Command::set_param(KalmanPairModel::kStateNoise, 0.01));
        m.apply(Command::set_param(KalmanPairModel::kBetaTrue, 2.0));
        CHECK(m.snapshot().state_noise == 0.01);
        for (int i = 0; i < 3000; ++i) m.step(kDt);
        CHECK_THAT(m.snapshot().beta_true, WithinAbs(2.0, 0.8));
    }

    SECTION("bad input is clamped and unknown parameter ids are ignored") {
        KalmanPairModel m(pair_config());
        m.apply(Command::set_param(KalmanPairModel::kObsNoise, -1.0));
        CHECK(m.snapshot().obs_noise > 0.0);  // R = 0 は S = x²P + R を特異にするので > 0 に保つ
        m.apply(Command::set_param(KalmanPairModel::kStateNoise, -1.0));
        CHECK(m.snapshot().state_noise == 0.0);  // Q ≥ 0

        const auto before = m.snapshot();
        m.apply(Command::set_param(999, 123.0));  // 未知 id は無視
        const auto after = m.snapshot();
        CHECK(after.obs_noise == before.obs_noise);
        CHECK(after.state_noise == before.state_noise);
        CHECK(after.beta_true == before.beta_true);

        // NaN は「どの比較も false」なので std::max(v, lo) だと素通りする。素通りすると R = NaN →
        // S = x²P⁻ + R が非有限 → core::Kalman::update が毎ステップ更新を捨て、Reset は cfg_ を
        // 保持するので二度と復帰しない（ImGui のスライダーは ctrl+click で "nan" と打てる）。
        // 下限を第 1 引数に置き、非有限を明示的に弾くことでこれを防ぐ。
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();
        m.apply(Command::set_param(KalmanPairModel::kObsNoise, nan));
        CHECK(std::isfinite(m.snapshot().obs_noise));
        CHECK(m.snapshot().obs_noise > 0.0);
        m.apply(Command::set_param(KalmanPairModel::kStateNoise, nan));
        CHECK(m.snapshot().state_noise == 0.0);
        m.apply(Command::set_param(KalmanPairModel::kObsNoise, inf));
        CHECK(std::isfinite(m.snapshot().obs_noise));

        for (int i = 0; i < 100; ++i) m.step(kDt);  // クランプ後も有限値を保つ
        CHECK(std::isfinite(m.snapshot().beta_hat));
        CHECK(std::isfinite(m.snapshot().spread));
        CHECK(std::isfinite(m.snapshot().beta_var));
        CHECK(m.snapshot().skipped == 0);  // 観測更新が捨てられていない
    }

    SECTION("a non-finite true beta is rejected and the previous value is kept") {
        const double    nan = std::numeric_limits<double>::quiet_NaN();
        KalmanPairModel m(pair_config());
        REQUIRE(m.snapshot().beta_true == 1.20);

        m.apply(Command::set_param(KalmanPairModel::kBetaTrue, nan));
        CHECK(m.snapshot().beta_true == 1.20);
        m.apply(Command::set_param(KalmanPairModel::kBetaTrue, -std::numeric_limits<double>::infinity()));
        CHECK(m.snapshot().beta_true == 1.20);

        for (int i = 0; i < 100; ++i) m.step(kDt);
        CHECK(std::isfinite(m.snapshot().beta_true));
        CHECK(std::isfinite(m.snapshot().beta_hat));
        CHECK(m.snapshot().skipped == 0);

        // Config 経由の非有限・範囲外の値も構築時に弾く（UI 以外の経路も塞ぐ）。
        auto bad           = pair_config();
        bad.beta_center    = nan;
        bad.x0             = 0.0;
        bad.beta_reversion = 5.0;
        KalmanPairModel guarded(bad);
        const auto      gs = guarded.snapshot();
        CHECK(std::isfinite(gs.beta_true));
        CHECK(gs.x > 0.0);
        CHECK(std::isfinite(gs.y));
    }
}
