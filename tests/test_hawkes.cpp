// HAWKES-xx — core/models/hawkes.hpp の仕様テスト（docs/03_tdd_spec.md §6.2）
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "quantviz/core/math/mat.hpp"
#include "quantviz/core/models/hawkes.hpp"
#include "quantviz/core/rng.hpp"
#include "quantviz/core/stats/optim.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using quantviz::core::hawkes_from_unconstrained;
using quantviz::core::hawkes_intensity;
using quantviz::core::hawkes_compensator;
using quantviz::core::hawkes_fit;
using quantviz::core::hawkes_fit_into;
using quantviz::core::hawkes_log_likelihood;
using quantviz::core::hawkes_rescaled_residuals;
using quantviz::core::hawkes_simulate;
using quantviz::core::hawkes_stable;
using quantviz::core::hawkes_to_unconstrained;
using quantviz::core::HawkesIntensity;
using quantviz::core::HawkesFit;
using quantviz::core::HawkesParams;
using quantviz::core::Mat;
using quantviz::core::OptimOptions;
using quantviz::core::Rng;

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
using Theta           = std::array<double, 3>;  // 無制約パラメータ

// 分岐比 α/β = 0.5、単位時間あたりの定常強度 μ/(1−α/β) = 2 の代表値
constexpr HawkesParams kTruth{1.0, 0.5, 1.0};

/// seed 固定の合成パス（[0, T) の事象時刻、昇順）。
std::vector<double> simulate(HawkesParams p, double T, std::uint64_t seed, std::size_t cap = 20000) {
    std::vector<double> out(cap);
    Rng                 rng(seed);
    out.resize(hawkes_simulate(p, T, rng, out));
    return out;
}

/// テスト側の直接計算（O(n²)）: Σ log λ(t_i) − Λ(T)。λ は直接和、補償子は定義どおりの総和。
double direct_log_likelihood(HawkesParams p, std::span<const double> times, double T) {
    double s = 0.0;
    for (const double ti : times) s += std::log(hawkes_intensity(p, times, ti));
    double comp = p.mu * T;
    for (const double ti : times) comp += (p.alpha / p.beta) * (1.0 - std::exp(-p.beta * (T - ti)));
    return s - comp;
}

/// 4 点 Gauss–Legendre を m 分割で合成した ∫_a^b f。節点が開区間の内部にしか無いので、
/// 事象時刻での λ の跳びを踏まずに済む（Simpson のように端点を評価しない）。
template <class F>
double gauss_legendre(F&& f, double a, double b, int m) {
    constexpr std::array<double, 4> x{-0.8611363115940526, -0.3399810435848563, 0.3399810435848563,
                                      0.8611363115940526};
    constexpr std::array<double, 4> w{0.3478548451374538, 0.6521451548625461, 0.6521451548625461,
                                      0.3478548451374538};
    const double h   = (b - a) / static_cast<double>(m);
    double       sum = 0.0;
    for (int k = 0; k < m; ++k) {
        const double lo  = a + h * static_cast<double>(k);
        const double mid = lo + 0.5 * h;
        for (std::size_t q = 0; q < x.size(); ++q) sum += w[q] * f(mid + 0.5 * h * x[q]);
    }
    return 0.5 * h * sum;
}
}  // namespace

TEST_CASE("HAWKES-01: the intensity mu + sum alpha exp(-beta (t - t_i)) matches hand-computed values",
          "[hawkes][numeric]") {
    // β = ln 2 を選ぶと e^{−β Δ} = 2^{−Δ} で手計算が二進で厳密になる。
    // μ = 0.5, α = 2, 事象 {0, 1, 2}（λ は t_i < t の和。t = t_i ちょうどでは自分の寄与を含めない）
    //   λ(0) = 0.5
    //   λ(1) = 0.5 + 2·2^{−1}                     = 1.5
    //   λ(2) = 0.5 + 2·(2^{−2} + 2^{−1})          = 2.0
    //   λ(3) = 0.5 + 2·(2^{−3} + 2^{−2} + 2^{−1}) = 2.25
    //   λ(5) = 0.5 + 2·(2^{−5} + 2^{−4} + 2^{−3}) = 0.9375
    const HawkesParams          p{0.5, 2.0, std::log(2.0)};
    const std::array<double, 3> times{0.0, 1.0, 2.0};

    CHECK_THAT(hawkes_intensity(p, times, 0.0), WithinRel(0.5, 1e-12));
    CHECK_THAT(hawkes_intensity(p, times, 1.0), WithinRel(1.5, 1e-12));
    CHECK_THAT(hawkes_intensity(p, times, 2.0), WithinRel(2.0, 1e-12));
    CHECK_THAT(hawkes_intensity(p, times, 3.0), WithinRel(2.25, 1e-12));
    CHECK_THAT(hawkes_intensity(p, times, 5.0), WithinRel(0.9375, 1e-12));

    SECTION("no events: the intensity is the baseline mu") {
        const std::span<const double> none;
        CHECK(hawkes_intensity(p, none, 0.0) == p.mu);
        CHECK(hawkes_intensity(p, none, 1e9) == p.mu);
    }

    SECTION("HawkesIntensity updates in O(1) and agrees with the direct sum") {
        // 逐次版は「事象時刻を含む」λ(t⁺) を返す（thinning の上界と LOB シーンが使う値）。
        // したがって t = t_i では直接和より α だけ大きく、t > t_i では一致する。
        HawkesIntensity in(p);
        CHECK(in.at(0.0) == p.mu);
        for (const double t : times) in.add_event(t);
        CHECK_THAT(in.at(2.0), WithinRel(2.0 + p.alpha, 1e-12));  // λ(2⁺) = 4.0
        CHECK_THAT(in.at(3.0), WithinRel(2.25, 1e-12));
        CHECK_THAT(in.at(5.0), WithinRel(0.9375, 1e-12));

        // 1000 事象の系列で直接和と突き合わせる（O(1) 更新が O(n) 和と同じ値を出すこと）
        Rng                 rng(20260912);
        std::vector<double> ts;
        HawkesIntensity     inc(kTruth);
        double              t = 0.0;
        for (int i = 0; i < 1000; ++i) {
            t += 0.01 + 0.5 * rng.uniform();
            ts.push_back(t);
            inc.add_event(t);
            const double probe = t + 0.25;
            REQUIRE_THAT(inc.at(probe), WithinRel(hawkes_intensity(kTruth, ts, probe), 1e-12));
        }

        SECTION("reset returns to the baseline") {
            inc.reset();
            CHECK(inc.at(0.0) == kTruth.mu);
            CHECK(inc.at(1e6) == kTruth.mu);
        }
    }
}

TEST_CASE("HAWKES-02: the intensity never falls below the baseline mu and decays monotonically between events",
          "[hawkes][property]") {
    Rng                 rng(7);
    std::vector<double> ts;
    double              t = 0.0;
    for (int i = 0; i < 200; ++i) {
        t += 0.05 + rng.uniform();
        ts.push_back(t);
    }
    HawkesIntensity in(kTruth);
    for (const double e : ts) in.add_event(e);

    for (int k = 0; k <= 4000; ++k) {
        const double u = 0.05 * static_cast<double>(k);  // [0, 200] を 0.05 刻みで
        REQUIRE(hawkes_intensity(kTruth, ts, u) >= kTruth.mu);
        REQUIRE(in.at(u) >= kTruth.mu);
    }

    SECTION("alpha = 0 is exactly the Poisson baseline") {
        const HawkesParams p{kTruth.mu, 0.0, kTruth.beta};
        for (const double u : {0.0, 1.0, 10.0, 123.4}) CHECK(hawkes_intensity(p, ts, u) == p.mu);
    }

    SECTION("between two events the intensity is non-increasing") {
        // 事象 {1.0, 4.0} の開区間 (1, 4) 上で単調非増加。t = 1 ちょうどは λ(1) = μ（自分の寄与を
        // 含めない）で、直後に α だけ跳ねるので開区間で見る。
        const std::array<double, 2> pair{1.0, 4.0};
        CHECK(hawkes_intensity(kTruth, pair, 1.0) == kTruth.mu);
        double prev = kInf;
        for (int k = 1; k < 300; ++k) {
            const double u = 1.0 + 0.01 * static_cast<double>(k);
            const double v = hawkes_intensity(kTruth, pair, u);
            REQUIRE(v <= prev);
            prev = v;
        }
        CHECK(hawkes_intensity(kTruth, pair, 1.01) > kTruth.mu);  // 事象の直後は跳ねている
    }
}

TEST_CASE("HAWKES-03: the mean event count from thinning matches mu T / (1 - alpha/beta) within 4 SE",
          "[hawkes][statistical]") {
    // 真値 μ = 1, α = 0.5, β = 1（分岐比 η = 0.5）、T = 200、R = 200 反復、seed 固定。
    // 定常近似の期待値 E[N] ≈ μT/(1−η) = 200/0.5 = 400。判定は標本から作った平均の標準誤差 s/√R の 4 倍
    // （§2.4「モンテカルロ・推定量 = 4 SE」）。理論的には Var(N) ≈ μT/(1−η)³ = 1600（sd = 40）なので
    // SE ≈ 40/√200 ≈ 2.8、許容幅は ≈ ±11.3。実測の SE は下でテスト自身が計算して使う。
    //
    // 注意（許容幅の内訳）: t = 0 で履歴が空なので過程はまだ定常ではない。期待強度は
    //   E[λ(t)] = μ/(1−η)·(1 − η e^{−β(1−η)t})
    // で、有限区間の期待件数は厳密には
    //   E[N(T)] = μ/(1−η)·[T − η(1 − e^{−β(1−η)T})/(β(1−η))] = 398（この設定）
    // になる。定常近似の 400 との差は 2 件（この決定論的なずれは T を伸ばしても μη/(β(1−η)²) = 2 で
    // 頭打ちになる一方、SE は √T で伸びる）。T = 50 ではこの差が 4 SE の許容のうち 1.44 SE を占め、
    // 別ツールチェーン（CI の macOS / Windows は同じ seed でも別の乱数列を引く）では平均 −1.44 の
    // 正規変量を引き直すことになって ≈ 0.5 % が赤になる。T = 200 では 0.71 SE まで下がり、
    // 定常近似との比較でも 3.3 SE 以上の統計的余裕が残る。
    // 下では定常近似（仕様どおり 400）と厳密値（398）の両方を 4 SE で確認する。
    // 実測（seed 20260913）: mean = 398.56、sd = 40.63、SE = 2.873
    //   （定常近似から −0.50 SE、厳密値から +0.20 SE）。
    constexpr int         kReps = 200;
    constexpr double      kT    = 200.0;
    constexpr std::size_t kCap  = 4096;  // 上側の裾でも溢れない容量（平均 400、sd 40。実測の最大は 543）
    const double          expected = kTruth.mu * kT / (1.0 - kTruth.alpha / kTruth.beta);
    REQUIRE(expected == 400.0);

    Rng                   rng(20260913);
    std::array<double, kCap> buf{};
    double                sum = 0.0, sum2 = 0.0;
    for (int r = 0; r < kReps; ++r) {
        const std::size_t n = hawkes_simulate(kTruth, kT, rng, buf);
        REQUIRE(n < kCap);  // 打ち切られていない（打ち切られると件数が偏る）
        const double x = static_cast<double>(n);
        sum += x;
        sum2 += x * x;
        // 生成された時刻は昇順で [0, T) に入る
        REQUIRE(buf[0] >= 0.0);
        for (std::size_t i = 1; i < n; ++i) REQUIRE(buf[i] > buf[i - 1]);
        if (n > 0) REQUIRE(buf[n - 1] < kT);
    }
    const double mean = sum / kReps;
    const double var  = (sum2 - sum * mean) / (kReps - 1);
    const double se   = std::sqrt(var / kReps);
    INFO("mean = " << mean << ", sd = " << std::sqrt(var) << ", SE = " << se);
    CHECK(se > 0.0);
    CHECK_THAT(mean, WithinAbs(expected, 4.0 * se));

    // 履歴が空な立ち上がりを勘定に入れた厳密な期待件数（上のコメントの式）
    const double eta   = kTruth.alpha / kTruth.beta;
    const double exact = kTruth.mu / (1.0 - eta) *
                         (kT - eta * (1.0 - std::exp(-kTruth.beta * (1.0 - eta) * kT)) /
                                   (kTruth.beta * (1.0 - eta)));
    CHECK_THAT(exact, WithinRel(398.0, 1e-9));
    CHECK_THAT(mean, WithinAbs(exact, 4.0 * se));

    SECTION("alpha = 0 degenerates to a Poisson process with mean mu T") {
        // E[N] = μT = 200、Var(N) = 200 なので SE = √(200/200) = 1.0、許容 ±4.0
        const HawkesParams poisson{kTruth.mu, 0.0, kTruth.beta};
        Rng                prng(4242);
        double             psum = 0.0;
        for (int r = 0; r < kReps; ++r) psum += static_cast<double>(hawkes_simulate(poisson, kT, prng, buf));
        CHECK_THAT(psum / kReps, WithinAbs(kTruth.mu * kT, 4.0 * std::sqrt(kTruth.mu * kT / kReps)));
    }
}

TEST_CASE("HAWKES-04: a branching ratio alpha/beta >= 1 is rejected and unreachable through the transform",
          "[hawkes][unit]") {
    SECTION("hawkes_stable requires mu > 0, alpha >= 0, beta > 0 and alpha/beta < 1") {
        CHECK(hawkes_stable(kTruth));
        CHECK(hawkes_stable({1e-9, 0.0, 1e-9}));
        CHECK(hawkes_stable({1.0, 0.999, 1.0}));
        CHECK_FALSE(hawkes_stable({1.0, 1.0, 1.0}));    // α/β = 1
        CHECK_FALSE(hawkes_stable({1.0, 2.0, 1.0}));    // α/β = 2（爆発）
        CHECK_FALSE(hawkes_stable({0.0, 0.5, 1.0}));    // μ = 0
        CHECK_FALSE(hawkes_stable({-1.0, 0.5, 1.0}));
        CHECK_FALSE(hawkes_stable({1.0, -0.1, 1.0}));   // α < 0
        CHECK_FALSE(hawkes_stable({1.0, 0.5, 0.0}));    // β = 0
        CHECK_FALSE(hawkes_stable({1.0, 0.5, -1.0}));
        CHECK_FALSE(hawkes_stable({kNaN, 0.5, 1.0}));
        CHECK_FALSE(hawkes_stable({1.0, kNaN, 1.0}));
        CHECK_FALSE(hawkes_stable({1.0, 0.5, kNaN}));
        CHECK_FALSE(hawkes_stable({1.0, kInf, kInf}));
    }

    SECTION("hawkes_simulate refuses an unstable parameter set and generates nothing") {
        std::array<double, 64> out{};
        out.fill(-1.0);
        Rng rng(1);
        for (const HawkesParams bad : {HawkesParams{1.0, 1.0, 1.0}, HawkesParams{1.0, 2.0, 1.0},
                                       HawkesParams{0.0, 0.5, 1.0}, HawkesParams{1.0, 0.5, 0.0},
                                       HawkesParams{kNaN, 0.5, 1.0}}) {
            CHECK(hawkes_simulate(bad, 10.0, rng, out) == 0);
            CHECK(out[0] == -1.0);  // 呼び手のバッファに触れない
        }
        CHECK(hawkes_simulate(kTruth, 0.0, rng, out) == 0);   // T = 0
        CHECK(hawkes_simulate(kTruth, -1.0, rng, out) == 0);  // T < 0
    }

    SECTION("every theta, including extreme and non-finite values, maps into the stable region") {
        const std::array<double, 12> grid{-1e300, -700.0, -30.0, -1.0,  0.0,  1.0,
                                          30.0,   700.0,  1e300, -kInf, kInf, kNaN};
        for (const double t0 : grid)
            for (const double t1 : grid)
                for (const double t2 : grid) {
                    const HawkesParams p = hawkes_from_unconstrained({t0, t1, t2});
                    REQUIRE(hawkes_stable(p));
                    REQUIRE(p.alpha / p.beta < 1.0);
                }
    }

    SECTION("params -> theta -> params round-trips") {
        for (const HawkesParams p : {kTruth, HawkesParams{0.05, 1.9, 2.0}, HawkesParams{3.0, 0.01, 100.0},
                                     HawkesParams{1e-6, 1e-6, 1e-3}, HawkesParams{2.0, 0.0, 5.0}}) {
            const HawkesParams q = hawkes_from_unconstrained(hawkes_to_unconstrained(p));
            CHECK_THAT(q.mu, WithinRel(p.mu, 1e-10));
            CHECK_THAT(q.beta, WithinRel(p.beta, 1e-10));
            CHECK_THAT(q.alpha, WithinRel(p.alpha, 1e-10) || WithinAbs(p.alpha, 1e-300));
        }
    }
}

TEST_CASE("HAWKES-05: thinning is deterministic for a fixed seed and truncates at the buffer capacity",
          "[hawkes][determinism]") {
    constexpr double      kT   = 200.0;
    constexpr std::size_t kCap = 2048;
    const auto run = [](std::uint64_t seed, std::span<double> out) {
        Rng rng(seed);
        return hawkes_simulate(kTruth, kT, rng, out);
    };

    std::array<double, kCap> a{}, b{}, c{};
    const std::size_t        na = run(123, a);
    const std::size_t        nb = run(123, b);
    const std::size_t        nc = run(124, c);
    REQUIRE(na > 100);
    REQUIRE(na < kCap);
    CHECK(nb == na);
    for (std::size_t i = 0; i < na; ++i) REQUIRE(a[i] == b[i]);  // ビット一致
    CHECK(nc != na);                                             // 別 seed は別の系列

    SECTION("a short buffer returns its capacity and holds the same prefix") {
        std::array<double, 16> small{};
        const std::size_t      ns = run(123, small);
        CHECK(ns == small.size());
        for (std::size_t i = 0; i < ns; ++i) REQUIRE(small[i] == a[i]);
    }

    SECTION("an empty buffer generates nothing") {
        const std::span<double> none;
        CHECK(run(123, none) == 0);
    }
}

TEST_CASE("HAWKES-06: the O(n) recursion for the log-likelihood equals the direct O(n^2) sum",
          "[hawkes][numeric]") {
    constexpr double  kT    = 200.0;
    const auto        times = simulate(kTruth, kT, 98765);
    REQUIRE(times.size() > 300);  // T = 200、定常強度 2 なので ≈ 400 事象

    // 生成に使った真値だけでなく、離れた（しかし安定な）パラメータでも一致すること
    for (const HawkesParams p : {kTruth, HawkesParams{0.5, 1.6, 2.0}, HawkesParams{2.0, 0.01, 0.05},
                                 HawkesParams{1.0, 0.0, 1.0}}) {
        const double ll = hawkes_log_likelihood(p, times, kT);
        CHECK(std::isfinite(ll));
        CHECK_THAT(ll, WithinRel(direct_log_likelihood(p, times, kT), 1e-10));
    }

    SECTION("no events on [0, T): the log-likelihood is just -mu T") {
        const std::span<const double> none;
        CHECK_THAT(hawkes_log_likelihood(kTruth, none, 10.0), WithinRel(-kTruth.mu * 10.0, 1e-12));
        CHECK(hawkes_log_likelihood(kTruth, none, 0.0) == 0.0);
    }

    SECTION("an unstable parameter set has log-likelihood -inf") {
        CHECK(hawkes_log_likelihood({1.0, 1.0, 1.0}, times, kT) == -kInf);   // α/β = 1
        CHECK(hawkes_log_likelihood({0.0, 0.5, 1.0}, times, kT) == -kInf);   // μ = 0
        CHECK(hawkes_log_likelihood({1.0, 0.5, -1.0}, times, kT) == -kInf);  // β < 0
    }

    SECTION("the true parameters beat perturbed ones on a long path") {
        // T = 2000（≈ 4000 事象）。摂動幅は下の HAWKES-08 で測る SE の 5 倍以上。
        const auto   big = simulate(kTruth, 2000.0, 5150);
        const double l0  = hawkes_log_likelihood(kTruth, big, 2000.0);
        for (const HawkesParams p : {HawkesParams{1.3, 0.5, 1.0}, HawkesParams{0.7, 0.5, 1.0},
                                     HawkesParams{1.0, 0.8, 1.0}, HawkesParams{1.0, 0.3, 1.0},
                                     HawkesParams{1.0, 0.5, 1.5}, HawkesParams{1.0, 0.5, 0.7}}) {
            REQUIRE(hawkes_log_likelihood(p, big, 2000.0) < l0);
        }
    }
}

TEST_CASE("HAWKES-07: the closed-form compensator equals a numerical integral of the intensity",
          "[hawkes][numeric]") {
    // Λ(T) = ∫_0^T λ = μT + (α/β) Σ (1 − e^{−β(T−t_i)})。数値積分は事象の区間ごとに 4 点
    // Gauss–Legendre を 64 分割で合成する（節点が開区間の内部なので λ の跳びを踏まない）。
    // 指数関数に対する合成 GL4 の誤差は分割幅の 8 乗で落ち、ここでは倍精度の丸め以下。許容は相対 1e-8。
    const auto check = [](HawkesParams p, std::span<const double> times, double T) {
        const auto lambda = [&](double s) { return hawkes_intensity(p, times, s); };
        double     num    = 0.0;
        double     prev   = 0.0;
        for (const double ti : times) {
            num += gauss_legendre(lambda, prev, ti, 64);
            prev = ti;
        }
        num += gauss_legendre(lambda, prev, T, 64);
        CHECK_THAT(hawkes_compensator(p, times, T), WithinRel(num, 1e-8));
    };

    SECTION("a hand-written event list") {
        const std::array<double, 5> times{0.3, 0.7, 1.1, 2.5, 6.0};
        check(kTruth, times, 10.0);
        check({0.5, 1.6, 2.0}, times, 10.0);
        check({2.0, 0.0, 0.05}, times, 10.0);  // α = 0: Λ = μT
        CHECK_THAT(hawkes_compensator({2.0, 0.0, 0.05}, times, 10.0), WithinRel(20.0, 1e-12));
    }

    SECTION("a simulated path") {
        const auto times = simulate(kTruth, 30.0, 246810);
        REQUIRE(times.size() > 30);
        check(kTruth, times, 30.0);
        check({0.5, 1.6, 2.0}, times, 30.0);
    }

    SECTION("no events: the compensator is mu T") {
        const std::span<const double> none;
        CHECK_THAT(hawkes_compensator(kTruth, none, 7.5), WithinRel(kTruth.mu * 7.5, 1e-12));
        CHECK(hawkes_compensator(kTruth, none, 0.0) == 0.0);
    }
}

namespace {

/// 最尤点での対数尤度の数値ヘッセ行列から推定 SE を作る: Cov ≈ (−∇²LL)^{−1}、SE_k = √Cov_kk。
/// 刻みは h_k = 1e-4·max(1, |x_k|)。二階中心差分の最適刻みは ε^{1/4} ≈ 1.2e-4 で、
/// 打ち切り誤差 O(h²) と丸め誤差 ε|LL|/h² が釣り合う（|LL| ≈ 3e3 で相対誤差 ≈ 1e-8）。
std::array<double, 3> mle_standard_errors(HawkesParams p, std::span<const double> times, double T) {
    const std::array<double, 3> x{p.mu, p.alpha, p.beta};
    std::array<double, 3>       h{};
    for (std::size_t i = 0; i < 3; ++i) h[i] = 1e-4 * std::max(1.0, std::fabs(x[i]));
    const auto f = [&](std::array<double, 3> v) {
        return hawkes_log_likelihood(HawkesParams{v[0], v[1], v[2]}, times, T);
    };
    const double f0 = f(x);
    Mat<3, 3>    info{};  // −∇²LL（観測情報行列）
    for (std::size_t i = 0; i < 3; ++i) {
        auto xp = x, xm = x;
        xp[i] += h[i];
        xm[i] -= h[i];
        info(i, i) = -(f(xp) - 2.0 * f0 + f(xm)) / (h[i] * h[i]);
        for (std::size_t j = i + 1; j < 3; ++j) {
            auto pp = x, pm = x, mp = x, mm = x;
            pp[i] += h[i];
            pp[j] += h[j];
            pm[i] += h[i];
            pm[j] -= h[j];
            mp[i] -= h[i];
            mp[j] += h[j];
            mm[i] -= h[i];
            mm[j] -= h[j];
            const double d = -(f(pp) - f(pm) - f(mp) + f(mm)) / (4.0 * h[i] * h[j]);
            info(i, j)     = d;
            info(j, i)     = d;
        }
    }
    const auto cov = info.inverse();
    REQUIRE(cov.has_value());
    std::array<double, 3> se{};
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE((*cov)(i, i) > 0.0);
        se[i] = std::sqrt((*cov)(i, i));
    }
    return se;
}

}  // namespace

TEST_CASE("HAWKES-08: MLE recovers mu, alpha and beta from 5000 events within 4 estimated SE",
          "[hawkes][statistical]") {
    // 真値 μ = 1, α = 0.5, β = 1（分岐比 0.5）、T = 2500 → n ≈ 5000 事象、seed 固定。
    // 許容は最尤点での対数尤度の数値ヘッセ行列 (−∇²LL)^{−1} から作った推定 SE の 4 倍
    // （§2.4「MLE パラメータ復元 = 推定 SE の 4 倍」）。初期値は真値から離した固定値。
    constexpr double kT    = 2500.0;
    const auto       times = simulate(kTruth, kT, 20260912);
    REQUIRE(times.size() > 4500);
    REQUIRE(times.size() < 5500);

    const double                ll_truth = hawkes_log_likelihood(kTruth, times, kT);
    const std::array<double, 3> se       = mle_standard_errors(kTruth, times, kT);
    INFO("SE(mu) = " << se[0] << ", SE(alpha) = " << se[1] << ", SE(beta) = " << se[2]);
    for (const double v : se) REQUIRE(v < 0.5);  // n = 5000 で SE は 0.1 の桁（4 SE でも真値の近傍）

    const HawkesParams init{2.0, 0.2, 2.0};  // μ 2 倍、β 2 倍、分岐比 0.1（真値は 0.5）
    const auto         check = [&](const HawkesFit& fit) {
        INFO("mu = " << fit.params.mu << ", alpha = " << fit.params.alpha << ", beta = " << fit.params.beta
                     << ", iters = " << fit.optim.iters);
        CHECK(fit.optim.converged);
        CHECK(hawkes_stable(fit.params));
        CHECK_THAT(fit.params.mu, WithinAbs(kTruth.mu, 4.0 * se[0]));
        CHECK_THAT(fit.params.alpha, WithinAbs(kTruth.alpha, 4.0 * se[1]));
        CHECK_THAT(fit.params.beta, WithinAbs(kTruth.beta, 4.0 * se[2]));
        // 最尤点の尤度は真値の尤度以上（局所解や早期停止で止まっていない）
        CHECK(fit.log_lik >= ll_truth - 1e-6);
        // 返却値の整合: log_lik = LL(params) = −optim.f、params = 変換(optim.x)、軌跡の終点 = 解
        CHECK_THAT(fit.log_lik, WithinRel(hawkes_log_likelihood(fit.params, times, kT), 1e-12));
        CHECK_THAT(fit.log_lik, WithinRel(-fit.optim.f, 1e-12));
        const HawkesParams from_x = hawkes_from_unconstrained(fit.optim.x);
        CHECK(from_x.mu == fit.params.mu);
        CHECK(from_x.alpha == fit.params.alpha);
        CHECK(from_x.beta == fit.params.beta);
        REQUIRE_FALSE(fit.optim.path.empty());
        CHECK(fit.optim.path.back() == fit.optim.x);
    };
    SECTION("Nelder-Mead") { check(hawkes_fit(times, kT, init)); }
    SECTION("BFGS") { check(hawkes_fit(times, kT, init, OptimOptions{}, true)); }

    SECTION("degenerate starts (alpha = 0, branching ratio at the boundary, absurd beta) still recover") {
        for (const HawkesParams bad : {HawkesParams{5.0, 0.0, 1.0}, HawkesParams{0.1, 1.0, 1.0},
                                       HawkesParams{1.0, 0.5, 100.0}, HawkesParams{1.0, 0.5, 0.01}}) {
            for (const bool use_bfgs : {false, true}) {
                const HawkesFit fit = hawkes_fit(times, kT, bad, OptimOptions{}, use_bfgs);
                INFO("start mu = " << bad.mu << ", alpha = " << bad.alpha << ", beta = " << bad.beta
                                   << ", bfgs = " << use_bfgs);
                REQUIRE_THAT(fit.params.mu, WithinAbs(kTruth.mu, 4.0 * se[0]));
                REQUIRE_THAT(fit.params.alpha, WithinAbs(kTruth.alpha, 4.0 * se[1]));
                REQUIRE_THAT(fit.params.beta, WithinAbs(kTruth.beta, 4.0 * se[2]));
                REQUIRE(fit.log_lik >= ll_truth - 1e-6);
            }
        }
    }

    SECTION("a warm start from the previous estimate reproduces the same optimum") {
        const HawkesFit cold = hawkes_fit(times, kT, init);
        const HawkesFit warm = hawkes_fit(times, kT, cold.params);
        check(warm);
        CHECK(warm.optim.iters < cold.optim.iters);  // 近くから始めれば反復は減る
        CHECK_THAT(warm.log_lik, WithinAbs(cold.log_lik, 1e-6));
    }

    SECTION("hawkes_fit_into reuses the result's allocation and matches hawkes_fit") {
        // 計算スレッド用の入口: 同じ HawkesFit を使い回すと path の capacity が再利用され、
        // 2 回目以降はヒープ確保が起きない（再試行が走った場合も scratch が同様に再利用される）。
        HawkesFit out;
        hawkes_fit_into(out, times, kT, init);
        check(out);
        const HawkesFit ref = hawkes_fit(times, kT, init);
        CHECK(out.optim.iters == ref.optim.iters);
        CHECK(out.optim.x == ref.optim.x);
        CHECK(out.log_lik == ref.log_lik);

        const HawkesFit first = out;
        const auto      cap   = out.optim.path.capacity();
        REQUIRE(cap >= out.optim.path.size());
        hawkes_fit_into(out, times, kT, out.params);  // ウォームスタートで再当てはめ
        check(out);
        CHECK(out.optim.path.capacity() == cap);
        hawkes_fit_into(out, times, kT, HawkesParams{5.0, 0.0, 1.0}, OptimOptions{}, true);  // BFGS・退化
        check(out);
        CHECK(out.optim.path.capacity() == cap);
        hawkes_fit_into(out, times, kT, init);  // 最初と同じ入力: 再利用しても結果はビット一致
        CHECK(out.optim.path.capacity() == cap);
        CHECK(out.log_lik == first.log_lik);
        CHECK(out.optim.x == first.optim.x);
        CHECK(out.optim.path == first.optim.path);
    }

    SECTION("degenerate data (no events, T <= 0) returns init unchanged with converged = false") {
        const std::span<const double> none;
        for (const HawkesFit& fit : {hawkes_fit(none, kT, init), hawkes_fit(times, 0.0, init),
                                    hawkes_fit(times, -1.0, init)}) {
            CHECK(fit.params.mu == init.mu);
            CHECK(fit.params.alpha == init.alpha);
            CHECK(fit.params.beta == init.beta);
            CHECK_FALSE(fit.optim.converged);
            CHECK(fit.optim.iters == 0);
            REQUIRE(fit.optim.path.size() == 1);
            CHECK(fit.optim.path.back() == fit.optim.x);
        }
    }
}

TEST_CASE("HAWKES-09: rescaled residuals have mean 1 within 4 SE (unit exponential)", "[hawkes][statistical]") {
    // 時間再スケール定理: 真のパラメータで τ_i = Λ(t_i) − Λ(t_{i−1}) は iid 単位指数分布になる。
    // T = 2500、μ=1, α=0.5, β=1（定常強度 2）で n ≈ 5000 事象、seed 固定。
    // 平均の SE = 1/√n（単位指数の sd = 1）≈ 0.0141 なので許容 ±4/√n ≈ ±0.057（§2.4）。
    // 分散の SE = √((μ4 − σ⁴)/n) = √(8/n) ≈ 0.040（単位指数の 4 次モーメント 9）で許容 ±0.16。
    constexpr double kT    = 2500.0;
    const auto       times = simulate(kTruth, kT, 31415926);
    REQUIRE(times.size() > 4500);
    REQUIRE(times.size() < 5500);

    std::vector<double> tau(times.size());
    const std::size_t   n = hawkes_rescaled_residuals(kTruth, times, tau);
    REQUIRE(n == times.size());

    double sum = 0.0, sum2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE(tau[i] > 0.0);
        sum += tau[i];
        sum2 += tau[i] * tau[i];
    }
    const double nd   = static_cast<double>(n);
    const double mean = sum / nd;
    const double var  = (sum2 - sum * mean) / (nd - 1.0);
    INFO("n = " << n << ", mean = " << mean << ", var = " << var);
    CHECK_THAT(mean, WithinAbs(1.0, 4.0 / std::sqrt(nd)));          // 平均 ≈ 1
    CHECK_THAT(var, WithinAbs(1.0, 4.0 * std::sqrt(8.0 / nd)));     // 分散 ≈ 1（おまけ）

    SECTION("the residuals telescope to the compensator at the last event") {
        CHECK_THAT(sum, WithinRel(hawkes_compensator(kTruth, times, times.back()), 1e-10));
    }

    SECTION("a short output buffer stops at its capacity") {
        std::array<double, 10> few{};
        CHECK(hawkes_rescaled_residuals(kTruth, times, few) == few.size());
        for (std::size_t i = 0; i < few.size(); ++i) REQUIRE(few[i] == tau[i]);
        const std::span<double>       nowhere;
        const std::span<const double> none;
        CHECK(hawkes_rescaled_residuals(kTruth, times, nowhere) == 0);
        CHECK(hawkes_rescaled_residuals(kTruth, none, tau) == 0);
    }

    SECTION("unstable parameters write nothing (the residuals would be meaningless)") {
        std::array<double, 8> out{};
        out.fill(-1.0);
        for (const HawkesParams bad : {HawkesParams{1.0, 1.0, 1.0}, HawkesParams{1.0, 0.5, 0.0},
                                       HawkesParams{0.0, 0.5, 1.0}, HawkesParams{kNaN, 0.5, 1.0}}) {
            CHECK(hawkes_rescaled_residuals(bad, times, out) == 0);
            CHECK(out[0] == -1.0);  // 呼び手のバッファに触れない
        }
    }
}

