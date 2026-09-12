#pragma once
// core/exec/almgren_chriss.hpp — Almgren–Chriss (2000) の離散最適執行（売り切りプログラム）。
// 状態も確保も持たない純関数群で、執行シーンはスライダが動くたびに 8 本の軌道と 32 点の
// フロンティアを丸ごと引き直す（1 評価あたり O(n_steps) の sinh、割り当てゼロ）。
//
// モデル（tau = T/N, t_j = j tau, j = 0..N。x_j = 時刻 t_j の残量、x_0 = X, x_N = 0、
// n_j = x_{j-1} - x_j が区間 j の取引量）:
//
//   価格:     S_j    = S_{j-1} + sigma sqrt(tau) xi_j - tau g(n_j/tau)     （算術ブラウン運動）
//   約定価格: S~_j   = S_{j-1} - h(n_j/tau)
//   恒久:     g(v)   = gamma v            =>  tau g(n_j/tau) = gamma n_j
//   一時:     h(v)   = epsilon sgn(v) + eta v   =>  h(n_j/tau) = epsilon sgn(n_j) + eta n_j / tau
//   実装ショートフォール: C = X S_0 - sum_j n_j S~_j
//
// これを展開すると（sum_j n_j = X, sum_j n_j (X - x_{j-1}) = X^2/2 - (1/2) sum_j n_j^2 を使う）
//
//   E[C] = (1/2) gamma X^2 + epsilon sum_j |n_j| + (eta~ / tau) sum_j n_j^2,   eta~ = eta - gamma tau / 2
//   V[C] = sigma^2 tau sum_{j=1..N} x_j^2
//
// 売り切り（X > 0, x が単調減少）では sum_j |n_j| = X なので epsilon 項は epsilon X に潰れる。
// C は独立な xi_j の線形結合 + 定数なので厳密に正規分布であり、上の 2 つが分布を決め切る。
//
// 最適軌道: E[C] + lambda V[C] を x_1..x_{N-1} について最小化すると
//
//   kappa~^2 = lambda sigma^2 / eta~,   cosh(kappa tau) = 1 + kappa~^2 tau^2 / 2
//   x_j = X sinh(kappa (T - t_j)) / sinh(kappa T)
//
// kappa の解き方: 教科書形の kappa = arccosh(1 + kappa~^2 tau^2 / 2) / tau は kappa~ tau -> 0 で
// arccosh(1 + z) の引数が 1 に潰れて桁落ちする（z ~ 1e-16 で有効桁が消える）。恒等式
// cosh(u) = 1 + 2 sinh^2(u/2) から 2 sinh^2(kappa tau / 2) = kappa~^2 tau^2 / 2、すなわち
//
//   kappa = (2 / tau) asinh(kappa~ tau / 2)                      ← 実装はこちら
//
// を使えば全域で相対誤差が丸め程度に収まり、kappa~ tau -> 0 で kappa -> kappa~ が自動的に出る
// （asinh(x) = x - x^3/6 + ... なので kappa = kappa~ (1 - kappa~^2 tau^2 / 24 + ...)）。
// lambda = 0 または sigma = 0 では kappa~ = 0 となり kappa は厳密に 0 になる。
//
// kappa T の小ささへの対処: kappa = 0 のとき sinh 比は 0/0 なので、kappa T < kAcLinearKappaT (=1e-6)
// では線形（TWAP）x_j = X (N - j)/N に落とす。切替点での食い違いは相対 (kappa T)^2/6 < 1.7e-13 で、
// 閉形式との比較許容（相対 1e-10）より十分小さい。
//
// kappa T の大きさへの対処: sinh(kappa (T - t_j)) / sinh(kappa T) をそのまま計算すると、
// kappa T が 710 を越えたところで分子も分母も inf になり、inf/inf = NaN が軌道・E[C]・V[C] の
// 全体に伝播する。これはクランプ域の内側で起きる（T = 250, N = 256, lambda = 1e-3 で
// kappa T = 1510、T = 5, N = 1024, eta = 1e-9, lambda = 1e-4 で kappa T = 2048）ので、
// a = kappa (T - t_j), b = kappa T（0 <= a <= b）と置いて恒等式
//
//   sinh a / sinh b = e^{a-b} (1 - e^{-2a}) / (1 - e^{-2b}) = e^{a-b} expm1(-2a) / expm1(-2b)
//
// を使う。e^{a-b} <= 1 も |expm1(-2a)| <= 1 も有界なのでオーバーフローしない（kappa T > 745 では
// e^{a-b} が素直に 0 へアンダーフローし、x_j は 0 に潰れて軌道は単調非増加のまま残る）。
// expm1 を使うので a -> 0 の桁落ちも無い。1 / expm1(-2 kappa T) は AcDerived に 1 回だけ持つ。
// 内点（0 < j < N）では t_j = j fl(T/N) < T が丸めを含めて成り立つので a > 0 は保証される。
//
// eta~ <= 0（1 ステップの間に恒久インパクトが一時インパクトを食い尽くす退化領域）は kAcMinEtaTilde
// で床を入れる。床が効いたときだけ ac_cost（eta~ を使う）と ac_cost_mc（eta と gamma を別々に使う）
// の期待値がずれる。正規のパラメータでは eta~ = eta - gamma tau / 2 が厳密に成り立ち両者は一致する。
//
// 定義域: 公開関数はすべて入口で ac_sanitize を通すので、NaN や負の入力でも有限値を返す
// （ac_sanitize は冪等）。クランプ域は「金融的にあり得る範囲」で切ってあり、上限まで振っても
// kappa~^2 tau^2 / 2 が inf にならないようにしてある。
//
// 参考: R. Almgren and N. Chriss, "Optimal execution of portfolio transactions",
//       Journal of Risk 3(2), 5–39, 2000（§2 の離散モデル、§3 の数値例が既定パラメータ）。

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include "quantviz/core/rng.hpp"

namespace quantviz::core {

/// 執行問題のパラメータ。既定値は Almgren–Chriss (2000) §3 の数値例で、ac_sanitize が
/// NaN を戻す先でもある。
struct AcParams {
    double      X       = 1.0e6;    ///< 執行する株数（売り切り）。> 0
    double      T       = 5.0;      ///< 執行期間（例では日）。> 0
    double      sigma   = 0.95;     ///< 価格ボラティリティ（絶対、価格単位 / sqrt(時間)）。>= 0
    double      eta     = 2.5e-6;   ///< 一時インパクト係数（価格 / (株 / 時間)）。> 0
    double      gamma   = 2.5e-7;   ///< 恒久インパクト係数（価格 / 株）。>= 0
    double      epsilon = 0.0625;   ///< 固定コスト（スプレッド半値 + 手数料、価格単位）。>= 0
    double      lambda  = 2.0e-6;   ///< リスク回避度。0 で TWAP、大きいほど前倒し。>= 0
    std::size_t n_steps = 5;        ///< 区間数 N。[1, kAcMaxSteps]
};

/// 実装ショートフォール C の期待値と分散。
struct AcCost {
    double expected = 0.0;
    double variance = 0.0;
};

// ac_sanitize のクランプ域。上限は kappa~^2 tau^2 / 2 と E[C] / V[C] が有限に留まる範囲。
inline constexpr double      kAcMinX        = 1e-9;
inline constexpr double      kAcMaxX        = 1e12;
inline constexpr double      kAcMinT        = 1e-9;
inline constexpr double      kAcMaxT        = 1e6;
inline constexpr double      kAcMaxSigma    = 1e6;
inline constexpr double      kAcMinEta      = 1e-15;
inline constexpr double      kAcMaxEta      = 1e6;
inline constexpr double      kAcMaxGamma    = 1e6;
inline constexpr double      kAcMaxEpsilon  = 1e6;
inline constexpr double      kAcMaxLambda   = 1e12;
inline constexpr std::size_t kAcMaxSteps    = 4096;
/// eta~ = eta - gamma tau / 2 の床（退化領域の保護。正規のパラメータでは決して効かない）。
inline constexpr double kAcMinEtaTilde = kAcMinEta;
/// これより小さい kappa T では sinh 比を使わず線形（TWAP）に落とす。
inline constexpr double kAcLinearKappaT = 1e-6;

namespace detail {

/// [lo, hi] に丸める。NaN は fallback へ。比較は !(v >= lo) の形にして NaN を必ず弾く。
constexpr double ac_clamp(double v, double lo, double hi, double fallback) noexcept {
    if (!(v == v)) return fallback;  // NaN
    if (!(v >= lo)) return lo;
    if (v > hi) return hi;
    return v;
}

}  // namespace detail

/// UI 由来の値を定義域に丸める。NaN は AcParams の既定値へ、範囲外は境界へ。冪等。
[[nodiscard]] inline AcParams ac_sanitize(AcParams p) noexcept {
    constexpr AcParams d{};
    p.X       = detail::ac_clamp(p.X, kAcMinX, kAcMaxX, d.X);
    p.T       = detail::ac_clamp(p.T, kAcMinT, kAcMaxT, d.T);
    p.sigma   = detail::ac_clamp(p.sigma, 0.0, kAcMaxSigma, d.sigma);
    p.eta     = detail::ac_clamp(p.eta, kAcMinEta, kAcMaxEta, d.eta);
    p.gamma   = detail::ac_clamp(p.gamma, 0.0, kAcMaxGamma, d.gamma);
    p.epsilon = detail::ac_clamp(p.epsilon, 0.0, kAcMaxEpsilon, d.epsilon);
    p.lambda  = detail::ac_clamp(p.lambda, 0.0, kAcMaxLambda, d.lambda);
    p.n_steps = std::clamp<std::size_t>(p.n_steps, 1, kAcMaxSteps);
    return p;
}

namespace detail {

/// sanitize 済みパラメータから 1 回だけ引いておく派生量。
struct AcDerived {
    double tau         = 0.0;  ///< T / N
    double eta_tilde   = 0.0;  ///< eta - gamma tau / 2（kAcMinEtaTilde で床）
    double kappa_tilde = 0.0;  ///< sqrt(lambda sigma^2 / eta~)（連続時間の kappa）
    double kappa       = 0.0;  ///< (2/tau) asinh(kappa~ tau / 2)（離散の kappa）
    double inv_denom   = 0.0;  ///< 1 / expm1(-2 kappa T)（線形分岐では 0）
    bool   linear      = true; ///< kappa T < kAcLinearKappaT なら TWAP に落とす
};

[[nodiscard]] inline AcDerived ac_derive(const AcParams& p) noexcept {
    AcDerived d;
    d.tau       = p.T / static_cast<double>(p.n_steps);
    d.eta_tilde = p.eta - 0.5 * p.gamma * d.tau;
    if (!(d.eta_tilde > kAcMinEtaTilde)) d.eta_tilde = kAcMinEtaTilde;
    d.kappa_tilde = std::sqrt(p.lambda * p.sigma * p.sigma / d.eta_tilde);
    d.kappa       = 2.0 * std::asinh(0.5 * d.kappa_tilde * d.tau) / d.tau;
    d.linear      = !(d.kappa * p.T >= kAcLinearKappaT);
    d.inv_denom   = d.linear ? 0.0 : 1.0 / std::expm1(-2.0 * d.kappa * p.T);
    return d;
}

/// 残量 x_j（0 < j < N）。端点 x_0 = X, x_N = 0 は呼び手が厳密値で埋める。
[[nodiscard]] inline double ac_x_at(const AcParams& p, const AcDerived& d, std::size_t j) noexcept {
    if (d.linear) return p.X * (static_cast<double>(p.n_steps - j) / static_cast<double>(p.n_steps));
    const double a = d.kappa * (p.T - static_cast<double>(j) * d.tau);  // 0 < a <= kappa T
    return p.X * std::exp(a - d.kappa * p.T) * std::expm1(-2.0 * a) * d.inv_denom;
}

}  // namespace detail

/// eta~ = eta - gamma tau / 2（kAcMinEtaTilde で床を入れた値）。
[[nodiscard]] inline double ac_eta_tilde(AcParams p) noexcept {
    p = ac_sanitize(p);
    return detail::ac_derive(p).eta_tilde;
}

/// 連続時間の kappa~ = sqrt(lambda sigma^2 / eta~)。tau -> 0 で ac_kappa の極限。
[[nodiscard]] inline double ac_kappa_tilde(AcParams p) noexcept {
    p = ac_sanitize(p);
    return detail::ac_derive(p).kappa_tilde;
}

/// 離散の kappa。cosh(kappa tau) - 1 = kappa~^2 tau^2 / 2 を満たす唯一の正根
/// （lambda = 0 / sigma = 0 では厳密に 0）。実装は等価で安定な (2/tau) asinh(kappa~ tau / 2)。
[[nodiscard]] inline double ac_kappa(AcParams p) noexcept {
    p = ac_sanitize(p);
    return detail::ac_derive(p).kappa;
}

/// 最適軌道 x_0..x_N（n_steps + 1 点）を書き、書いた点数を返す。span が短ければ 0 を返し何も書かない。
[[nodiscard]] inline std::size_t ac_trajectory(AcParams p, std::span<double> x_out) noexcept {
    p                   = ac_sanitize(p);
    const std::size_t m = p.n_steps + 1;
    if (x_out.size() < m) return 0;
    const detail::AcDerived d = detail::ac_derive(p);
    x_out[0]                  = p.X;
    for (std::size_t j = 1; j < p.n_steps; ++j) x_out[j] = detail::ac_x_at(p, d, j);
    x_out[p.n_steps] = 0.0;
    return m;
}

/// 区間ごとの取引量 n_j = x_{j-1} - x_j（n_steps 点）を書き、書いた点数を返す。
[[nodiscard]] inline std::size_t ac_trades(AcParams p, std::span<double> n_out) noexcept {
    p = ac_sanitize(p);
    if (n_out.size() < p.n_steps) return 0;
    const detail::AcDerived d    = detail::ac_derive(p);
    double                  prev = p.X;
    for (std::size_t j = 1; j <= p.n_steps; ++j) {
        const double cur = (j == p.n_steps) ? 0.0 : detail::ac_x_at(p, d, j);
        n_out[j - 1]     = prev - cur;
        prev             = cur;
    }
    return p.n_steps;
}

/// 閉形式の E[C], V[C]。sum_j |n_j| = X（売り切り）を使って epsilon 項を epsilon X に潰す。
[[nodiscard]] inline AcCost ac_cost(AcParams p) noexcept {
    p                            = ac_sanitize(p);
    const detail::AcDerived d    = detail::ac_derive(p);
    double                  sn2  = 0.0;  // sum n_j^2
    double                  sx2  = 0.0;  // sum_{j=1..N} x_j^2
    double                  prev = p.X;
    for (std::size_t j = 1; j <= p.n_steps; ++j) {
        const double cur = (j == p.n_steps) ? 0.0 : detail::ac_x_at(p, d, j);
        const double n   = prev - cur;
        sn2 += n * n;
        sx2 += cur * cur;
        prev = cur;
    }
    AcCost c;
    c.expected = 0.5 * p.gamma * p.X * p.X + p.epsilon * p.X + d.eta_tilde * sn2 / d.tau;
    c.variance = p.sigma * p.sigma * d.tau * sx2;
    return c;
}

/// 最適軌道に沿って価格経路を実際に回した実装ショートフォールの標本平均・標本分散（n-1 で割る）。
/// 検証用。確保なし・逐次で、軌道は毎ステップ閉形式から引き直す。同じ seed ならビット一致する。
/// n_sim = 0 は {0, 0} を返す（標本が無い）。n_sim = 1 の分散も 0。
/// 引く乱数は n_sim x (n_steps - 1) 本: 最終区間の xi_N は x_N = 0 の係数を持つので
/// ショートフォールに寄与せず、引かない。
[[nodiscard]] inline AcCost ac_cost_mc(AcParams p, std::uint64_t seed, std::size_t n_sim) noexcept {
    p                              = ac_sanitize(p);
    const detail::AcDerived d      = detail::ac_derive(p);
    const double            sq_tau = std::sqrt(d.tau);
    Rng                     rng(seed);

    double      mean = 0.0;
    double      m2   = 0.0;
    std::size_t cnt  = 0;
    for (std::size_t s = 0; s < n_sim; ++s) {
        double dev  = 0.0;  // S_j - S_0（S_0 は C から消えるので偏差だけ回す）
        double cost = 0.0;
        double prev = p.X;
        for (std::size_t j = 1; j <= p.n_steps; ++j) {
            const double cur  = (j == p.n_steps) ? 0.0 : detail::ac_x_at(p, d, j);
            const double n    = prev - cur;
            const double sgn  = (n > 0.0) ? 1.0 : ((n < 0.0) ? -1.0 : 0.0);
            const double exec = dev - (p.epsilon * sgn + p.eta * n / d.tau);  // S~_j - S_0
            cost += n * (0.0 - exec);                                         // n_j (S_0 - S~_j)
            // j = N の更新は読まれない（x_N = 0）ので乱数ごと省く。
            if (j < p.n_steps) dev += p.sigma * sq_tau * rng.normal() - p.gamma * n;
            prev = cur;
        }
        ++cnt;
        const double delta = cost - mean;  // Welford
        mean += delta / static_cast<double>(cnt);
        m2 += delta * (cost - mean);
    }

    AcCost c;
    c.expected = mean;
    c.variance = cnt > 1 ? m2 / static_cast<double>(cnt - 1) : 0.0;
    return c;
}

/// base の lambda だけを差し替えて ac_cost を評価する（効率的フロンティア）。
/// 書いた点数 = min(lambdas.size(), out.size()) を返す。
[[nodiscard]] inline std::size_t ac_frontier(AcParams base, std::span<const double> lambdas,
                                             std::span<AcCost> out) noexcept {
    const std::size_t m = std::min(lambdas.size(), out.size());
    for (std::size_t i = 0; i < m; ++i) {
        AcParams p = base;
        p.lambda   = lambdas[i];
        out[i]     = ac_cost(p);
    }
    return m;
}

}  // namespace quantviz::core
