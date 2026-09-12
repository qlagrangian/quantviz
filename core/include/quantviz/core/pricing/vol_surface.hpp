#pragma once
// core/pricing/vol_surface.hpp — Gatheral–Jacquier の SSVI（Surface SVI）インプライド・ボラ面。
// パラメータ 4 本から (k, T) 格子上の総分散とインプライド・ボラを閉形式で返すだけの純関数群で、
// 状態も割り当ても持たない（ボラ面シーンはパラメータが変わるたびに 64×32 点を舐める）。
//
//   w(k, θ) = θ/2 { 1 + ρ φ(θ) k + sqrt((φ(θ) k + ρ)² + 1 − ρ²) },   θ(T) = σ_atm² T
//   φ(θ)    = η / θ^γ                                                 （power-law パラメータ化）
//   iv(k, T) = sqrt(w(k, T) / T)
//
// k は log-moneyness log(K/F)、w は総分散（= iv² T）。θ(T) は ATM の総分散で、w(0, θ) = θ が
// ρ, φ に依らず成り立つ。ρ はスキュー（ρ < 0 なら put 側が高い）、η は φ の水準（＝ 翼の傾き）、
// γ は φ の減衰速度（γ = 0 で満期に依らず一定、γ = 1 で θφ(θ) = η が一定）。
//
// 定義域の責任分担: パラメータ（σ_atm, ρ, η, γ）は UI 由来でも必ず ssvi_clamp を通す。評価関数の
// 側はパラメータを正規化しない（|ρ| ≥ 1 だと 1 − ρ² < 0 で sqrt が NaN になる）。一方 k と T は
// 呼び出し側の責任で有限値を渡すこと: k = NaN はそのまま NaN として伝播し、T = +inf は
// w = inf・iv = sqrt(inf/inf) = NaN になる。ただし T ≤ 0 と T = NaN だけは床に落として
// 有限の数を返す（格子が T = 0 を含んでも落ちないため。パラメータの NaN が既定値に戻るのとは
// 別の扱いで、T の NaN は「既定の T」ではなく kSsviMinT になる）。
//
// 床の入れ方: γ > 0 では θ → 0 で φ → ∞ なので、φ を評価する前に θ に kSsviMinTheta の床を、
// iv の割り算の前に T に kSsviMinT の床を入れる。kSsviMinTheta は「クランプ済みパラメータと
// T ≥ kSsviMinT から到達できる最小の θ」= kSsviMinSigmaAtm² · kSsviMinT に一致させてあるので、
// 正規の入力では床は決して発火せず、恒等式 w(0, T) = σ_atm² T は常に厳密に成り立つ。床が効くのは
// T ≤ 0 / T = NaN / 未クランプのパラメータのときだけで、そこでの値は数値的な保護であって
// 満期直前の外挿として意味を持たせてはいけない。
//
// 正値性: x = φ(θ) k と置くと括弧の中は 1 + ρx + sqrt(x² + 2ρx + 1)。1 + ρx ≥ 0 なら自明に正、
// 1 + ρx < 0 でも (1 + ρx)² = 1 + 2ρx + ρ²x² < x² + 2ρx + 1（|ρ| < 1）より sqrt の方が大きい。
// よって θ > 0 かつ |ρ| < 1 の全 (k, T) で w > 0、iv > 0（VOLSURF-01）。数値的には 1 + ρx と
// sqrt(·) の桁落ちが起きるが、クランプ後の |ρ| ≤ 0.999 では打ち消し後の値が元の大きさの
// 1e-3 倍以上残るので、倍精度の丸め（相対 1e-16）で符号が変わることはない。
//
// 参考: J. Gatheral and A. Jacquier, "Arbitrage-free SVI volatility surfaces",
//       Quantitative Finance 14(1), 59–71, 2014（§4 の SSVI）。

#include <cmath>

namespace quantviz::core {

/// SSVI のパラメータ。既定値は ssvi_clamp が NaN を戻す先でもある。
struct SsviParams {
    double sigma_atm = 0.2;   ///< ATM ボラ。θ(T) = sigma_atm² T。> 0
    double rho       = -0.3;  ///< スキュー。(−1, 1)
    double eta       = 1.0;   ///< φ の水準。> 0
    double gamma     = 0.5;   ///< φ の減衰指数。[0, 1]
};

// ssvi_clamp の許容域。上限は「金融的にあり得る範囲」で切る（DBL_MAX まで許すと σ_atm² T や
// η/θ^γ が inf になり、VOLSURF-01 の「クランプを通せば必ず有限・正」が成り立たなくなる）。
inline constexpr double kSsviMinSigmaAtm = 1e-6;
inline constexpr double kSsviMaxSigmaAtm = 5.0;    ///< 年率 500% ボラ
inline constexpr double kSsviMaxAbsRho   = 0.999;  ///< |ρ| の上限（1 に触れると sqrt が壊れる）
inline constexpr double kSsviMinEta      = 1e-6;
inline constexpr double kSsviMaxEta      = 100.0;
inline constexpr double kSsviMinGamma    = 0.0;
inline constexpr double kSsviMaxGamma    = 1.0;

/// 翼のバンド η (1 + |ρ|) ≤ kSsviMaxEtaBand（Gatheral–Jacquier Thm 4.2 / Remark 4.4 由来）。
inline constexpr double kSsviMaxEtaBand = 2.0;

inline constexpr double kSsviMinT = 1e-12;  ///< w / T の割り算を止める T の床
/// φ(θ) = η/θ^γ の発散を止める θ の床。クランプ済み定義域で到達できる最小の θ に一致させる。
inline constexpr double kSsviMinTheta = kSsviMinSigmaAtm * kSsviMinSigmaAtm * kSsviMinT;

namespace detail {

/// [lo, hi] に丸める。NaN は fallback へ。比較は !(v > lo) の形にして NaN を必ず弾く。
constexpr double ssvi_clamp_field(double v, double lo, double hi, double fallback) noexcept {
    if (!(v == v)) return fallback;  // NaN
    if (!(v > lo)) return lo;
    if (!(v < hi)) return hi;
    return v;
}

/// θ(T) = σ_atm² T に床を入れたもの（T ≤ 0 や NaN もここで床に落ちる）。
inline double ssvi_theta(double T, const SsviParams& p) noexcept {
    const double theta = p.sigma_atm * p.sigma_atm * T;
    return theta > kSsviMinTheta ? theta : kSsviMinTheta;
}

/// φ(θ) = η / θ^γ。θ は ssvi_theta で床を入れた値であること。
inline double ssvi_phi(double theta, const SsviParams& p) noexcept {
    return p.eta / std::pow(theta, p.gamma);
}

/// (A) カレンダー・スプレッド裁定なし（Gatheral–Jacquier 2014, Theorem 4.1）。同定理の条件は
///     (i) ∂_t θ_t ≥ 0、(ii) 0 ≤ ∂_θ(θ φ(θ)) ≤ (1 + sqrt(1 − ρ²)) φ(θ) / ρ²（全 θ > 0）。
///     本ファイルの θ_t = σ_atm² t と power-law φ(θ) = η θ^{−γ} では
///       (i) ⟺ σ_atm > 0、
///       (ii) ∂_θ(θ φ(θ)) = (1 − γ) φ(θ) なので、η > 0 と γ ∈ [0, 1] があれば
///            0 ≤ (1 − γ) φ ≤ φ ≤ (1 + sqrt(1 − ρ²)) φ / ρ²（右端は |ρ| ≤ 1 で常に φ 以上）。
///     すなわち下の 4 条件と (A) は同値で、θ に依らず（= 全満期で）成立する。
inline bool ssvi_calendar_ok(const SsviParams& p) noexcept {
    if (!(p.sigma_atm > 0.0)) return false;  // Thm 4.1 (i): ∂_t θ_t > 0
    if (!(p.eta > 0.0)) return false;        // φ > 0
    if (!(p.gamma >= kSsviMinGamma) || !(p.gamma <= kSsviMaxGamma)) return false;  // Thm 4.1 (ii)
    return std::fabs(p.rho) < 1.0;
}

/// (B) 翼のバンド η (1 + |ρ|) ≤ 2。Theorem 4.2（バタフライ裁定がないための十分条件
///     θφ(θ)(1 + |ρ|) < 4 かつ θφ(θ)²(1 + |ρ|) ≤ 4）を Remark 4.4 の Heston 型
///     φ(θ) = η / (θ^γ (1 + θ)^{1−γ}) に当てはめたときの有名な境界で、ここでは翼が立ちすぎた面を
///     UI から締め出すための保守的な制約として使う（下の「過大な主張を避ける注記」を参照）。
inline bool ssvi_wing_band_ok(const SsviParams& p) noexcept {
    if (!(p.eta > 0.0)) return false;  // NaN / 非正の η をここでも弾く
    return p.eta * (1.0 + std::fabs(p.rho)) <= kSsviMaxEtaBand;
}

}  // namespace detail

/// 総分散 w(k, T) = iv(k, T)² T。k は log-moneyness、T は満期までの年数。
/// パラメータの正規化はしない（ssvi_clamp を通してから渡す）。T ≤ 0 でも有限値を返す。
inline double ssvi_total_variance(double k, double T, SsviParams p) noexcept {
    const double theta = detail::ssvi_theta(T, p);
    const double x     = detail::ssvi_phi(theta, p) * k;  // φ(θ) k
    const double xr    = x + p.rho;
    const double disc  = std::sqrt(xr * xr + (1.0 - p.rho * p.rho));
    return 0.5 * theta * (1.0 + p.rho * x + disc);
}

/// インプライド・ボラ sqrt(w(k, T) / T)。T は kSsviMinT で床を入れる（T ≤ 0 でも有限・正）。
inline double ssvi_implied_vol(double k, double T, SsviParams p) noexcept {
    const double t = T > kSsviMinT ? T : kSsviMinT;  // NaN もここで床へ
    return std::sqrt(ssvi_total_variance(k, t, p) / t);
}

/// パラメータが「この面を使ってよい」領域にあるか = (A) カレンダー条件（detail::ssvi_calendar_ok）
/// かつ (B) 翼のバンド（detail::ssvi_wing_band_ok）。各条件の出典と導出は各 detail 関数の説明にある。
///
/// 名前と中身のずれについて: ssvi_clamp を通したパラメータは定義より (A) を必ず満たすので、
/// クランプ後にこの関数が false を返す理由は (B) だけである（つまり UI から見れば実質的に
/// 「翼のバンドに入っているか」を聞いている）。クランプ前の生の入力に対しては (A) も発火する。
///
/// 過大な主張を避けるための注記: 本ファイルの φ は pure power-law なので、γ < 1 では
/// θ → ∞ で θ φ(θ) = η θ^{1−γ} → ∞ となり Theorem 4.2 の条件 (i) は必ずどこかで破れる。
/// したがって本関数の true は「全 θ でバタフライ裁定がない」ことを保証しない。保証するのは
/// (A)（カレンダー裁定がないこと、必要十分）だけで、(B) は形を実用域に保つための追加制約である。
/// 逆に false も「カレンダー裁定がある」ことを意味しない（(B) だけで落ちうる）。
inline bool ssvi_calendar_arbitrage_free(SsviParams p) noexcept {
    return detail::ssvi_calendar_ok(p) && detail::ssvi_wing_band_ok(p);
}

/// UI からの入力を許容域へ丸める。σ_atm ∈ [1e-6, 5]、ρ ∈ [−0.999, 0.999]、η ∈ [1e-6, 100]、
/// γ ∈ [0, 1]。NaN はそのフィールドの既定値に戻す。±inf は有限の端点へ。冪等。
/// これを通した値なら ssvi_total_variance / ssvi_implied_vol は全ての有限 (k, T) で有限かつ正。
constexpr SsviParams ssvi_clamp(SsviParams p) noexcept {
    constexpr SsviParams d{};

    p.sigma_atm = detail::ssvi_clamp_field(p.sigma_atm, kSsviMinSigmaAtm, kSsviMaxSigmaAtm, d.sigma_atm);
    p.rho       = detail::ssvi_clamp_field(p.rho, -kSsviMaxAbsRho, kSsviMaxAbsRho, d.rho);
    p.eta       = detail::ssvi_clamp_field(p.eta, kSsviMinEta, kSsviMaxEta, d.eta);
    p.gamma     = detail::ssvi_clamp_field(p.gamma, kSsviMinGamma, kSsviMaxGamma, d.gamma);
    return p;
}

}  // namespace quantviz::core
