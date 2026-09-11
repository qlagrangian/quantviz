#pragma once
// core/stats/garch.hpp — GARCH(1,1) の分散フィルタ・対数尤度・最尤推定・合成パス生成。
//
//   σ²_t = ω + α r²_{t−1} + β σ²_{t−1},   r_t = σ_t z_t,  z_t ~ N(0,1)
//
// 定常条件は ω > 0, α ≥ 0, β ≥ 0, α+β < 1 で、無条件分散は V = ω/(1−α−β)。
// フィルタの初期値 σ²_0 は V（非定常なら標本二乗平均）。対数尤度は
//   LL = −½ Σ_t ( log 2π + log σ²_t + r²_t / σ²_t )
// で、非定常パラメータでは −∞ を返す（尤度格子を描く側はこれを最小値にクランプする）。
//
// 最尤推定は無制約 θ ∈ R³ 上で行う:
//   ω = softplus(θ0),  s = kGarchMaxPersistence · sigmoid(θ1),
//   α = s · sigmoid(θ2),  β = s · sigmoid(−θ2)
// 倍精度の sigmoid は θ ≳ 37 で厳密に 1 になるので、係数 kGarchMaxPersistence < 1 を掛けて
// α+β = 1 を変換の像から外す（GARCH-04）。θ は ±kGarchThetaMax にクランプするので ±∞ や NaN が
// 混ざっても定常な点に落ちる。逆変換は θ2 = log α − log β なので α/β の割合は両裾とも厳密に往復する。
// θ1 は α+β を倍精度で保持する以上 1 − (α+β)/kGarchMaxPersistence の相殺で分解能が落ち、
// 相対 1e-12 の往復は |θ1| ≲ 10 まで。
//
// ホットパス（garch_filter / garch_log_likelihood / garch_simulate）はヒープを使わない。
// garch_fit_into は呼び出し側の GarchFit を再利用し、2 回目以降はヒープ確保しない（下記）。

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

#include "quantviz/core/rng.hpp"
#include "quantviz/core/stats/optim.hpp"

namespace quantviz::core {

struct GarchParams {
    double omega;
    double alpha;
    double beta;
};

/// 変換で到達できる α+β の上限。1 との差 1e-8 は倍精度の丸め（~1e-16）より十分大きい。
inline constexpr double kGarchMaxPersistence = 1.0 - 1e-8;
/// 無制約パラメータのクランプ幅。exp(±700) が倍精度で有限に収まる範囲。
inline constexpr double kGarchThetaMax = 700.0;
/// log(2π)。std::log は constexpr でないので数値リテラルで持つ。
inline constexpr double kLog2Pi = 1.8378770664093454835606594728112;

inline bool garch_stationary(GarchParams p) noexcept {
    return p.omega > 0.0 && p.alpha >= 0.0 && p.beta >= 0.0 && p.alpha + p.beta < 1.0;
}

/// 無条件分散 ω/(1−α−β)。非定常なら +∞（存在しない）。
inline double garch_unconditional_variance(GarchParams p) noexcept {
    return garch_stationary(p) ? p.omega / (1.0 - p.alpha - p.beta) : std::numeric_limits<double>::infinity();
}

/// ショックの半減期 ln(0.5)/ln(α+β)。α+β ≥ 1 なら +∞、α+β ≤ 0 なら 0。
/// 変換の上限 kGarchMaxPersistence では ≈ 6.9e7（日）になるので、表示側は適当な上限でクランプすること。
inline double garch_half_life(GarchParams p) noexcept {
    const double s = p.alpha + p.beta;
    if (!(s > 0.0)) return 0.0;
    if (s >= 1.0) return std::numeric_limits<double>::infinity();
    return std::log(0.5) / std::log(s);
}

namespace detail {

inline double garch_mean_square(std::span<const double> r) noexcept {
    if (r.empty()) return 0.0;
    double s = 0.0;
    for (const double x : r) s += x * x;
    return s / static_cast<double>(r.size());
}

/// フィルタの初期分散: 定常なら無条件分散、そうでなければ標本二乗平均（空なら ω）。
inline double garch_initial_variance(GarchParams p, std::span<const double> r) noexcept {
    if (garch_stationary(p)) return p.omega / (1.0 - p.alpha - p.beta);
    return r.empty() ? p.omega : garch_mean_square(r);
}

/// NaN を含めて [−kGarchThetaMax, kGarchThetaMax] に入れる（NaN は比較が偽なので下限に落ちる）。
inline double garch_clamp_theta(double th) noexcept {
    if (!(th > -kGarchThetaMax)) return -kGarchThetaMax;
    if (th > kGarchThetaMax) return kGarchThetaMax;
    return th;
}

}  // namespace detail

/// 各 t の σ²_t を sigma2_out に書く（min(r.size(), sigma2_out.size()) 要素）。
/// sigma2_out[0] = σ²_0（初期分散）、sigma2_out[t] = ω + α r²_{t−1} + β σ²_{t−1}。
inline void garch_filter(GarchParams p, std::span<const double> r, std::span<double> sigma2_out) noexcept {
    const std::size_t n = std::min(r.size(), sigma2_out.size());
    if (n == 0) return;
    double s2     = detail::garch_initial_variance(p, r);
    sigma2_out[0] = s2;
    for (std::size_t t = 1; t < n; ++t) {
        s2            = p.omega + p.alpha * r[t - 1] * r[t - 1] + p.beta * s2;
        sigma2_out[t] = s2;
    }
}

/// 対数尤度 −½ Σ_t (log 2π + log σ²_t + r²_t/σ²_t)。非定常なら −∞。空なら 0。
inline double garch_log_likelihood(GarchParams p, std::span<const double> r) noexcept {
    if (!garch_stationary(p)) return -std::numeric_limits<double>::infinity();
    if (r.empty()) return 0.0;
    double s2   = p.omega / (1.0 - p.alpha - p.beta);
    double prev = r[0];
    double sum  = std::log(s2) + prev * prev / s2;
    for (std::size_t t = 1; t < r.size(); ++t) {
        s2              = p.omega + p.alpha * prev * prev + p.beta * s2;
        const double rt = r[t];
        sum += std::log(s2) + rt * rt / s2;
        prev = rt;
    }
    return -0.5 * (static_cast<double>(r.size()) * kLog2Pi + sum);
}

/// 無制約 θ ∈ R³ → 定常な (ω, α, β)。像は必ず定常領域に入る（±∞・NaN を含む）。
inline GarchParams garch_from_unconstrained(std::array<double, 3> th) noexcept {
    const double t0 = detail::garch_clamp_theta(th[0]);
    const double t1 = detail::garch_clamp_theta(th[1]);
    const double t2 = detail::garch_clamp_theta(th[2]);
    const double s  = kGarchMaxPersistence * to_unit(t1);  // α+β
    // α, β とも sigmoid で直接計算する（s·(1−w) だと β の裾で相殺誤差が出る）
    return GarchParams{to_positive(t0), s * to_unit(t2), s * to_unit(-t2)};
}

/// (ω, α, β) → 無制約 θ。定常領域の外（UI からの不正値など）は像の境界へ丸め、常に有限な θ を返す。
inline std::array<double, 3> garch_to_unconstrained(GarchParams p) noexcept {
    const double s = p.alpha + p.beta;
    double       u = s / kGarchMaxPersistence;  // ∈ (0,1) なら逆変換できる
    if (!(u < 1.0)) u = 1.0;                    // α+β ≥ kMax（NaN 含む）→ 上端
    if (!(u > 0.0)) u = 0.0;                    // α+β ≤ 0 → 下端
    // θ2 = logit(α/(α+β)) = log α − log β。両裾で厳密。片方が 0 なら ±∞（後でクランプ）、両方 0 なら中央。
    double t2 = 0.0;
    if (p.alpha > 0.0 && p.beta > 0.0) {
        t2 = std::log(p.alpha) - std::log(p.beta);
    } else if (p.alpha > 0.0) {
        t2 = std::numeric_limits<double>::infinity();
    } else if (p.beta > 0.0) {
        t2 = -std::numeric_limits<double>::infinity();
    }
    return {detail::garch_clamp_theta(from_positive(p.omega)), detail::garch_clamp_theta(from_unit(u)),
            detail::garch_clamp_theta(t2)};
}

/// 最尤推定の結果。`scratch` は再試行用の作業領域で、garch_fit_into が capacity を再利用するために
/// 保持している。利用側が読む必要はない。
struct GarchFit {
    GarchParams    params{};
    double         log_lik = 0.0;
    OptimResult<3> optim{};
    OptimResult<3> scratch{};
};

namespace detail {

/// 初期値の内部射影に使う |θ1|, |θ2| の上限。sigmoid(±6) ≈ 0.0025 / 0.9975 で、導関数がまだ 2.5e-3 残る。
inline constexpr double kGarchStartThetaMax = 6.0;
/// 再試行の反復上限（初期値からの当てはめより短くする）。
inline constexpr std::size_t kGarchRetryMaxIter = 200;
/// 再試行の参照点 (α, β)。ω は無条件分散が標本二乗平均に一致するように取る。
inline constexpr std::array<std::array<double, 2>, 3> kGarchReferenceStarts{
    {{0.05, 0.90}, {0.10, 0.80}, {0.02, 0.95}}};

/// 初期値を尤度面の「動ける」内部へ射影した θ を返す。
/// * β = 0 や α+β = 0 は θ2, θ1 = ±700 に写り、そこでは sigmoid' が倍精度で厳密に 0 なので目的関数が
///   完全に平坦になり、単体法も数値勾配も動けない。|θ1|, |θ2| ≤ kGarchStartThetaMax に寄せる。
/// * ω は [1e-6 v, v]（v = 標本二乗平均）に入れる。最適解では ω = V(1−α−β) ≤ V なので ω > v は無意味で、
///   桁違いに大きい ω から始めると単体法が平坦な方向で偽収束する。ω ≪ v では σ² が ω に依存せず平坦。
inline std::array<double, 3> garch_start_theta(GarchParams init, double v) noexcept {
    double omega = init.omega;
    if (!(omega >= 1e-6 * v)) omega = 1e-6 * v;  // NaN もここで直る
    if (omega > v) omega = v;
    std::array<double, 3> th = garch_to_unconstrained(GarchParams{omega, init.alpha, init.beta});
    th[1]                    = std::clamp(th[1], -kGarchStartThetaMax, kGarchStartThetaMax);
    th[2]                    = std::clamp(th[2], -kGarchStartThetaMax, kGarchStartThetaMax);
    return th;
}

}  // namespace detail

/// 最尤推定（結果を out に書き込む版）。init（前回の推定値など）を初期値に、無制約 θ 上で −LL を最小化する。
///
/// コストと推奨設定:
/// * 対数尤度 1 回の評価は窓長 n に比例（n=500 で ≈ 3 µs、Release）。NM は 1 反復 ≈ 1.5 評価、BFGS は
///   1 反復 ≈ 7 評価（勾配 6 + 直線探索 1〜2）。健全な当てはめは NM ≈ 100–120 反復 / BFGS ≈ 21 反復で
///   収束するので、前回の推定値からの再当てはめでは `OptimOptions{.max_iter = 200}` を推奨する。
///   実測（n=500、10 ステップごとに再当てはめ、前回値から開始）: α+β=0.98 のデータで平均 ≈ 0.35 ms、
///   最大 ≈ 1.2 ms（NM）/ 3.6 ms（BFGS）。iid に近いデータでは平均 ≈ 0.6 ms、最大 ≈ 1.6 / 3.3 ms。
/// * 同じ GarchFit を使い回すと path の capacity が再利用され、2 回目以降はヒープ確保が起きない
///   （再試行が走った場合も scratch が同様に再利用される）。
///
/// 局所解対策（根拠: N=20000 の合成パスで 8×8×4 通りの初期値から当てはめた調査）:
/// * 内部の初期値からは NM / BFGS とも全て真の最尤点に達した。失敗は全て β = 0 や α+β = 0 のように
///   変換の像の端（sigmoid の飽和で勾配が厳密に 0 の平坦部）に写る初期値と、ω が標本分散の桁違いに
///   大きい初期値からで、後者は単体法が平坦な方向で偽収束する（再スタートや BFGS なら抜ける）。
///   罠に落ちた結果は、真の最尤点より対数尤度が数百 nat 低い。
/// * そこで初期値は detail::garch_start_theta で内部へ射影してから始める。さらに固定の参照点
///   kGarchReferenceStarts（標本二乗平均に無条件分散を合わせた 3 点）の対数尤度を評価し（3 回の評価、
///   n=500 で ≈ 30 µs）、その最良点が当てはめ結果より高い尤度を持つときだけ、そこから
///   max_iter ≤ kGarchRetryMaxIter で再当てはめして良い方を採る。参照点は固定なので結果は決定的。
/// * 再試行の条件を「未収束」や「境界に張り付いた」で決めないのは、iid に近い窓では尤度面が β 方向に
///   平坦で、未収束・α → 0・β → 0 のどれも正当な（尤度がほぼ最大の）終点になり、再試行が費用を 4 倍に
///   するだけで何も改善しないため。参照点との尤度比較なら、そのような窓では発火せず、罠は必ず捕まえる。
///
/// 当てはめが定義できないデータ（空、全て 0、NaN 混入 = 標本二乗平均が正でない）では最適化せず、
/// params = init、log_lik = init での対数尤度（空なら 0、NaN 混入なら NaN）、
/// optim = {x = θ(init), iters = 0, converged = false, path = {x}} を返す。
inline void garch_fit_into(GarchFit& out, std::span<const double> r, GarchParams init, OptimOptions o = {},
                           bool use_bfgs = false) {
    const double v = detail::garch_mean_square(r);
    if (!(v > 0.0)) {  // 空・全て 0・NaN 混入
        out.params          = init;
        out.log_lik         = garch_log_likelihood(init, r);
        out.optim.x         = garch_to_unconstrained(init);
        out.optim.f         = -out.log_lik;
        out.optim.iters     = 0;
        out.optim.converged = false;
        out.optim.path.clear();
        out.optim.path.push_back(out.optim.x);
        return;
    }

    const auto objective = [r](const std::array<double, 3>& th) noexcept {
        return -garch_log_likelihood(garch_from_unconstrained(th), r);
    };
    const auto run_into = [&](OptimResult<3>& res, GarchParams start, const OptimOptions& opts) {
        const std::array<double, 3> x0 = detail::garch_start_theta(start, v);
        if (use_bfgs) {
            bfgs_into(res, objective, x0, opts);
        } else {
            nelder_mead_into(res, objective, x0, opts);
        }
    };

    run_into(out.optim, init, o);

    // 再試行判定: 固定の参照点のうち最良のものが当てはめ結果より高い尤度なら、そこから再当てはめする
    GarchParams best_ref{};
    double      best_ref_f = std::numeric_limits<double>::infinity();  // −LL
    for (const auto& ab : detail::kGarchReferenceStarts) {
        const GarchParams ref{v * (1.0 - ab[0] - ab[1]), ab[0], ab[1]};
        const double      f = -garch_log_likelihood(ref, r);
        if (f < best_ref_f) {
            best_ref_f = f;
            best_ref   = ref;
        }
    }
    if (best_ref_f < out.optim.f) {
        OptimOptions retry = o;
        retry.max_iter     = std::min(o.max_iter, detail::kGarchRetryMaxIter);
        run_into(out.scratch, best_ref, retry);
        if (out.scratch.f < out.optim.f) std::swap(out.optim, out.scratch);  // O(1)、確保なし
    }
    out.params  = garch_from_unconstrained(out.optim.x);
    out.log_lik = -out.optim.f;
}

/// 最尤推定（値を返す版）。garch_fit_into の薄いラッパ。
inline GarchFit garch_fit(std::span<const double> r, GarchParams init, OptimOptions o = {},
                          bool use_bfgs = false) {
    GarchFit out;
    garch_fit_into(out, r, init, o, use_bfgs);
    return out;
}

/// 合成 GARCH パス。σ²_0 = 無条件分散（非定常なら ω）。乱数は core::Rng(seed) なので決定的。
/// r_out.size() ステップ生成し、sigma2_out には収まる分だけ σ²_t を書く。
inline void garch_simulate(GarchParams p, std::uint64_t seed, std::span<double> r_out,
                           std::span<double> sigma2_out) noexcept {
    Rng          rng(seed);
    const double v0 = garch_stationary(p) ? p.omega / (1.0 - p.alpha - p.beta) : std::max(p.omega, 0.0);
    double       s2 = v0;
    for (std::size_t t = 0; t < r_out.size(); ++t) {
        if (!(s2 >= 0.0)) s2 = 0.0;  // 非定常・不正パラメータでも sqrt が NaN にならないように
        const double rt = std::sqrt(s2) * rng.normal();
        r_out[t]        = rt;
        if (t < sigma2_out.size()) sigma2_out[t] = s2;
        s2 = p.omega + p.alpha * rt * rt + p.beta * s2;
    }
}

}  // namespace quantviz::core
