#pragma once
// core/models/hawkes.hpp — 指数カーネルの自己励起過程（Hawkes 過程）。強度・生成・対数尤度・最尤推定。
//
//   λ(t) = μ + Σ_{t_i < t} α e^{−β (t − t_i)}
//
// 事象が起きるたびに強度が α だけ跳ね、時定数 1/β で μ へ戻る。分岐比 η = α/β は 1 事象が直接
// 生む子の期待数で、η < 1（安定条件）のとき定常強度は μ/(1−η)。μ > 0, α ≥ 0, β > 0, η < 1 を
// 「安定」と呼ぶ（hawkes_stable）。t = 0 で履歴が空から始めると立ち上がりの分だけ強度が低く、
// [0,T) の事象数の期待値は μT/(1−η) ではなく
//   E[N(T)] = μ/(1−η)·[T − η(1 − e^{−β(1−η)T})/(β(1−η))]
// になる。定常近似との差は μη(1 − e^{−β(1−η)T})/(β(1−η)²) 件で、T → ∞ で μη/(β(1−η)²) に漸近する
// （減衰の時定数 1/(β(1−η)) より長い窓ならほぼ上限。HAWKES-03 の許容幅の注記を参照）。
//
// λ は t_i < t の和（左連続・可予測）で定義する。事象時刻ちょうどでは自分の寄与を含めない。
// 逐次版 HawkesIntensity::at(t) は逆に「t までの全事象」を含む λ(t⁺) を返す:
// thinning の上界も LOB シーンの表示もジャンプ後の値が要るため。両者は t が事象時刻でなければ一致する。
//
// 計算量とホットパス:
// * HawkesIntensity は 1 事象あたり exp 1 回の O(1) 更新（A ← e^{−βΔt} A + α。事象を含めない
//   表現で書けば A⁻ ← e^{−βΔt}(A⁻ + α) と同じ漸化式）。LOB シーンは 1 ms ステップでこれを回す。
// * hawkes_simulate / hawkes_log_likelihood / hawkes_compensator / hawkes_rescaled_residuals は
//   呼び手のバッファにだけ書き、ヒープを使わず例外も投げない。
// * hawkes_fit / hawkes_fit_into だけが OptimResult の path（std::vector）を通じてヒープに触れる。
//   当てはめは O(n) の尤度評価を数百回回すオフライン診断で、LOB シーンの step() からは決して呼ばない
//   （ホットパスで使ってよいのは HawkesIntensity の O(1) 更新だけ）。計算スレッドで周期的に
//   当てはめたい場合は同じ HawkesFit を hawkes_fit_into に渡せば 2 回目以降の確保は起きない。
//
// 事前条件（Debug の assert で検査し、NDEBUG では消える。クランプはしない: 呼び手のバグを隠さない）:
// * times は昇順（hawkes_log_likelihood / hawkes_rescaled_residuals）。
// * 事象は窓の中: times.back() < T（対数尤度。窓の外の事象は Σ log λ に入るのに補償子からは落ちるので
//   尤度が非対称になり、当てはめが「正の対数尤度」に収束してしまう）。補償子は Λ(T) を事象時刻ちょうどで
//   評価する使い方（残差のテレスコープ）があるので times.back() ≤ T まで許す。
// * HawkesIntensity::add_event の時刻は非負・単調非減少。

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>

#include "quantviz/core/rng.hpp"
#include "quantviz/core/stats/optim.hpp"

namespace quantviz::core {

struct HawkesParams {
    double mu;     ///< 基底強度 μ > 0
    double alpha;  ///< 跳ね幅 α ≥ 0
    double beta;   ///< 減衰率 β > 0（時定数 1/β）
};

/// 変換で到達できる分岐比 α/β の上限。1 との差 1e-8 は倍精度の丸め（~1e-16）より十分大きい。
inline constexpr double kHawkesMaxBranching = 1.0 - 1e-8;
/// 無制約パラメータのクランプ幅。exp(±700) が倍精度で有限に収まる範囲。
inline constexpr double kHawkesThetaMax = 700.0;

/// 安定（定常）条件: μ > 0, α ≥ 0, β > 0, α/β < 1。NaN / ±inf は全て false。
inline bool hawkes_stable(HawkesParams p) noexcept {
    return p.mu > 0.0 && p.alpha >= 0.0 && p.beta > 0.0 && p.alpha / p.beta < 1.0;
}

/// λ(t) = μ + Σ_{t_i < t} α e^{−β(t−t_i)} の直接和（O(n)、テストの参照実装）。times の並びは問わない。
inline double hawkes_intensity(HawkesParams p, std::span<const double> times, double t) noexcept {
    double s = 0.0;
    for (const double ti : times) {
        if (ti < t) s += std::exp(-p.beta * (t - ti));
    }
    return p.mu + p.alpha * s;
}

/// 逐次強度。事象を 1 つ足すたびに O(1)（exp 1 回）で更新する。
/// at(t) は「これまでに add_event した全事象」を含む λ(t⁺)。過去（t < 直近事象時刻）を問い合わせた
/// ときは直近事象時刻での値を返す（時間を巻き戻さない。巻き戻しは reset）。
class HawkesIntensity {
public:
    explicit HawkesIntensity(HawkesParams p) noexcept : p_(p) {}

    /// 時刻 t に事象を 1 つ加える（t は直前の事象以降であること。Debug では assert で検査する）。
    void add_event(double t) noexcept {
        assert(t >= last_);
        excite_ = excitation_at(t) + p_.alpha;
        last_   = t;
    }

    /// λ(t⁺) = μ + Σ_{t_i ≤ t} α e^{−β(t−t_i)}
    double at(double t) const noexcept { return p_.mu + excitation_at(t); }

    /// 事象履歴を捨てて基底強度に戻す。
    void reset() noexcept {
        excite_ = 0.0;
        last_   = 0.0;
    }

private:
    double excitation_at(double t) const noexcept {
        if (excite_ == 0.0) return 0.0;  // 事象がまだ無い（exp を呼ばない近道）
        const double dt = t - last_;
        return dt > 0.0 ? excite_ * std::exp(-p_.beta * dt) : excite_;
    }

    HawkesParams p_;
    double       excite_ = 0.0;  ///< 直近事象時刻 last_ における励起項 Σ α e^{−β(last_−t_i)}
    double       last_   = 0.0;  ///< 直近事象時刻
};

/// Ogata の thinning で [0, T) の事象時刻を昇順に out へ書き、件数を返す。
///
/// 事象の間で λ は非増加なので、現在時刻の λ(t⁺) がそれ以降の上界 λ* になる。Exp(λ*) の間隔で候補
/// 時刻へ進み、確率 λ(候補)/λ* で採択する（棄却しても、その候補時刻の λ が新しい（より小さい）上界に
/// なるので効率が落ちない）。1 候補あたり一様乱数 2 個を消費するので、同じ seed・同じパラメータなら
/// 系列はビット一致する。
///
/// out が満ちたらそこで打ち切り out.size() を返す。安定でないパラメータ（α/β ≥ 1 など。爆発しうる）
/// や T ≤ 0、空の out では何も書かずに 0 を返す。ヒープ確保なし・例外なし。
inline std::size_t hawkes_simulate(HawkesParams p, double T, Rng& rng, std::span<double> out) noexcept {
    if (!hawkes_stable(p) || !(T > 0.0) || out.empty()) return 0;
    HawkesIntensity intensity(p);
    std::size_t     n = 0;
    double          t = 0.0;
    while (true) {
        const double bound = intensity.at(t);  // λ(t⁺) = 以後の上界
        if (!(bound > 0.0)) break;             // μ > 0 なので起きないが、無限ループを作らない保険
        // U ∈ [0,1) から 1−U ∈ (0,1] を作り、−log(1−U)/λ* を指数間隔にする（log(0) = −∞ を避ける）
        t += -std::log1p(-rng.uniform()) / bound;
        if (!(t < T)) break;
        if (rng.uniform() * bound <= intensity.at(t)) {  // 採択確率 λ(t)/λ*
            out[n] = t;
            ++n;
            intensity.add_event(t);
            if (n == out.size()) break;
        }
    }
    return n;
}

// ---- 推定（対数尤度・補償子・残差） -----------------------------------------------------------
//
// 以下は times が昇順で [0, T) に入っていることを前提にする（hawkes_simulate の出力がそれ）。

/// 補償子 Λ(T) = ∫_0^T λ(s) ds = μT + (α/β) Σ_{t_i < T} (1 − e^{−β(T−t_i)})。O(n)。
/// 1 − e^{−x} は −expm1(−x) で計算する（x が小さいときの相殺を避ける）。
/// t_i ≥ T の事象は落とす（Λ の定義どおり）。T を最後の事象時刻に取る使い方（残差のテレスコープ）は
/// 正当なので、Debug の assert は times.back() ≤ T まで許す。
inline double hawkes_compensator(HawkesParams p, std::span<const double> times, double T) noexcept {
    assert(times.empty() || times.back() <= T);
    double s = 0.0;
    for (const double ti : times) {
        if (ti < T) s += -std::expm1(-p.beta * (T - ti));
    }
    return p.mu * T + (p.alpha / p.beta) * s;
}

/// 対数尤度 Σ_i log λ(t_i) − Λ(T)。
/// 励起項は O(n) の漸化式 R_i = e^{−β(t_i − t_{i−1})}(1 + R_{i−1})（R_0 = 0、λ(t_i) = μ + α R_i）で、
/// 補償子は閉形式。安定でないパラメータ（μ ≤ 0 / β ≤ 0 / α/β ≥ 1）では −∞ を返す
/// （尤度面を描く側はこれを最小値にクランプする。garch_log_likelihood と同じ約束）。
inline double hawkes_log_likelihood(HawkesParams p, std::span<const double> times, double T) noexcept {
    assert(std::is_sorted(times.begin(), times.end()));
    assert(times.empty() || times.back() < T);  // 窓の外の事象があると尤度が非対称になる
    if (!hawkes_stable(p)) return -std::numeric_limits<double>::infinity();
    double sum  = 0.0;
    double r    = 0.0;  // R_i
    double prev = 0.0;
    bool   first = true;
    for (const double ti : times) {
        if (!first) r = std::exp(-p.beta * (ti - prev)) * (1.0 + r);
        sum += std::log(p.mu + p.alpha * r);
        prev  = ti;
        first = false;
    }
    return sum - hawkes_compensator(p, times, T);
}

/// 時間再スケール残差 τ_i = Λ(t_i) − Λ(t_{i−1})（t_{−1} = 0）を out に書き、件数を返す。
/// 真のパラメータなら τ は iid 単位指数分布（random time change）になるので、適合度診断に使う。
/// 同じ漸化式から O(n) で出る:
///   τ_0 = μ t_0、 τ_i = μ(t_i − t_{i−1}) + (α/β)(1 − e^{−βΔ_i})(1 + R_{i−1})、
///   R_i = e^{−βΔ_i}(1 + R_{i−1})
/// out が短ければそこで打ち切って out.size() を返す。安定でないパラメータでは残差が意味を持たない
/// （τ が負や NaN になりうる）ので、何も書かずに 0 を返す。
/// 事前条件: times は昇順（Debug の assert で検査する）。
inline std::size_t hawkes_rescaled_residuals(HawkesParams p, std::span<const double> times,
                                             std::span<double> out) noexcept {
    assert(std::is_sorted(times.begin(), times.end()));
    if (!hawkes_stable(p)) return 0;
    const std::size_t n     = std::min(times.size(), out.size());
    const double      ratio = p.alpha / p.beta;
    double            r     = 0.0;  // R_{i−1}
    double            prev  = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dt = times[i] - prev;
        if (i == 0) {
            out[i] = p.mu * dt;  // 先行事象が無いので励起項は 0
        } else {
            const double decay = std::exp(-p.beta * dt);
            const double grow  = -std::expm1(-p.beta * dt);  // 1 − e^{−βΔ}（Δ が小さくても相対誤差なし）
            out[i]             = p.mu * dt + ratio * grow * (1.0 + r);
            r                  = decay * (1.0 + r);
        }
        prev = times[i];
    }
    return n;
}

namespace detail {

/// NaN を含めて [−kHawkesThetaMax, kHawkesThetaMax] に入れる（NaN は比較が偽なので下限に落ちる）。
inline double hawkes_clamp_theta(double th) noexcept {
    if (!(th > -kHawkesThetaMax)) return -kHawkesThetaMax;
    if (th > kHawkesThetaMax) return kHawkesThetaMax;
    return th;
}

}  // namespace detail

/// 無制約 θ ∈ R³ → 安定な (μ, α, β): μ = softplus(θ0), β = softplus(θ1), α = β·kMaxBranching·sigmoid(θ2)。
/// 分岐比を sigmoid で作るので α/β < 1 は構造的に破れない（±∞・NaN を入れても安定な点に落ちる）。
inline HawkesParams hawkes_from_unconstrained(std::array<double, 3> th) noexcept {
    const double mu   = to_positive(detail::hawkes_clamp_theta(th[0]));
    const double beta = to_positive(detail::hawkes_clamp_theta(th[1]));
    const double eta  = kHawkesMaxBranching * to_unit(detail::hawkes_clamp_theta(th[2]));
    return HawkesParams{mu, beta * eta, beta};
}

/// (μ, α, β) → 無制約 θ。安定領域の外（UI からの不正値など）は像の境界へ丸め、常に有限な θ を返す。
inline std::array<double, 3> hawkes_to_unconstrained(HawkesParams p) noexcept {
    double u = p.alpha / (p.beta * kHawkesMaxBranching);  // ∈ (0,1) なら逆変換できる
    if (!(u < 1.0)) u = 1.0;                              // α ≥ β·kMax（NaN 含む）→ 上端
    if (!(u > 0.0)) u = 0.0;                              // α ≤ 0 / β ≤ 0 → 下端
    return {detail::hawkes_clamp_theta(from_positive(p.mu)),
            detail::hawkes_clamp_theta(from_positive(p.beta)), detail::hawkes_clamp_theta(from_unit(u))};
}


// ---- 最尤推定 ---------------------------------------------------------------------------------

/// 最尤推定の結果。optim.x は無制約 θ で、params = hawkes_from_unconstrained(optim.x)、
/// log_lik = −optim.f。optim.path は θ 空間の軌跡（尤度面の描画用）。
/// `scratch` は再試行用の作業領域で、hawkes_fit_into が capacity を再利用するために保持している。
/// 利用側が読む必要はない。
struct HawkesFit {
    HawkesParams   params{};
    double         log_lik = 0.0;
    OptimResult<3> optim{};
    OptimResult<3> scratch{};
};

namespace detail {

/// 初期値の内部射影に使う |θ2| の上限。sigmoid(±6) ≈ 0.0025 / 0.9975 で導関数がまだ 2.5e-3 残る。
inline constexpr double kHawkesStartThetaMax = 6.0;
/// 再試行の反復上限（初期値からの当てはめより短くする）。
inline constexpr std::size_t kHawkesRetryMaxIter = 200;
/// 再試行の参照点 {分岐比 η, β/観測レート}。μ は観測レート n/T = μ/(1−η) に合うように取る。
inline constexpr std::array<std::array<double, 2>, 3> kHawkesReferenceStarts{
    {{0.3, 1.0}, {0.6, 2.0}, {0.8, 5.0}}};

/// 初期値を尤度面の「動ける」内部へ射影した θ を返す。rate = n/T は観測された事象レート。
/// * μ ∈ [1e-4·rate, rate]: 定常なら μ = rate·(1−η) ≤ rate なので rate より大きい μ は意味がなく、
///   rate から桁違いに小さい μ では log λ が μ に依存しなくなって面が平坦になる。
/// * β ∈ [0.01·rate, 100·rate]: 時定数 1/β が事象間隔 1/rate と桁違いだと、励起が次の事象までに
///   消える（β 大）か窓の中で減衰しない（β 小）かで尤度が β にほぼ依存しなくなる。
/// * |θ2| ≤ kHawkesStartThetaMax: 分岐比が 0 や 1 に張り付くと sigmoid が飽和して勾配が厳密に 0 になる。
inline std::array<double, 3> hawkes_start_theta(HawkesParams init, double rate) noexcept {
    double mu = init.mu;
    if (!(mu >= 1e-4 * rate)) mu = 1e-4 * rate;  // NaN もここで直る
    if (mu > rate) mu = rate;
    double beta = init.beta;
    if (!(beta >= 0.01 * rate)) beta = 0.01 * rate;
    if (beta > 100.0 * rate) beta = 100.0 * rate;
    std::array<double, 3> th = hawkes_to_unconstrained(HawkesParams{mu, init.alpha, beta});
    th[2]                    = std::clamp(th[2], -kHawkesStartThetaMax, kHawkesStartThetaMax);
    return th;
}

}  // namespace detail

/// 最尤推定（結果を out に書き込む版）。init（前回の推定値など）を初期値に、無制約 θ 上で −LL を最小化する。
/// θ ↦ (μ = softplus(θ0), β = softplus(θ1), α = β·kMaxBranching·sigmoid(θ2)) なので、
/// 探索中も結果も必ず α/β < 1（安定領域）に入る。既定は Nelder–Mead、use_bfgs = true で BFGS。
///
/// 局所解対策は garch_fit と同じ形: 初期値を detail::hawkes_start_theta で内部へ射影してから始め、
/// さらに観測レートに合わせた固定の参照点 kHawkesReferenceStarts（3 点、評価は O(n) 3 回）の尤度を
/// 見て、その最良点が当てはめ結果より高い尤度なら、そこから max_iter ≤ kHawkesRetryMaxIter で
/// 再当てはめして良い方を採る。参照点は固定なので結果は決定的。
///
/// 同じ HawkesFit を使い回すと path の capacity が再利用され、2 回目以降はヒープ確保が起きない
/// （再試行が走った場合も scratch が同様に再利用される）。ただし当てはめ自体は O(n) × 数百評価の
/// オフライン診断で、LOB シーンの step() から呼んではいけない（ホットパスは HawkesIntensity だけ）。
///
/// 当てはめが定義できないデータ（事象なし、T ≤ 0）では最適化せず、params = init、
/// log_lik = init での対数尤度（T ≤ 0 では定義できないので NaN）、
/// optim = {x = θ(init), iters = 0, converged = false, path = {x}} を返す。
inline void hawkes_fit_into(HawkesFit& out, std::span<const double> times, double T, HawkesParams init,
                            OptimOptions o = {}, bool use_bfgs = false) {
    if (times.empty() || !(T > 0.0)) {
        out.params  = init;
        // 窓が無い（T ≤ 0）と対数尤度は定義できないので NaN。事象が無いだけなら LL = −μT は定義できる。
        out.log_lik         = (T > 0.0) ? hawkes_log_likelihood(init, times, T)
                                        : std::numeric_limits<double>::quiet_NaN();
        out.optim.x         = hawkes_to_unconstrained(init);
        out.optim.f         = -out.log_lik;
        out.optim.iters     = 0;
        out.optim.converged = false;
        out.optim.path.clear();
        out.optim.path.push_back(out.optim.x);
        return;
    }

    const double rate      = static_cast<double>(times.size()) / T;
    const auto   objective = [times, T](const std::array<double, 3>& th) noexcept {
        return -hawkes_log_likelihood(hawkes_from_unconstrained(th), times, T);
    };
    const auto run_into = [&](OptimResult<3>& res, HawkesParams start, const OptimOptions& opts) {
        const std::array<double, 3> x0 = detail::hawkes_start_theta(start, rate);
        if (use_bfgs) {
            bfgs_into(res, objective, x0, opts);
        } else {
            nelder_mead_into(res, objective, x0, opts);
        }
    };

    run_into(out.optim, init, o);

    // 再試行判定: 固定の参照点のうち最良のものが当てはめ結果より高い尤度なら、そこから再当てはめする
    HawkesParams best_ref{};
    double       best_ref_f = std::numeric_limits<double>::infinity();  // −LL
    for (const auto& eb : detail::kHawkesReferenceStarts) {
        const double       beta = eb[1] * rate;
        const HawkesParams ref{rate * (1.0 - eb[0]), eb[0] * beta, beta};
        const double       f = -hawkes_log_likelihood(ref, times, T);
        if (f < best_ref_f) {
            best_ref_f = f;
            best_ref   = ref;
        }
    }
    if (best_ref_f < out.optim.f) {
        OptimOptions retry = o;
        retry.max_iter     = std::min(o.max_iter, detail::kHawkesRetryMaxIter);
        run_into(out.scratch, best_ref, retry);
        if (out.scratch.f < out.optim.f) std::swap(out.optim, out.scratch);  // O(1)、確保なし
    }
    out.params  = hawkes_from_unconstrained(out.optim.x);
    out.log_lik = -out.optim.f;
}

/// 最尤推定（値を返す版）。hawkes_fit_into の薄いラッパ。
inline HawkesFit hawkes_fit(std::span<const double> times, double T, HawkesParams init, OptimOptions o = {},
                            bool use_bfgs = false) {
    HawkesFit out;
    hawkes_fit_into(out, times, T, init, o, use_bfgs);
    return out;
}

}  // namespace quantviz::core
