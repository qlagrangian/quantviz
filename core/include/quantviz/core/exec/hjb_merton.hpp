#pragma once
// core/exec/hjb_merton.hpp — Merton の消費なし最適投資問題（CRRA 効用）の HJB 方程式を対数富の格子で
// 満期から後退に解く。`step_backward()` 1 回 = 1 時間ステップで、HJB シーンはこれを 1 反復ずつ呼んで
// `values()` / `optimal_fraction()` を面と折れ線に積む。閉形式 `merton_fraction` / `merton_value` は
// 数値解の参照線。
//
//   max E[U(W_T)],  U(w) = w^{1−γ}/(1−γ)（γ = 1 は log w）,  dW = [r + π(μ−r)] W dt + π σ W dB
//   HJB: V_t + max_π { [r + π(μ−r)] w V_w + ½ π² σ² w² V_ww } = 0,  V(w, T) = U(w)
//   閉形式（Merton 1969, Rev. Econ. Stat. 51(3); 1971, J. Econ. Theory 3(4)）: π* = (μ−r) / (γσ²),
//          V(w, t) = U(w) e^{(1−γ) κ (T−t)},  κ = r + (μ−r)² / (2γσ²)   （γ = 1 は V = ln w + κ (T−t)）
//
// 制御は π ∈ [0, kHjbPiMax = 5]（空売りなし・レバレッジ 5 倍まで）に制限する。制約付きの解も分離形のままで、
// π_c = clamp(π*, 0, 5) の定数方策の値関数 V = U(w) e^{(1−γ) κ_c τ}, κ_c = r + π_c (μ−r) − ½ γ π_c² σ² になる
// （クランプが効かなければ κ_c ≡ κ）。`merton_fraction` / `merton_value` と境界の Dirichlet 値はすべて
// この**制約付き**の形を使う: 非制約の κ で境界を伸ばすと、クランプ後の π で解く内部と境界が別の速度で
// 成長し、収束先が別の関数になる（実測、修正前: μ = 0.01 < r で境界の相対誤差 3.3e-3 = e^{(1−γ)(κ−κ_c)T} − 1、
// γ = 0.1（π* = 12.5）で 1.1e-1）。κ_c は |κ_c| ≤ |r| + 5|μ−r| で有界なので exp がオーバーフローしない
// （非制約の κ は σ → 0 で発散し、広いスライダー範囲では values() が非有限になった）。
//
// 格子は対数富 y = ln w で等間隔（`wealth()` は w_i = e^{y_i}、両端は w_min / w_max ちょうど）。
// y では dY = [r + π(μ−r) − ½π²σ²] dt + πσ dB で係数が w に依らないので、離散化が格子全体で同じ
// 条件数になる。`n_w` は節点数（≥ 8）、`n_t` は時間ステップ数（≥ 1）。
//
// 各節点の最適比率（`optimal_fraction()`）は中心差分 V_y, V_yy から作る離散ハミルトニアン
//   H(π) = π (μ−r) V_y + ½ π² σ² (V_yy − V_y)   （π に依らない r V_y は省く）
// の [0, kPiMax] 上の最大点。V_y − V_yy > 0（w で凹）なら π は凹 2 次式の解析最大点
// (μ−r) V_y / (σ² (V_y − V_yy)) を区間にクランプしたもの、そうでなければ端点 0 / kPiMax の良い方。
// 境界節点の π は隣の内点をコピーする（境界値は Dirichlet で置くので差分が取れない）。
//
// 時間ステップ（τ = T − t、Δτ = T / n_t）は「方策を凍結した陰的 Euler」= 方策反復 1 回:
//   (1) π_i = 現在の V^m から上の最大点（陽的予測子）
//   (2) (I − Δτ L_π) V^{m+1} = V^m + 境界寄与 を三重対角（Thomas 法、math/tridiag.hpp）で解く
//   L_π V_i = d_i (V_{i+1} − 2V_i + V_{i−1})/h² + m_i (V_{i+1} − V_{i−1})/(2h),
//   d_i = ½ π_i² σ²,  m_i = r + π_i (μ−r) − ½ π_i² σ²（Itô 補正込みの y のドリフト）
// ドリフト項は d_i/h² ≥ |m_i|/(2h) なら中心差分、そうでなければ風上差分（m_i > 0 なら前進、< 0 なら
// 後退）に切り替える。どちらでも非対角係数 ≥ 0、対角 = 1 + Δτ(下 + 上) ≥ 1 なので行列は厳密に対角優位な
// M 行列: 陰的ステップは Δτ と h に関わらず最大値ノルムで安定（|V^{m+1}|∞ ≤ max(|V^m|∞, 境界値)）、
// 単調（凹性・単調性を壊す振動を作らない）、Thomas 法のピボットは ≥ 1 で退化しない。標準パラメータ
// （π* = 0.42、σ = 0.2）では中心差分の条件 h ≤ π²σ²/|m| ≈ 0.15 が n_w ≥ 32 で満たされ、風上は
// π → 0（μ ≈ r）の節点だけで働く。Crank–Nicolson（θ = ½）なら時間 2 次だが、Δτ > h²/d で陽的側の対角が
// 負になり離散最大値原理を失うので θ = 1 を採る。ただし凍結する π は前の時間層の V から作った「遅れた
// 方策」で、方策反復（Howard）を収束まで回してはいない: 各時間層は M 行列の線形解 1 回であり、無条件安定で
// 離散最大値原理を満たすが、Barles–Souganidis（単調 + 安定 + 整合 → 粘性解へ収束）を Forsyth–Labahn (2007)
// の形でそのまま主張するには各層で方策反復を収束させる必要がある。ここでは実測の収束（時間 1 次・空間 2 次、
// HJB-05）で裏付ける。
//
// 精度は時間 1 次（陰的 Euler; 凍結した π の誤差 O(Δτ) は最適点で ∂H/∂π = 0 なので V には 2 次でしか
// 入らない）、空間 2 次。実測（HJB-05、t = 0 の全節点最大相対誤差、標準パラメータ）:
//   空間だけ（n_t = 4096）: n_w = 64 / 128 / 256 で 1.5e-4 / 3.7e-5 / 8.5e-6（比 ≈ 4、符号 +）
//   時間だけ（n_w = 2048）: n_t = 32 / 64 / 128 で 1.0e-4 / 5.1e-5 / 2.5e-5（比 2.0、符号 −）
//   同時（n_w = n_t = N）: N = 64 / 128 / 256 / 512 で 1.0e-4 / 1.2e-5 / 3.4e-6 / 4.1e-6。符号が逆なので
//   N ≈ 256〜512 で部分的に打ち消し合い、比は一様でない（8.4 / 3.5 / 0.8）。時間誤差が支配する N ≥ 512 で
//   比 → 2。シーンの 200 × 200 では面全体（全時間層）の最大相対誤差 1.0e-6（≤ 2e-6）。
// 最適比率 π*(w) の誤差（HJB-01, n_w = n_t = 256）は内側 80 % の節点で 1.5e-5、境界の隣で 1.2e-4。
//
// 境界（y_min, y_max）は Dirichlet で閉形式を置く: V(w_b, t) = D_b · e^{(1−γ)κ_c τ}（γ = 1: D_b + κ_c τ）、
// D_b は終端データの端の値（init(p) なら U(w_b) なので閉形式そのもの）。閉形式が分離形 U(w) g(τ) で
// 半群 g(a + b) = g(a) g(b) を満たすため、途中時刻の値を終端にして再開しても境界は同じ物理時刻で同じ値
// になる（HJB-06）。境界を正確値で固定するぶん、離散解の誤差は境界で 0 に押し付けられ、その勾配が
// 境界近傍の π*（V_y / (V_y − V_yy) の比）に現れる。HJB-01 が両端 10 % を除くのはそのため。
//
// 確保はすべて init（std::vector の resize）。value_at は [w_min, w_max] の外・NaN で NaN（外挿しない）。
// 不正入力は hjb_sanitize で決定的に丸める（NaN は既定値 kHjbDefaults、範囲外は上下限にクランプ、
// w_max ≤ w_min は w_max を w_min × 25 に押し上げる）。
//
// γ が 1 に近い（|γ−1| ≈ 1e-7）と U = w^{1−γ}/(1−γ) ≈ ∓1e7 + ln w で有効桁を約 7 桁失う。γ == 1.0 ちょうどは
// log の分岐で正確なので、シーンは |γ−1| < 1e-4 を 1 に丸めることを推奨する（sanitize では丸めない: 数値的
// には連続な極限で、丸めは UI 側の判断）。

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "quantviz/core/math/tridiag.hpp"

namespace quantviz::core {

struct HjbParams {
    double      mu, r, sigma, gamma /* CRRA */, T;
    double      w_min, w_max;
    std::size_t n_w, n_t;  ///< n_w: 富の節点数, n_t: 時間ステップ数
};

/// NaN の既定値。HJB シーンの初期パラメータでもある（μ = 8 %, r = 3 %, σ = 20 %, γ = 3 → π* = 0.4167）。
inline constexpr HjbParams kHjbDefaults{0.08, 0.03, 0.2, 3.0, 1.0, 0.2, 5.0, 256, 256};

/// 最適比率の上限（レバレッジ 5 倍）。下限は 0（空売りなし）。閉形式もこの制約付きで書く。
inline constexpr double kHjbPiMax = 5.0;

/// 不正入力の決定的な丸め: NaN / ±inf → kHjbDefaults、|μ|, |r| ≤ 10、σ ≥ 1e-4、γ ≥ 1e-3、T ≥ 1e-6、
/// w_min ≥ 1e-6、w_max > w_min（満たさなければ w_min × 25）、8 ≤ n_w ≤ 4096、1 ≤ n_t ≤ 100000。
[[nodiscard]] inline HjbParams hjb_sanitize(HjbParams p) noexcept {
    constexpr double      kRateMax = 10.0, kSigmaMin = 1e-4, kGammaMin = 1e-3, kTMin = 1e-6;
    constexpr double      kWMinFloor = 1e-6;
    constexpr std::size_t kNwMax = 4096, kNtMax = 100000;
    const HjbParams&      d = kHjbDefaults;
    p.mu = std::isfinite(p.mu) ? std::clamp(p.mu, -kRateMax, kRateMax) : d.mu;
    p.r  = std::isfinite(p.r) ? std::clamp(p.r, -kRateMax, kRateMax) : d.r;
    p.sigma = std::isfinite(p.sigma) ? std::max(p.sigma, kSigmaMin) : d.sigma;
    p.gamma = std::isfinite(p.gamma) ? std::max(p.gamma, kGammaMin) : d.gamma;
    p.T     = std::isfinite(p.T) ? std::max(p.T, kTMin) : d.T;
    p.w_min = std::isfinite(p.w_min) ? std::max(p.w_min, kWMinFloor) : d.w_min;
    if (!std::isfinite(p.w_max)) p.w_max = d.w_max;
    if (!(p.w_max > p.w_min)) p.w_max = p.w_min * (d.w_max / d.w_min);
    p.n_w = std::clamp<std::size_t>(p.n_w, 8, kNwMax);
    p.n_t = std::clamp<std::size_t>(p.n_t, 1, kNtMax);
    return p;
}

/// CRRA 効用 U(w) = w^{1−γ}/(1−γ)（γ = 1 は ln w）。w ≤ 0 / NaN は NaN。
[[nodiscard]] inline double crra_utility(double w, double gamma) noexcept {
    if (!(w > 0.0)) return std::numeric_limits<double>::quiet_NaN();
    return gamma == 1.0 ? std::log(w) : std::pow(w, 1.0 - gamma) / (1.0 - gamma);
}

/// 制約付き最適比率 π_c = clamp((μ−r)/(γσ²), 0, kHjbPiMax)。p は sanitize 済みであること。
[[nodiscard]] inline double merton_fraction_clamped(const HjbParams& p) noexcept {
    return std::clamp((p.mu - p.r) / (p.gamma * p.sigma * p.sigma), 0.0, kHjbPiMax);
}

/// 制約付きの成長率 κ_c = r + π_c(μ−r) − ½γπ_c²σ²（V の指数は (1−γ)κ_c）。クランプが効かなければ
/// r + (μ−r)²/(2γσ²) に一致する。|κ_c| ≤ |r| + 5|μ−r|。p は sanitize 済みであること。
[[nodiscard]] inline double merton_kappa(const HjbParams& p) noexcept {
    const double ex = p.mu - p.r;
    const double pc = merton_fraction_clamped(p);
    return p.r + pc * ex - 0.5 * p.gamma * pc * pc * p.sigma * p.sigma;
}

/// 解析解 π* = (μ−r)/(γσ²) を [0, kHjbPiMax] にクランプしたもの（数値解と同じ制約）。
/// パラメータは hjb_sanitize 済みとして扱う（数値解と同じ入力に揃える）。
[[nodiscard]] inline double merton_fraction(HjbParams p) noexcept {
    return merton_fraction_clamped(hjb_sanitize(p));
}

/// 制約付き閉形式 V(w, t) = U(w) e^{(1−γ)κ_c(T−t)}（γ = 1: ln w + κ_c(T−t)）。w ≤ 0 / NaN は NaN。
[[nodiscard]] inline double merton_value(HjbParams p, double w, double t) noexcept {
    p                = hjb_sanitize(p);
    const double tau = p.T - t;
    const double u   = crra_utility(w, p.gamma);
    if (p.gamma == 1.0) return u + merton_kappa(p) * tau;
    return u * std::exp((1.0 - p.gamma) * merton_kappa(p) * tau);
}

class HjbMerton {
public:
    /// 最適比率の上限（= kHjbPiMax、レバレッジ 5 倍）。下限は 0（空売りなし）。
    static constexpr double kPiMax = kHjbPiMax;

    HjbMerton() = default;

    /// 格子とパラメータを設定し、終端 V(w, T) = U(w) を置いて t = T にする。ヒープ確保はここだけ。
    /// それ以前に wealth() / values() / optimal_fraction() が返した span は無効になる。
    void init(HjbParams p) { init(p, {}); }

    /// 終端データを指定して init する: V(w_i, T) = terminal[i]（同じ w 格子、size == n_w）。途中時刻の
    /// values() を渡して T を残り時間にすれば、そこから解き直せる（HJB-06 の時間整合）。size が n_w
    /// （sanitize 後）と一致しなければ U(w) に落ちる。値はそのまま置く（検査しない）。
    /// terminal が自身の values() を指していてもよい: 長さが同じなら resize は再確保せず span は有効のままで
    /// 自己コピーは省く。長さが違えば（再確保されうる）terminal からは size() しか読まない。
    void init(HjbParams p, std::span<const double> terminal) {
        allocate(hjb_sanitize(p));
        if (terminal.size() == w_.size()) {
            if (terminal.data() != v_.data()) std::copy(terminal.begin(), terminal.end(), v_.begin());
        } else {
            for (std::size_t i = 0; i < w_.size(); ++i) v_[i] = crra_utility(w_[i], p_.gamma);
        }
        start();
    }

    /// 1 時間ステップ後退（方策を凍結した陰的 Euler）。t が既に 0（remaining() == 0）なら何もせず false。
    /// 三重対角ソルバが失敗した（NaN 入力）場合は内点を旧値のまま境界と時刻だけ進める。
    bool step_backward() noexcept {
        if (remaining_ == 0 || y_.empty()) return false;
        const std::size_t n       = y_.size();
        const std::size_t m       = n - 2;  // 内点 i = 1..n−2 ↔ j = i − 1
        const double      dtau    = p_.T / static_cast<double>(p_.n_t);
        const double      t_new   = p_.T * static_cast<double>(remaining_ - 1) / static_cast<double>(p_.n_t);
        const double      tau_new = p_.T - t_new;
        const double      ex      = p_.mu - p_.r;
        const double      s2      = p_.sigma * p_.sigma;
        const double      inv_h2  = 1.0 / (h_ * h_);
        const double      inv_2h  = 0.5 / h_;
        const double      inv_h   = 1.0 / h_;

        double lo_first = 0.0, up_last = 0.0;
        for (std::size_t j = 0; j < m; ++j) {
            const double pi    = pi_[j + 1];
            const double d     = 0.5 * pi * pi * s2;
            const double drift = p_.r + pi * ex - d;
            double       lo    = d * inv_h2 - drift * inv_2h;  // V_{i−1} の係数
            double       up    = d * inv_h2 + drift * inv_2h;  // V_{i+1} の係数
            if (lo < 0.0 || up < 0.0) {                          // 中心差分が正係数を破る → 風上
                lo = d * inv_h2 + (drift < 0.0 ? -drift * inv_h : 0.0);
                up = d * inv_h2 + (drift > 0.0 ? drift * inv_h : 0.0);
            }
            b_[j] = 1.0 + dtau * (lo + up);
            if (j > 0) a_[j - 1] = -dtau * lo;
            if (j + 1 < m) c_[j] = -dtau * up;
            rhs_[j] = v_[j + 1];
            if (j == 0) lo_first = lo;
            if (j + 1 == m) up_last = up;
        }
        const double v_lo = boundary_value(term_lo_, tau_new);
        const double v_hi = boundary_value(term_hi_, tau_new);
        rhs_[0] += dtau * lo_first * v_lo;
        rhs_[m - 1] += dtau * up_last * v_hi;

        if (tridiag_solve(a_, b_, c_, rhs_, rhs_, work_)) std::copy(rhs_.begin(), rhs_.end(), v_.begin() + 1);
        v_[0]     = v_lo;
        v_[n - 1] = v_hi;

        --remaining_;
        time_ = t_new;
        update_policy();
        return true;
    }

    [[nodiscard]] std::size_t             remaining() const noexcept { return remaining_; }
    [[nodiscard]] double                  time() const noexcept { return time_; }
    [[nodiscard]] std::span<const double> wealth() const noexcept { return w_; }
    [[nodiscard]] std::span<const double> values() const noexcept { return v_; }
    /// 現時点の π*(w)（各節点の離散ハミルトニアンの最大点、[0, kPiMax]）。step_backward の後は**新しい** V
    /// から再計算した値、すなわち次の step_backward が凍結して使う方策。
    [[nodiscard]] std::span<const double> optimal_fraction() const noexcept { return pi_; }
    [[nodiscard]] const HjbParams&        params() const noexcept { return p_; }

    /// 現在時刻の V(w)。y = ln w で線形補間。[w_min, w_max] の外・NaN・未初期化は NaN。
    [[nodiscard]] double value_at(double w) const noexcept {
        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        if (y_.empty() || !(w >= p_.w_min) || !(w <= p_.w_max)) return kNaN;
        const std::size_t n = y_.size();
        const double      x = (std::log(w) - y_[0]) / h_;
        if (!(x >= 0.0)) return v_[0];
        const std::size_t i = std::min(static_cast<std::size_t>(x), n - 2);
        const double      t = std::clamp(x - static_cast<double>(i), 0.0, 1.0);
        return (1.0 - t) * v_[i] + t * v_[i + 1];
    }

private:
    void allocate(HjbParams p) {
        p_                  = p;
        const std::size_t n = p_.n_w;
        y_.resize(n);
        w_.resize(n);
        v_.resize(n);
        pi_.resize(n);
        rhs_.resize(n - 2);
        work_.resize(n - 2);
        a_.resize(n - 3);
        b_.resize(n - 2);
        c_.resize(n - 3);

        const double y_min = std::log(p_.w_min);
        const double y_max = std::log(p_.w_max);
        h_                 = (y_max - y_min) / static_cast<double>(n - 1);
        for (std::size_t i = 0; i < n; ++i) {
            y_[i] = y_min + h_ * static_cast<double>(i);
            w_[i] = std::exp(y_[i]);
        }
        w_[0]     = p_.w_min;  // 両端は入力値ちょうど（exp(log(w)) の丸めを避ける）
        w_[n - 1] = p_.w_max;
        kappa_    = merton_kappa(p_);
    }

    /// 終端データが v_ に入った状態で時刻・境界・π を初期化する。
    void start() noexcept {
        remaining_ = p_.n_t;
        time_      = p_.T;
        term_lo_   = v_.front();
        term_hi_   = v_.back();
        update_policy();
    }

    /// 境界の Dirichlet 値: 終端データの端 D_b を閉形式の時間因子で伸ばす（γ = 1 は加法）。
    [[nodiscard]] double boundary_value(double terminal, double tau) const noexcept {
        if (p_.gamma == 1.0) return terminal + kappa_ * tau;
        return terminal * std::exp((1.0 - p_.gamma) * kappa_ * tau);
    }

    /// 各内点の π* = argmax_{[0, kPiMax]} H(π)、H(π) = π(μ−r)V_y + ½π²σ²(V_yy − V_y)（中心差分）。
    /// 境界節点は隣をコピー。NaN が出た節点は比較が偽になり 0 に落ちる。
    void update_policy() noexcept {
        const std::size_t n  = y_.size();
        const double      ex = p_.mu - p_.r;
        const double      s2 = p_.sigma * p_.sigma;
        for (std::size_t i = 1; i + 1 < n; ++i) {
            const double vy    = (v_[i + 1] - v_[i - 1]) / (2.0 * h_);
            const double vyy   = (v_[i + 1] - 2.0 * v_[i] + v_[i - 1]) / (h_ * h_);
            const double denom = vy - vyy;  // > 0 ⇔ w で凹 ⇔ H は π の凹 2 次式
            double       pi;
            if (denom > 0.0 && std::isfinite(denom)) {
                pi = std::clamp(ex * vy / (s2 * denom), 0.0, kPiMax);
            } else {
                const double h_max = kPiMax * ex * vy + 0.5 * kPiMax * kPiMax * s2 * (vyy - vy);
                pi                 = h_max > 0.0 ? kPiMax : 0.0;  // H(0) = 0 との比較
            }
            pi_[i] = pi;
        }
        pi_[0]     = pi_[1];
        pi_[n - 1] = pi_[n - 2];
    }

    HjbParams p_{kHjbDefaults};

    std::vector<double> y_, w_, v_, pi_;
    std::vector<double> rhs_, work_;
    std::vector<double> a_, b_, c_;  ///< (I − Δτ L_π) の三重対角（内点）

    double      h_         = 0.0;
    double      kappa_     = 0.0;
    double      term_lo_   = 0.0;  ///< 終端データの両端（境界値の基点）
    double      term_hi_   = 0.0;
    std::size_t remaining_ = 0;
    double      time_      = 0.0;
};

}  // namespace quantviz::core
