#pragma once
// core/pricing/fdm_cn.hpp — Black–Scholes PDE の Crank–Nicolson 後退反復（European: 三重対角ソルバ、
// American: PSOR）。S 均等格子 [0, S_max] を n_space 分割（n_space + 1 節点）、時間 [0, T] を n_time
// 分割し、満期 t = T のペイオフから `step_backward()` 1 回につき 1 ステップ t = 0 へ向かって進める。
// FDM シーンはこれを 1 反復ずつ呼んで `values()` / `exercise_boundary()` を 3D 面に積む。
//
//   V_τ = ½σ²S² V_SS + (r−q) S V_S − r V,   τ = T − t
//   S_i = i h, h = S_max / N。中心差分で L V_i = α_i V_{i−1} + β_i V_i + γ_i V_{i+1},
//   α_i = ½σ²i² − ½(r−q)i,  β_i = −σ²i² − r,  γ_i = ½σ²i² + ½(r−q)i（h は打ち消えて i だけ残る）
//   θ 法: (I − θΔτ L) V^{m+1} = (I + (1−θ)Δτ L) V^m + 境界寄与
//
// 精度の担い手は 2 つあり、役割は別（どちらも実測、N = 200, T = 1, K = 100）:
//
// (1) キンクのセル平均 — 価格精度と 2 次収束（FDM-02/03）を担う。満期ペイオフの折れ点 K が格子点に
//     乗ると（S_max = 4K, N = 200 で K は 50 番目の節点）中心差分の誤差定数が大きく、N = M = 200 の
//     ATM 相対誤差は 9.5e-4（許容 1e-3 ぎりぎり）、S = 90 では 1.7e-3 で許容を割る。最初の後退ステップの
//     直前に K を内部に含むセルの節点値をペイオフのセル平均 (1/h)∫ payoff dS に置き換える
//     （Pooley–Vetzal–Forsyth 2003）と ATM 相対誤差は 5e-5、誤差比は 4.1 / 4.0 になる。これは
//     Rannacher なしでも同じ（4.8e-5、比 4.15 / 4.04）で、Rannacher は ATM 価格を相対 2e-6 しか動かさない。
//     K が節点間にあると value_at の線形補間誤差 h²Γ/8 が加わり、S_max = 404（K/h = 49.5）では 8.7e-4
//     まで許容に迫る（FDM-02 の SECTION）。平均化は PDE の初期値にだけ効かせ、t = T で values() が
//     返すのは正確なペイオフのまま（FDM-01 / FDMSCENE-04「満期ペイオフ」）。American の障害
//     （本源的価値）も正確なペイオフを使う。
//
// (2) Rannacher 始動 — 粗い時間格子での Γ の滑らかさを担う。最初の kRannacherSteps (= 2) ステップを、
//     それぞれ陰的 Euler（θ = 1）の半ステップ ×2 に置き換える（Rannacher 1984 / Giles–Carter 2006）。
//     ペイオフのキンクが励起する高周波成分を CN（|増幅率| → 1）は減衰できないので、M が小さいと Γ が
//     節点ごとに振動する。離散 Γ の 2 階差分 max|Γ_{i+1} − 2Γ_i + Γ_{i−1}| は、CN + セル平均だけだと
//     M = 10 で 4.3e-2、M = 5 で 2.8e-1。Rannacher ありでは 2.9e-4 / 2.8e-4 で、M = 200 の収束値 2.88e-4
//     と同じ。FDM シーンを StepOnce で手送りするときの小さい M がまさにこの領域で、FDM-05 の SECTION が
//     固定する。時間刻みの総和は変わらない（1 回の step_backward = Δτ = T / n_time）。
//
// 境界条件（τ における値。American はこれを本源的価値で下から抑える = max(European 境界, ペイオフ)）:
//   put : V(0) = K e^{−rτ}（American は K）,  V(S_max) = 0
//   call: V(0) = 0,                            V(S_max) = S_max e^{−qτ} − K e^{−rτ}
//
// American: LCP  V ≥ payoff, (A V − rhs) ≥ 0, 補完性。各ステップで PSOR（射影 SOR）:
//   v_i ← max(payoff_i, v_i + ω (gs_i − v_i)),  gs_i = (rhs_i − a v_{i−1} − c v_{i+1}) / b_i
// 前ステップの値を初期値にし、1 スイープの最大更新量が psor_tol 未満で収束。反復回数と収束フラグは
// last_psor_iterations() / last_psor_converged() で読める。psor_max_iter で打ち切った場合も時間は進む
// （シーンは status に反映する）。European では last_psor_converged() は三重対角ソルバの成否、
// last_psor_iterations() は 0。
//
// 行使境界（American のみ、それ以外は NaN）: V_i が本源的価値と一致（差 ≤ kBoundaryTol·max(K,1)）する
// 節点のうち、put は S_i < K で最大の S_i、call は S_i > K で最小の S_i。該当なしなら NaN。
// PSOR の射影は行使域で V_i = payoff_i をビット一致で与えるので、許容は事実上「射影が起きたか」の判定。
//
// 確保はすべて init（std::vector の resize）で行い、step_backward / value_at / delta_at は確保なし・
// 例外なし。不正入力は init で決定的に丸める（n_space ≥ 2, n_time ≥ 1, S_max > 0, T ≥ 0, σ ≥ 0,
// ω ∈ [1, 1.99], psor_tol > 0, psor_max_iter ≥ 1。NaN は既定値へ）。
// value_at / delta_at は [0, S_max] の外や NaN で NaN を返す（外挿しない）。

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "quantviz/core/math/tridiag.hpp"
#include "quantviz/core/pricing/black_scholes.hpp"

namespace quantviz::core {

struct FdmGrid {
    double      s_max;
    std::size_t n_space;  ///< 空間分割数 N（節点は N + 1）
    std::size_t n_time;   ///< 時間分割数 M
};

struct FdmParams {
    double      K, T, r, sigma, q;
    OptionType  type;
    bool        american;
    double      psor_omega    = 1.2;    ///< SOR 緩和係数。init で [1, 1.99] に丸める
    std::size_t psor_max_iter = 10000;  ///< 1 サブステップあたりの最大スイープ数
    /// PSOR の停止許容: 1 スイープの最大更新量 max|Δv_i| がこれ未満で収束。価格スケールの**絶対量**
    /// （K = 100 なら 1e-10 は相対 1e-12 に相当。K を 1 桁変えたら合わせて変える）。
    double psor_tol = 1e-10;
};

class FdmCn {
public:
    /// 陰的 Euler 半ステップ ×2 に置き換える先頭ステップ数（Rannacher 始動）。
    static constexpr std::size_t kRannacherSteps = 2;
    /// 行使境界判定の相対許容（× max(K, 1)）。
    static constexpr double kBoundaryTol = 1e-9;

    FdmCn() = default;

    /// 格子とパラメータを設定し、満期ペイオフを置いて t = T にする。ヒープ確保はここだけ。
    /// それ以前に spots() / values() が返した span は無効になる（再確保されうる）。
    void init(FdmGrid g, FdmParams p) {
        grid_   = sanitize(g, p);
        params_ = sanitize(p);

        const std::size_t n = grid_.n_space;
        spots_.resize(n + 1);
        payoff_.resize(n + 1);
        v_.resize(n + 1);
        rhs_.resize(n - 1);
        work_.resize(n - 1);
        alpha_.resize(n - 1);
        beta_.resize(n - 1);
        gamma_.resize(n - 1);
        a_.resize(n - 2);
        b_.resize(n - 1);
        c_.resize(n - 2);

        for (std::size_t i = 0; i <= n; ++i) {
            spots_[i]  = grid_.s_max * static_cast<double>(i) / static_cast<double>(n);
            payoff_[i] = intrinsic(spots_[i]);
            v_[i]      = payoff_[i];
        }
        const double s2 = params_.sigma * params_.sigma;
        const double mu = params_.r - params_.q;
        for (std::size_t j = 0; j + 1 < n; ++j) {
            const double i = static_cast<double>(j + 1);
            alpha_[j]      = 0.5 * s2 * i * i - 0.5 * mu * i;
            beta_[j]       = -s2 * i * i - params_.r;
            gamma_[j]      = 0.5 * s2 * i * i + 0.5 * mu * i;
        }

        remaining_       = grid_.n_time;
        time_            = params_.T;
        asm_theta_       = std::numeric_limits<double>::quiet_NaN();  // 次の sub_step で必ず組み直す
        asm_dt_          = std::numeric_limits<double>::quiet_NaN();
        last_iterations_ = 0;
        last_converged_  = true;
    }

    /// 1 時間ステップ後退。t が既に 0（remaining() == 0）なら何もせず false。
    bool step_backward() noexcept {
        if (remaining_ == 0 || spots_.empty()) return false;
        const std::size_t m    = grid_.n_time;
        const std::size_t done = m - remaining_;
        const double      t_new = params_.T * static_cast<double>(remaining_ - 1) / static_cast<double>(m);
        const double      dt    = params_.T / static_cast<double>(m);

        last_iterations_ = 0;
        last_converged_  = true;
        if (done == 0) smooth_kink();
        if (done < kRannacherSteps) {
            const double t_mid = 0.5 * (time_ + t_new);
            sub_step(1.0, 0.5 * dt, params_.T - t_mid);
            sub_step(1.0, 0.5 * dt, params_.T - t_new);
        } else {
            sub_step(0.5, dt, params_.T - t_new);
        }
        --remaining_;
        time_ = t_new;
        return true;
    }

    [[nodiscard]] std::size_t remaining() const noexcept { return remaining_; }
    [[nodiscard]] double      time() const noexcept { return time_; }
    [[nodiscard]] std::span<const double> spots() const noexcept { return spots_; }
    [[nodiscard]] std::span<const double> values() const noexcept { return v_; }
    [[nodiscard]] const FdmGrid&          grid() const noexcept { return grid_; }
    [[nodiscard]] const FdmParams&        params() const noexcept { return params_; }

    /// 現在時刻の V(s)。格子点間は線形補間。[0, S_max] の外・NaN・未初期化は NaN。
    [[nodiscard]] double value_at(double s) const noexcept {
        std::size_t i;
        double      w;
        if (!locate(s, i, w)) return std::numeric_limits<double>::quiet_NaN();
        return (1.0 - w) * v_[i] + w * v_[i + 1];
    }

    /// 現在時刻の ∂V/∂S(s)。節点では中心差分（端は片側差分）、節点間はそれを線形補間。
    [[nodiscard]] double delta_at(double s) const noexcept {
        std::size_t i;
        double      w;
        if (!locate(s, i, w)) return std::numeric_limits<double>::quiet_NaN();
        return (1.0 - w) * node_delta(i) + w * node_delta(i + 1);
    }

    /// American の行使境界 S*。put は V = 本源的価値となる S < K の最大節点、call は S > K の最小節点。
    /// European、または該当節点なしなら NaN。
    [[nodiscard]] double exercise_boundary() const noexcept {
        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        if (!params_.american || spots_.empty()) return kNaN;
        const std::size_t n   = grid_.n_space;
        const double      tol = kBoundaryTol * std::max(params_.K, 1.0);
        if (params_.type == OptionType::Put) {
            for (std::size_t i = n + 1; i-- > 0;) {
                if (!(spots_[i] < params_.K)) continue;
                if (v_[i] - payoff_[i] <= tol) return spots_[i];
            }
        } else {
            for (std::size_t i = 0; i <= n; ++i) {
                if (!(spots_[i] > params_.K)) continue;
                if (v_[i] - payoff_[i] <= tol) return spots_[i];
            }
        }
        return kNaN;
    }

    /// 直前の step_backward() で回した PSOR スイープ数。Rannacher の最初の 2 ステップは半ステップ 2 回の
    /// **合算**（psor_max_iter で打ち切ると 2 × psor_max_iter になる）。European は 0。
    [[nodiscard]] std::size_t last_psor_iterations() const noexcept { return last_iterations_; }
    /// 直前の step_backward() の解法が成功したか。American: 全サブステップの PSOR が psor_tol で収束。
    /// European: 三重対角ソルバが成功。false でも時間は進んでいる。
    [[nodiscard]] bool last_psor_converged() const noexcept { return last_converged_; }

private:
    static FdmGrid sanitize(FdmGrid g, const FdmParams& p) noexcept {
        g.n_space = std::max<std::size_t>(g.n_space, 2);
        g.n_time  = std::max<std::size_t>(g.n_time, 1);
        if (!(g.s_max > 0.0) || !std::isfinite(g.s_max)) g.s_max = 4.0 * (p.K > 0.0 ? p.K : 1.0);
        return g;
    }
    static FdmParams sanitize(FdmParams p) noexcept {
        if (!(p.K > 0.0)) p.K = 0.0;  // NaN / 負 → 0（call のペイオフは S、put は 0 になる）
        if (!(p.T >= 0.0)) p.T = 0.0;
        if (!(p.sigma >= 0.0)) p.sigma = 0.0;
        if (!std::isfinite(p.r)) p.r = 0.0;
        if (!std::isfinite(p.q)) p.q = 0.0;
        if (!(p.psor_omega >= 1.0)) p.psor_omega = std::isnan(p.psor_omega) ? 1.2 : 1.0;
        if (p.psor_omega > 1.99) p.psor_omega = 1.99;
        if (!(p.psor_tol > 0.0)) p.psor_tol = 1e-10;
        if (p.psor_max_iter == 0) p.psor_max_iter = 1;
        return p;
    }

    [[nodiscard]] double intrinsic(double s) const noexcept {
        return params_.type == OptionType::Call ? std::max(s - params_.K, 0.0) : std::max(params_.K - s, 0.0);
    }

    /// 節点 i と補間重み w（s = S_i + w h, 0 ≤ w ≤ 1）。範囲外 / NaN / 未初期化なら false。
    bool locate(double s, std::size_t& i, double& w) const noexcept {
        const std::size_t n = grid_.n_space;
        if (spots_.empty() || !(s >= 0.0) || !(s <= grid_.s_max)) return false;
        const double h = grid_.s_max / static_cast<double>(n);
        i              = std::min(static_cast<std::size_t>(s / h), n - 1);
        w              = std::clamp((s - spots_[i]) / h, 0.0, 1.0);
        return true;
    }

    [[nodiscard]] double node_delta(std::size_t i) const noexcept {
        const std::size_t n = grid_.n_space;
        const double      h = grid_.s_max / static_cast<double>(n);
        if (i == 0) return (v_[1] - v_[0]) / h;
        if (i >= n) return (v_[n] - v_[n - 1]) / h;
        return (v_[i + 1] - v_[i - 1]) / (2.0 * h);
    }

    /// τ における境界値（American は本源的価値で下から抑える）。
    void boundary_values(double tau, double& lo, double& hi) const noexcept {
        const std::size_t n      = grid_.n_space;
        const double      disc_r = std::exp(-params_.r * tau);
        const double      disc_q = std::exp(-params_.q * tau);
        if (params_.type == OptionType::Put) {
            lo = params_.K * disc_r;
            hi = 0.0;
        } else {
            lo = 0.0;
            hi = grid_.s_max * disc_q - params_.K * disc_r;
        }
        if (params_.american) {
            lo = std::max(lo, payoff_[0]);
            hi = std::max(hi, payoff_[n]);
        }
    }

    /// 最初の後退ステップの直前に 1 回だけ呼ぶ。K を内部に含むセル [S_i − h/2, S_i + h/2] の節点値を
    /// ペイオフのセル平均 (1/h)∫ payoff dS = d²/(2h)（d = キンクからセル端までの距離）に置き換える
    /// （Pooley–Vetzal–Forsyth 2003 の averaging）。K がセル境界に乗る（節点の中点）ときは平均が
    /// 本源的価値と一致するので何もしない。values() が t = T で見せる満期ペイオフは変えない。
    void smooth_kink() noexcept {
        const std::size_t n = grid_.n_space;
        const double      h = grid_.s_max / static_cast<double>(n);
        const double      x = params_.K / h;
        if (!(x > 0.0) || !(x < static_cast<double>(n))) return;  // K が格子の外・端・NaN
        const std::size_t i = static_cast<std::size_t>(x + 0.5);
        if (i == 0 || i >= n) return;  // 境界節点には書かない（境界値は sub_step が別に置く）
        const double lo = spots_[i] - 0.5 * h;
        const double      hi = spots_[i] + 0.5 * h;
        if (!(lo < params_.K && params_.K < hi)) return;
        const double d = params_.type == OptionType::Call ? hi - params_.K : params_.K - lo;
        v_[i]          = 0.5 * d * d / h;
    }

    /// (I − θΔτ L) を三重対角 (a_, b_, c_) に組む。(θ, Δτ) が前回と同じなら何もしない。
    void assemble(double theta, double dt) noexcept {
        if (theta == asm_theta_ && dt == asm_dt_) return;
        asm_theta_          = theta;
        asm_dt_             = dt;
        const std::size_t n = grid_.n_space - 1;  // 内点数
        const double      f = theta * dt;
        for (std::size_t j = 0; j < n; ++j) {
            b_[j] = 1.0 - f * beta_[j];
            if (j > 0) a_[j - 1] = -f * alpha_[j];
            if (j + 1 < n) c_[j] = -f * gamma_[j];
        }
    }

    /// θ 法の 1 サブステップ（Δτ = dt、新しい時刻 τ_new）。
    void sub_step(double theta, double dt, double tau_new) noexcept {
        const std::size_t big_n = grid_.n_space;
        const std::size_t n     = big_n - 1;  // 内点 i = 1..N−1 ↔ j = i − 1
        assemble(theta, dt);

        // 右辺: 旧値の陽的部分（旧境界値を含む）
        const double g = (1.0 - theta) * dt;
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t i = j + 1;
            rhs_[j] = v_[i] + g * (alpha_[j] * v_[i - 1] + beta_[j] * v_[i] + gamma_[j] * v_[i + 1]);
        }
        // 新しい境界値の寄与を右辺へ
        double lo, hi;
        boundary_values(tau_new, lo, hi);
        const double f = theta * dt;
        rhs_[0] += f * alpha_[0] * lo;
        rhs_[n - 1] += f * gamma_[n - 1] * hi;

        if (params_.american) {
            if (!psor()) last_converged_ = false;  // 打ち切り: 途中の反復値のまま時間だけ進める
        } else if (tridiag_solve(a_, b_, c_, rhs_, rhs_, work_)) {
            std::copy(rhs_.begin(), rhs_.end(), v_.begin() + 1);
        } else {
            last_converged_ = false;  // 旧値を保持して時間だけ進める
        }
        v_[0]     = lo;
        v_[big_n] = hi;
    }

    /// 射影 SOR。内点 v_[1..N−1]（前ステップの値 = 初期値）を (a_, b_, c_) v = rhs_、v ≥ payoff_ の
    /// 下で更新する。境界の寄与は rhs_ に入っているので v_[0], v_[N] は読まない。1 スイープの最大更新量が
    /// psor_tol 未満で true。反復回数は last_iterations_ に加算（Rannacher の半ステップ 2 回は合算）。
    /// NaN が出た場合は「収束」と誤判定しないよう、比較は否定形で書く。
    bool psor() noexcept {
        const std::size_t n     = grid_.n_space - 1;
        const double      omega = params_.psor_omega;
        for (std::size_t it = 0; it < params_.psor_max_iter; ++it) {
            double max_delta = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                const std::size_t i  = j + 1;
                double            gs = rhs_[j];
                if (j > 0) gs -= a_[j - 1] * v_[i - 1];
                if (j + 1 < n) gs -= c_[j] * v_[i + 1];
                gs /= b_[j];
                const double vn = std::max(payoff_[i], v_[i] + omega * (gs - v_[i]));
                const double d  = std::abs(vn - v_[i]);
                if (!(d <= max_delta)) max_delta = d;  // NaN は max_delta を NaN にして収束させない
                v_[i] = vn;
            }
            ++last_iterations_;
            if (max_delta < params_.psor_tol) return true;
        }
        return false;
    }

    FdmGrid   grid_{0.0, 0, 0};
    FdmParams params_{0.0, 0.0, 0.0, 0.0, 0.0, OptionType::Call, false};

    std::vector<double> spots_, payoff_, v_;
    std::vector<double> rhs_, work_;
    std::vector<double> alpha_, beta_, gamma_;  ///< L の係数（内点、時間不変）
    std::vector<double> a_, b_, c_;             ///< (I − θΔτ L) の三重対角

    std::size_t remaining_       = 0;
    double      time_            = 0.0;
    double      asm_theta_       = std::numeric_limits<double>::quiet_NaN();
    double      asm_dt_          = std::numeric_limits<double>::quiet_NaN();
    std::size_t last_iterations_ = 0;
    bool        last_converged_  = true;
};

}  // namespace quantviz::core
