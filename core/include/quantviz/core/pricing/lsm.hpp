#pragma once
// core/pricing/lsm.hpp — Longstaff–Schwartz（LSM）による American put / call のモンテカルロ価格。
// GBM（無配当、リスク中立ドリフト r）のパスをアンチセティック対で init に生成し、満期 t = T から
// `step_backward()` 1 回につき 1 行使時点ずつ t = 0 へ向かって後ろ向き回帰を進める。LSM シーンはこれを
// 1 反復ずつ呼び、path(i) / quantile_at / continuation_value を描く。
//
// パス: S_{k+1} = S_k exp((r − σ²/2) dt + σ√dt Z)（厳密離散化、dt = T / n_steps）。対 j = 0..n_paths/2−1 は
// ステップごとに Z を 1 本引き、パス 2j に +Z、パス 2j+1 に −Z を与える（Rng は core::Rng(seed)、
// 引く順序は「対 → ステップ」）。n_paths が奇数なら init が +1 に丸める。パスは row-major（path(i) が連続）
// と column-major（spots_at(k) が連続 = 回帰・分位の走査順）の 2 レイアウトで持つ（メモリ 2 倍、いずれも
// n_paths (n_steps+1) 個の double。N = 20000, M = 50 で計 16 MB）。
//
// 後ろ向き回帰（時点 k = n_steps−1, …, 0）: ITM（本源的価値 h(S_k) > 0）のパスだけを使い、
// 「将来キャッシュフローを k まで割引いた y_i = cash_i e^{−r dt (ex_i − k)}」を基底 φ_j(x_i)（x = S/K）に
// 最小二乗回帰する。正規方程式 (ΦᵀΦ) c = Φᵀy を math/linsolve.hpp（部分ピボット LU、rel_tol =
// kPivotTol）で解き、フィット c(S) = Σ c_j φ_j(S/K) を継続価値の推定として h(S_k) > c(S_k) のパスを k で
// 行使する（cash_i = h, ex_i = k）。正規方程式が特異（σ = 0 で全パスの S が一致、ITM パスが基底数未満、
// 基底が狭い ITM 区間でほぼ線形従属、等）なら基底を末尾から 1 つずつ落として解き直す（ランク打ち切り:
// ΦᵀΦ の先頭ブロックは先頭基底の正規行列そのもの）。rank 1 は ITM パスの y の平均（定数）で必ず解ける
// （last_fit_rank() / last_fit_fallback()）。ITM パスが 0 ならフィットなし（continuation_value は NaN）で誰も
// 行使しない。最後の時点 k = 0 では全パスの S = S0 なので、h(S0) > 0 なら rank 1 になり「即時行使 h(S0) と
// 割引継続価値の平均の大きい方」という t = 0 の判定になる。ATM（h(S0) = 0）なら ITM パスが無く行使も無い。
//
// 条件数の実測（put, x ∈ (0.3, 1), N = 20000, seed 777）: 正規行列の LU ピボット比（|p_k| / max|a_ij|）は
// Laguerre で 1, 1e-3, 5e-6, 1e-9, 5e-14（5 番目が kPivotTol を割る。long double でも同じ値なので条件数の
// 実体であり、5 番目の double 値は 25 % ずれる = 丸めが効き始める位置）。狭い区間上の滑らかな関数は基底を
// 1 つ増やすごとに条件数が ~1e3〜1e4 悪化し、正規方程式はそれを 2 乗で受けるので double では基底 4〜5 個が
// 実用上限。打ち切りは「数値的に意味のある最大の基底数」で回帰することに相当し、価格は基底 3 個以上で
// ほぼ変わらない（Laguerre 3/4/5/8: 6.068 / 6.072 / 6.076 / 6.075、LSM-04。QR による最小二乗との照合でも
// 打ち切りは価格を偏らせない）。Power（x^j）は条件がやや良く 5 個までほぼ全時点でフルランク（50 時点中
// 1 つが 4）、8 個では 4〜8 が混ざる（価格 6.069）。基底を少なく保つ本当の理由は係数の打ち消し: Laguerre 3 でも
// 係数は O(700)（例: [696, −1138, 648]）で、x ∈ (0.3, 1) 上でそれが打ち消し合って O(1) の継続価値になる。
// 桁落ちを避けたければ正規方程式ではなく QR（Givens の逐次更新）にする — 将来の候補で、ここでは行わない。
//
// 基底（x = S/K に正規化 — S ≈ 100 で x^7 ≈ 1e14 になる悪条件を避ける。フィット値は正規化前と同じ関数空間）:
//   Power    : φ_0 = 1, φ_j = x^j
//   Laguerre : φ_0 = 1, φ_j = e^{−x/2} L_{j−1}(x)（Longstaff–Schwartz の「定数 + 重み付き Laguerre」）
//              L_0 = 1, L_1 = 1 − x, L_{n+1} = ((2n+1−x) L_n − n L_{n−1}) / (n+1)
// n_basis は「定数を含む回帰子の総数」（1..kMaxBasis = 8。1 = 定数だけの回帰は rank 1 フォールバックと同じ
// 意味を持つ正当な設定なので許す）。
//
// 文献: Longstaff, F. A. & Schwartz, E. S. (2001), "Valuing American Options by Simulation: A Simple
// Least-Squares Approach", Review of Financial Studies 14(1), 113–147.
//
// 結果 result(): price = 全パスの割引キャッシュフロー cash_i e^{−r dt ex_i} の平均。std_error はアンチセティック
// 対の対平均 z_j = (v_{2j} + v_{2j+1}) / 2 を標本単位とした sd(z) / √n_pairs（対内の負相関を織り込む。パスを
// 単位にすると SE を過小評価する）。これは「方策を固定したときの」標本誤差で、回帰方策そのものの推定誤差は
// 含まない（Longstaff–Schwartz 2001 と同じ扱い。実測: N = 20000, M = 50 の 400 seed で価格の sd は報告 SE の
// 1.01 倍（Laguerre 3）/ 1.03 倍（Laguerre 4）で、実用上無視できる）。ATM put（K = 100）の CN 参照 6.0848 との
// 差は Laguerre 3 で −0.026、Laguerre 4 で −0.005（400 seed 平均）。内訳は 50 行使日の Bermudan と連続 American
// の差 −0.012（CRR 木）と、方策の損から in-sample 評価（同じパスで方策を推定して評価する）の上方バイアスを
// 引いた残差。european_price は同じパスの割引満期ペイオフの平均、exercised_paths は満期より前に行使したパス数。
// step_backward を 1 回も呼んでいなければ price == european_price。
//
// 確保は init だけ（パス 2 面、割引表、cash / 行使時点、回帰の作業配列、分位用の作業配列。std::vector::resize
// なので失敗は std::bad_alloc — init は投げうる唯一の入口）。step_backward / result / continuation_value /
// quantile_at は確保なし・例外なし。quantile_at は const だが mutable な作業配列に列をコピーして nth_element
// するので再入不可（Model は単一の計算スレッドが持つ）。不正入力は init で決定的に丸める（s0 > 0, K > 0,
// T ≥ 0, σ ≥ 0, r 有限、n_paths ∈ [2, kMaxPaths] の偶数、n_steps ∈ [1, kMaxSteps]、n_basis ∈ [1, kMaxBasis]、
// NaN は既定値へ）。上限は確保量 2·N·(M+1)·8 B を有界にするためのもの（両方最大で 3.3 GB、シーンの既定は
// N = 20000, M = 50 の 16 MB）。

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
#include "quantviz/core/rng.hpp"

namespace quantviz::core {

enum class LsmBasis : std::uint8_t { Power, Laguerre };

struct LsmParams {
    double        s0, K, T, r, sigma;
    OptionType    type;
    std::size_t   n_paths;  ///< 偶数（奇数は init が +1 に丸める）
    std::size_t   n_steps;  ///< 行使時点数 M（dt = T / M）
    std::size_t   n_basis;  ///< 定数を含む回帰子の数 1..Lsm::kMaxBasis
    LsmBasis      basis = LsmBasis::Laguerre;
    std::uint64_t seed;
};

struct LsmResult {
    double      price;            ///< 割引キャッシュフローの平均（全時点処理後が LSM 価格）
    double      std_error;        ///< 対平均を標本単位とした標準誤差
    double      european_price;   ///< 同パスの European MC
    std::size_t exercised_paths;  ///< 満期より前に行使したパス数
};

class Lsm {
public:
    static constexpr std::size_t kMaxBasis = 8;       ///< 定数を含む回帰子の上限（1 は定数だけ = rank 1 と同義）
    static constexpr std::size_t kMaxPaths = 200000;  ///< init の確保量を有界にする（偶数）
    static constexpr std::size_t kMaxSteps = 1024;
    /// 正規方程式の特異判定（linsolve の rel_tol）。σ = 0 の n φφᵀ で残る丸めピボット（~1e-16 · scale）を
    /// 弾き、健全な回帰（Laguerre 3 の条件数 ~1e4、Power 5 でも ~1e10）は通す。
    static constexpr double kPivotTol = 1e-13;

    Lsm() = default;

    /// パラメータを丸めてパスを生成し、満期 t = T（current_step() == n_steps）に置く。ヒープ確保はここだけで、
    /// 失敗すれば std::bad_alloc を投げる（投げうる唯一のメンバ。step_backward / result / quantile_at は投げない）。
    /// それ以前に path() / spots_at() / continuation_coeffs() が返した span は無効になる。
    void init(LsmParams p) {
        params_ = sanitize(p);
        const std::size_t n = params_.n_paths, m = params_.n_steps, nb = params_.n_basis;

        paths_.resize(n * (m + 1));
        spots_.resize((m + 1) * n);
        disc_.resize(m + 1);
        cash_.resize(n);
        ex_step_.resize(n);
        ata_.resize(nb * nb);
        lu_.resize(nb * nb);
        atb_.resize(nb);
        coeffs_.resize(nb);
        phi_.resize(nb);
        piv_.resize(nb);
        scratch_.resize(n);

        generate_paths(params_, paths_, {});
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k <= m; ++k) spots_[k * n + i] = paths_[i * (m + 1) + k];

        const double dt = params_.T / static_cast<double>(m);
        for (std::size_t k = 0; k <= m; ++k) disc_[k] = std::exp(-params_.r * dt * static_cast<double>(k));
        for (std::size_t i = 0; i < n; ++i) {
            cash_[i]    = intrinsic(spots_[m * n + i]);
            ex_step_[i] = m;
        }
        std::fill(coeffs_.begin(), coeffs_.end(), 0.0);
        current_step_ = m;
        fit_valid_    = false;
        fit_fallback_ = false;
        fit_rank_     = 0;
        last_itm_     = 0;
    }

    /// 1 行使時点分の回帰 + 行使判定（時点 current_step() − 1 を処理）。t = 0（remaining() == 0）なら
    /// 何もせず false。
    bool step_backward() noexcept {
        if (current_step_ == 0 || paths_.empty()) return false;
        const std::size_t k = current_step_ - 1;
        regress_and_exercise(k);
        current_step_ = k;
        return true;
    }

    [[nodiscard]] std::size_t remaining() const noexcept { return current_step_; }
    [[nodiscard]] std::size_t current_step() const noexcept { return current_step_; }
    /// 現在時刻 t = T · current_step / n_steps。
    [[nodiscard]] double time() const noexcept {
        if (paths_.empty()) return 0.0;
        return params_.T * static_cast<double>(current_step_) / static_cast<double>(params_.n_steps);
    }
    [[nodiscard]] const LsmParams& params() const noexcept { return params_; }

    /// 現時点までの部分結果（全時点処理後が最終価格）。
    [[nodiscard]] LsmResult result() const noexcept {
        LsmResult r{0.0, 0.0, 0.0, 0};
        if (paths_.empty()) return r;
        const std::size_t n = params_.n_paths, m = params_.n_steps;
        const double*     terminal = &spots_[m * n];
        const double      disc_t   = disc_[m];
        double            se_eu    = 0.0;
        mean_and_se([&](std::size_t i) noexcept { return cash_[i] * disc_[ex_step_[i]]; }, r.price,
                    r.std_error);
        mean_and_se([&](std::size_t i) noexcept { return intrinsic(terminal[i]) * disc_t; }, r.european_price,
                    se_eu);
        for (std::size_t i = 0; i < n; ++i)
            if (ex_step_[i] < m) ++r.exercised_paths;
        return r;
    }

    /// i 番目のパス（n_steps + 1 点）。範囲外・未初期化なら空。
    [[nodiscard]] std::span<const double> path(std::size_t i) const noexcept {
        if (paths_.empty() || i >= params_.n_paths) return {};
        const std::size_t len = params_.n_steps + 1;
        return {paths_.data() + i * len, len};
    }
    /// 時点 step の全パスの S（列、n_paths 個）。範囲外・未初期化なら空。
    [[nodiscard]] std::span<const double> spots_at(std::size_t step) const noexcept {
        if (paths_.empty() || step > params_.n_steps) return {};
        return {spots_.data() + step * params_.n_paths, params_.n_paths};
    }
    /// パス i の行使時点。満期より前に行使していなければ n_steps（範囲外・未初期化も n_steps）。
    [[nodiscard]] std::size_t exercise_step(std::size_t i) const noexcept {
        if (paths_.empty() || i >= params_.n_paths) return params_.n_steps;
        return ex_step_[i];
    }
    /// 直近の時点の回帰係数（n_basis 個、x = S/K の基底に対する）。フォールバック時は [平均, 0, …]。
    [[nodiscard]] std::span<const double> continuation_coeffs() const noexcept { return coeffs_; }
    /// 直近の時点のフィット c(s) = Σ c_j φ_j(s/K)。フィットがまだ無い（満期、または ITM パス 0）、
    /// s が負・非有限なら NaN。
    [[nodiscard]] double continuation_value(double s) const noexcept {
        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        if (!fit_valid_ || !(s >= 0.0) || !std::isfinite(s)) return kNaN;
        std::array<double, kMaxBasis> phi{};
        basis_row(s / params_.K, phi);
        double c = 0.0;
        for (std::size_t j = 0; j < params_.n_basis; ++j) c += coeffs_[j] * phi[j];
        return c;
    }
    /// 直近の回帰に使った ITM パス数。
    [[nodiscard]] std::size_t last_itm_paths() const noexcept { return last_itm_; }
    /// 直近の回帰で実際に使った基底の数（先頭から）。n_basis 未満ならランク打ち切りが起きた。フィットなしは 0。
    [[nodiscard]] std::size_t last_fit_rank() const noexcept { return fit_rank_; }
    /// 直近の回帰が特異でランクを落としたか（rank 1 = 定数 = ITM パスの y の平均）。
    [[nodiscard]] bool last_fit_fallback() const noexcept { return fit_fallback_; }

    /// 時点 step の S の分位（最近接順位: index = round(q (N − 1))）。描画用。step が範囲外、q が [0, 1] 外
    /// または NaN、未初期化なら NaN。作業配列に列をコピーして nth_element する（O(N)、確保なし、再入不可）。
    [[nodiscard]] double quantile_at(std::size_t step, double q) const noexcept {
        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        if (paths_.empty() || step > params_.n_steps || !(q >= 0.0) || !(q <= 1.0)) return kNaN;
        const std::size_t n   = params_.n_paths;
        const double*     col = &spots_[step * n];
        std::copy(col, col + n, scratch_.begin());
        const std::size_t idx =
            std::min(static_cast<std::size_t>(q * static_cast<double>(n - 1) + 0.5), n - 1);
        std::nth_element(scratch_.begin(), scratch_.begin() + static_cast<std::ptrdiff_t>(idx),
                         scratch_.end());
        return scratch_[idx];
    }

    // -----------------------------------------------------------------------
    // パス生成（init が使う。LSM-01 が単体で検査する）
    // -----------------------------------------------------------------------

    /// GBM の 1 ステップ: s · exp(drift + vol · z)。generate_paths とテストが同じ式を共有してビット一致を保つ。
    [[nodiscard]] static double gbm_step(double s, double drift, double vol, double z) noexcept {
        return s * std::exp(drift + vol * z);
    }

    /// p のパスを paths（n_paths × (n_steps+1) row-major）に生成する。対 j のステップ k で Z を 1 本引き、
    /// パス 2j に +Z、パス 2j+1 に −Z を与える。normals が空でなければ Z を normals[j n_steps + k] に書く。
    /// n_paths が奇数、または span が足りなければ何も書かずに false。p は丸めない（init 済みの値を渡す）。
    static bool generate_paths(const LsmParams& p, std::span<double> paths,
                               std::span<double> normals) noexcept {
        const std::size_t n = p.n_paths, m = p.n_steps;
        if (n % 2 != 0 || m == 0 || paths.size() < n * (m + 1)) return false;
        if (!normals.empty() && normals.size() < (n / 2) * m) return false;

        const double dt    = p.T / static_cast<double>(m);
        const double drift = (p.r - 0.5 * p.sigma * p.sigma) * dt;
        const double vol   = p.sigma * std::sqrt(dt);
        Rng          rng(p.seed);
        for (std::size_t j = 0; j < n / 2; ++j) {
            double* a = &paths[(2 * j) * (m + 1)];
            double* b = &paths[(2 * j + 1) * (m + 1)];
            a[0]      = p.s0;
            b[0]      = p.s0;
            for (std::size_t k = 0; k < m; ++k) {
                const double z = rng.normal();
                if (!normals.empty()) normals[j * m + k] = z;
                a[k + 1] = gbm_step(a[k], drift, vol, z);
                b[k + 1] = gbm_step(b[k], drift, vol, -z);
            }
        }
        return true;
    }

private:
    static LsmParams sanitize(LsmParams p) noexcept {
        if (!(p.K > 0.0) || !std::isfinite(p.K)) p.K = 1.0;
        if (!(p.s0 > 0.0) || !std::isfinite(p.s0)) p.s0 = p.K;
        if (!(p.T >= 0.0) || !std::isfinite(p.T)) p.T = 0.0;
        if (!(p.sigma >= 0.0) || !std::isfinite(p.sigma)) p.sigma = 0.0;
        if (!std::isfinite(p.r)) p.r = 0.0;
        p.n_paths = std::clamp<std::size_t>(p.n_paths, 2, kMaxPaths);
        if (p.n_paths % 2 != 0) ++p.n_paths;  // kMaxPaths は偶数なので上限を超えない
        p.n_steps = std::clamp<std::size_t>(p.n_steps, 1, kMaxSteps);
        p.n_basis = std::clamp<std::size_t>(p.n_basis, 1, kMaxBasis);
        return p;
    }

    [[nodiscard]] double intrinsic(double s) const noexcept {
        return params_.type == OptionType::Call ? std::max(s - params_.K, 0.0) : std::max(params_.K - s, 0.0);
    }

    /// 基底 φ_0..φ_{n_basis−1} を x = S/K で評価して out に書く。
    void basis_row(double x, std::span<double> out) const noexcept {
        const std::size_t nb = params_.n_basis;
        out[0]               = 1.0;
        if (params_.basis == LsmBasis::Power) {
            double pw = 1.0;
            for (std::size_t j = 1; j < nb; ++j) {
                pw *= x;
                out[j] = pw;
            }
        } else {
            // φ_j = e^{−x/2} L_{j−1}(x)、三項漸化 L_{n+1} = ((2n+1−x) L_n − n L_{n−1}) / (n+1)
            const double w      = std::exp(-0.5 * x);
            double       l_prev = 0.0, l_cur = 1.0;  // L_{-1}（未使用）, L_0
            for (std::size_t j = 1; j < nb; ++j) {
                out[j]           = w * l_cur;
                const double nn  = static_cast<double>(j - 1);
                const double l_n = ((2.0 * nn + 1.0 - x) * l_cur - nn * l_prev) / (nn + 1.0);
                l_prev           = l_cur;
                l_cur            = l_n;
            }
        }
    }

    /// 時点 k の後ろ向き回帰と行使判定。ITM パスの y_i = cash_i e^{−r dt (ex_i − k)} を φ(x_i) に回帰し、
    /// h(S_k) > c(S_k) のパスを k で行使する。回帰の走査順・加算順は固定（決定性）。
    void regress_and_exercise(std::size_t k) noexcept {
        const std::size_t n = params_.n_paths, nb = params_.n_basis;
        const double*     col   = &spots_[k * n];

        std::fill(ata_.begin(), ata_.end(), 0.0);
        std::fill(atb_.begin(), atb_.end(), 0.0);
        std::size_t n_itm = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double h = intrinsic(col[i]);
            if (!(h > 0.0)) continue;
            const double y = cash_[i] * disc_[ex_step_[i] - k];
            basis_row(col[i] / params_.K, phi_);
            for (std::size_t a = 0; a < nb; ++a) {
                atb_[a] += phi_[a] * y;
                for (std::size_t b = a; b < nb; ++b) ata_[a * nb + b] += phi_[a] * phi_[b];
            }
            ++n_itm;
        }
        last_itm_ = n_itm;
        std::fill(coeffs_.begin(), coeffs_.end(), 0.0);
        if (n_itm == 0) {  // 誰も ITM でない: フィットなし、行使なし
            fit_valid_    = false;
            fit_fallback_ = false;
            fit_rank_     = 0;
            return;
        }
        for (std::size_t a = 0; a < nb; ++a)
            for (std::size_t b = 0; b < a; ++b) ata_[a * nb + b] = ata_[b * nb + a];  // 対称に埋める

        // ランク打ち切り: ΦᵀΦ の先頭 rank×rank ブロックは先頭 rank 個の基底の正規行列そのものなので、
        // 特異なら基底を 1 つずつ落として解き直す。rank = 1 は定数フィット（ITM パスの y の平均）で、
        // n_itm > 0 なら必ず解ける。
        std::size_t rank = nb;
        bool        ok   = false;
        while (rank >= 1 && !ok) {
            for (std::size_t a = 0; a < rank; ++a)
                for (std::size_t b = 0; b < rank; ++b) lu_[a * rank + b] = ata_[a * nb + b];
            std::fill(coeffs_.begin(), coeffs_.end(), 0.0);
            std::copy(atb_.begin(), atb_.begin() + static_cast<std::ptrdiff_t>(rank), coeffs_.begin());
            ok = linsolve(rank, lu_, coeffs_, piv_, kPivotTol);
            for (std::size_t j = 0; ok && j < rank; ++j)
                if (!std::isfinite(coeffs_[j])) ok = false;
            if (!ok) --rank;
        }
        if (!ok) {  // Σy が非有限（パスが overflow）: フィットなし
            std::fill(coeffs_.begin(), coeffs_.end(), 0.0);
            fit_valid_    = false;
            fit_fallback_ = true;
            fit_rank_     = 0;
            return;
        }
        fit_valid_    = true;
        fit_rank_     = rank;
        fit_fallback_ = rank < nb;

        for (std::size_t i = 0; i < n; ++i) {
            const double h = intrinsic(col[i]);
            if (!(h > 0.0)) continue;
            basis_row(col[i] / params_.K, phi_);
            double c = 0.0;
            for (std::size_t j = 0; j < nb; ++j) c += coeffs_[j] * phi_[j];
            if (h > c) {  // c が NaN なら偽 → 行使しない
                cash_[i]    = h;
                ex_step_[i] = k;
            }
        }
    }

    /// 対平均 z_j = (v_{2j} + v_{2j+1}) / 2 を標本単位として平均と標準誤差（2 パス）。
    template <class F>
    void mean_and_se(F&& value, double& mean, double& se) const noexcept {
        const std::size_t n_pairs = params_.n_paths / 2;
        double            sum     = 0.0;
        for (std::size_t j = 0; j < n_pairs; ++j) sum += 0.5 * (value(2 * j) + value(2 * j + 1));
        mean = sum / static_cast<double>(n_pairs);
        if (n_pairs < 2) {
            se = 0.0;
            return;
        }
        double ss = 0.0;
        for (std::size_t j = 0; j < n_pairs; ++j) {
            const double d = 0.5 * (value(2 * j) + value(2 * j + 1)) - mean;
            ss += d * d;
        }
        se = std::sqrt(ss / static_cast<double>(n_pairs - 1) / static_cast<double>(n_pairs));
    }

    LsmParams params_{0.0, 0.0, 0.0, 0.0, 0.0, OptionType::Put, 0, 0, 0, LsmBasis::Laguerre, 0};

    std::vector<double>         paths_;    ///< row-major: path(i) が連続
    std::vector<double>         spots_;    ///< column-major: spots_at(k) が連続
    std::vector<double>         disc_;     ///< disc_[k] = e^{−r dt k}
    std::vector<double>         cash_;     ///< パス i のキャッシュフロー（行使時点 ex_step_[i] の本源的価値）
    std::vector<std::size_t>    ex_step_;  ///< パス i の行使時点（未行使は n_steps）
    std::vector<double>         ata_, lu_, atb_, coeffs_, phi_;  ///< 正規方程式（ata_ は対称、lu_ は解く用の写し）
    std::vector<std::size_t>    piv_;
    mutable std::vector<double> scratch_;  ///< quantile_at の作業配列

    std::size_t current_step_ = 0;
    bool        fit_valid_    = false;
    bool        fit_fallback_ = false;
    std::size_t fit_rank_     = 0;
    std::size_t last_itm_     = 0;
};

}  // namespace quantviz::core
