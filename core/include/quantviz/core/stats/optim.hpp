#pragma once
// core/stats/optim.hpp — 小規模（N は数個）な無制約最小化器と制約変換。
//
// * nelder_mead: 導関数を使わない単体法（Lagarias et al. 1998: 反射 1, 拡大 2, 収縮 1/2, 縮小 1/2）。
// * bfgs:        中心差分の数値勾配 + Armijo 後退直線探索による準 Newton 法。
//                BFGS 方向が降下方向でない、または直線探索が失敗したときは H を単位行列に戻して
//                最急降下方向にフォールバックする。
// * 制約変換:    (0,1) ↔ R は sigmoid / logit、(0,∞) ↔ R は softplus / 逆 softplus。
//                |x| が大きくてもオーバーフローや NaN を出さない形に書き分けてある。
//
// 契約:
// * `OptimResult::path` は先頭が x0、以後「各反復後の best x」で、要素数 = iters + 1（≤ max_iter + 1）。
//   したがって path は空にならず、path.back() == x が常に成り立つ（尤度面上の軌跡描画が依存する）。
// * `converged` は max_iter に達する前に (tol_f, tol_x) を同時に満たしたときだけ true。
//   したがって converged ⇒ iters < max_iter。さらに converged ⇒ f が有限で、非有限な目的関数に対して
//   収束を報告することはない。f(x0) が有限でなければ何もせず iters = 0, converged = false,
//   x = x0, f = f(x0)（NaN / ±inf のまま）で返す。
// * `*_into` は呼び出し側の OptimResult に書き込む入口で、path は clear() して capacity を再利用する。
//   max_iter ≤ 1024 なら 2 回目以降の呼び出しでヒープ確保は起きない（計算スレッド向け）。
//   値を返す nelder_mead / bfgs はその薄いラッパ。
// * 目的関数 F は転送参照で受け取り、左辺値として何度も呼ぶ（ムーブしない）。例外を投げないことを
//   前提とし、内部でも例外は使わない。

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace quantviz::core {

template <std::size_t N>
struct OptimResult {
    std::array<double, N>              x{};
    double                             f         = 0.0;
    std::size_t                        iters     = 0;
    bool                               converged = false;
    std::vector<std::array<double, N>> path{};  // x0 と反復ごとの best x（尤度面上の軌跡描画用）
};

struct OptimOptions {
    std::size_t max_iter     = 1000;
    double      tol_f        = 1e-10;  // f の広がり（NM）/ 1 反復の f 減少量（BFGS）の絶対許容
    double      tol_x        = 1e-8;   // 単体の大きさ（NM）/ 1 反復の移動量（BFGS）の絶対許容（∞ノルム）
    double      initial_step = 0.1;    // 初期単体の一辺（NM）/ 最初の直線探索の歩幅（BFGS）
};

namespace detail {

template <std::size_t N>
using Vec = std::array<double, N>;

/// ∞ノルム。NaN を飲み込まない（std::max は NaN を捨てるので使わない）: 一つでも NaN なら NaN。
template <std::size_t N>
double inf_norm_diff(const Vec<N>& a, const Vec<N>& b) noexcept {
    double m = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double v = std::fabs(a[i] - b[i]);
        if (v > m || v != v) m = v;
    }
    return m;
}

template <std::size_t N>
double inf_norm(const Vec<N>& a) noexcept {
    double m = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double v = std::fabs(a[i]);
        if (v > m || v != v) m = v;
    }
    return m;
}

template <std::size_t N>
double dot(const Vec<N>& a, const Vec<N>& b) noexcept {
    double s = 0.0;
    for (std::size_t i = 0; i < N; ++i) s += a[i] * b[i];
    return s;
}

/// 中心差分勾配。h_i = kGradStep · max(1, |x_i|)。打ち切り誤差 O(h²) と丸め誤差 ε|f|/h の
/// 釣り合いから h ≈ ε^{1/3} ≈ 6e-6 を採る。分母は実際に動いた幅 (x+h) − (x−h) を使う。
inline constexpr double kGradStep = 6e-6;

template <std::size_t N, class F>
Vec<N> central_gradient(F& f, const Vec<N>& x) {
    Vec<N> g{};
    Vec<N> xp = x;
    for (std::size_t i = 0; i < N; ++i) {
        const double h  = kGradStep * std::max(1.0, std::fabs(x[i]));
        const double xh = x[i] + h;
        const double xl = x[i] - h;
        xp[i]           = xh;
        const double fh = f(xp);
        xp[i]           = xl;
        const double fl = f(xp);
        xp[i]           = x[i];
        g[i]            = (fh - fl) / (xh - xl);
    }
    return g;
}

/// path の予約数: x0 + 反復ぶん。max_iter が大きくても 1025 で頭打ち（以後は伸長に任せる）。
inline std::size_t path_reserve(std::size_t max_iter) noexcept {
    return std::min<std::size_t>(max_iter, 1024) + 1;
}

/// 結果を x0 で初期化する共通処理。path は clear して capacity を再利用する。
template <std::size_t N>
void seed_result(OptimResult<N>& res, const Vec<N>& x0, double f0, std::size_t max_iter) {
    res.x         = x0;
    res.f         = f0;
    res.iters     = 0;
    res.converged = false;
    res.path.clear();
    res.path.reserve(path_reserve(max_iter));
    res.path.push_back(x0);
}

}  // namespace detail

/// Nelder–Mead 単体法（結果を res に書き込む版）。
/// 収束判定は「頂点間の f の広がり ≤ tol_f かつ 単体の ∞ノルム直径 ≤ tol_x」。
template <std::size_t N, class F>
void nelder_mead_into(OptimResult<N>& res, F&& f, std::array<double, N> x0, OptimOptions o = {}) {
    static_assert(N >= 1);
    using Vec = detail::Vec<N>;

    detail::seed_result(res, x0, f(x0), o.max_iter);
    if (o.max_iter == 0 || !std::isfinite(res.f)) return;

    // 初期単体: x0 と x0 + initial_step · e_i
    std::array<Vec, N + 1>    v;
    std::array<double, N + 1> fv;
    v[0]  = x0;
    fv[0] = res.f;
    for (std::size_t i = 0; i < N; ++i) {
        v[i + 1] = x0;
        v[i + 1][i] += o.initial_step;
        fv[i + 1] = f(v[i + 1]);
    }

    // f 昇順に並べ替え（挿入ソート: 頂点は N+1 個しかない。安定なので同値は元の順を保つ）。
    // NaN は比較が偽なので前へ進まず、有限な best（v[0]）を追い越すことはない。
    const auto sort_simplex = [&]() noexcept {
        for (std::size_t i = 1; i <= N; ++i) {
            const Vec    xi = v[i];
            const double fi = fv[i];
            std::size_t  j  = i;
            while (j > 0 && fv[j - 1] > fi) {
                v[j]  = v[j - 1];
                fv[j] = fv[j - 1];
                --j;
            }
            v[j]  = xi;
            fv[j] = fi;
        }
    };
    sort_simplex();

    while (res.iters < o.max_iter) {
        // 収束判定（NaN な頂点があれば広がりは NaN になり、収束しない）
        double f_spread = 0.0, x_spread = 0.0;
        for (std::size_t i = 1; i <= N; ++i) {
            const double df = std::fabs(fv[i] - fv[0]);
            const double dx = detail::inf_norm_diff(v[i], v[0]);
            if (df > f_spread || df != df) f_spread = df;
            if (dx > x_spread || dx != dx) x_spread = dx;
        }
        if (f_spread <= o.tol_f && x_spread <= o.tol_x) {
            res.converged = true;
            break;
        }

        // 最悪点を除いた重心
        Vec c{};
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j) c[j] += v[i][j];
        for (std::size_t j = 0; j < N; ++j) c[j] /= static_cast<double>(N);

        Vec xr;
        for (std::size_t j = 0; j < N; ++j) xr[j] = c[j] + (c[j] - v[N][j]);
        const double fr = f(xr);

        if (fr < fv[0]) {
            // 拡大
            Vec xe;
            for (std::size_t j = 0; j < N; ++j) xe[j] = c[j] + 2.0 * (c[j] - v[N][j]);
            const double fe = f(xe);
            if (fe < fr) {
                v[N]  = xe;
                fv[N] = fe;
            } else {
                v[N]  = xr;
                fv[N] = fr;
            }
        } else if (fr < fv[N - 1]) {
            // 反射
            v[N]  = xr;
            fv[N] = fr;
        } else {
            bool shrink = false;
            if (fr < fv[N]) {
                // 外側収縮
                Vec xc;
                for (std::size_t j = 0; j < N; ++j) xc[j] = c[j] + 0.5 * (xr[j] - c[j]);
                const double fc = f(xc);
                if (fc <= fr) {
                    v[N]  = xc;
                    fv[N] = fc;
                } else {
                    shrink = true;
                }
            } else {
                // 内側収縮
                Vec xc;
                for (std::size_t j = 0; j < N; ++j) xc[j] = c[j] + 0.5 * (v[N][j] - c[j]);
                const double fc = f(xc);
                if (fc < fv[N]) {
                    v[N]  = xc;
                    fv[N] = fc;
                } else {
                    shrink = true;
                }
            }
            if (shrink) {
                for (std::size_t i = 1; i <= N; ++i) {
                    for (std::size_t j = 0; j < N; ++j) v[i][j] = v[0][j] + 0.5 * (v[i][j] - v[0][j]);
                    fv[i] = f(v[i]);
                }
            }
        }

        sort_simplex();
        ++res.iters;
        res.path.push_back(v[0]);
    }

    res.x         = v[0];
    res.f         = fv[0];
    res.converged = res.converged && std::isfinite(res.f);
}

/// Nelder–Mead 単体法（値を返す版）。
template <std::size_t N, class F>
OptimResult<N> nelder_mead(F&& f, std::array<double, N> x0, OptimOptions o = {}) {
    OptimResult<N> res;
    nelder_mead_into(res, f, x0, o);
    return res;
}

/// BFGS（逆ヘッセ近似 H を更新、結果を res に書き込む版）。勾配は中心差分、直線探索は Armijo 条件の後退法。
/// 収束判定は「直前の反復での f 減少量 ≤ tol_f かつ 移動量 ≤ tol_x」。
/// 最急降下方向でも減少が見つからなければ数値的な停留点とみなして converged = true で止める
/// （移動量 0・減少量 0 は許容を満たす。このとき f は有限）。勾配が非有限になったら converged = false で止める。
template <std::size_t N, class F>
void bfgs_into(OptimResult<N>& res, F&& f, std::array<double, N> x0, OptimOptions o = {}) {
    static_assert(N >= 1);
    using Vec = detail::Vec<N>;
    constexpr double kArmijo      = 1e-4;
    constexpr int    kMaxHalvings = 60;  // 2^-60 ≈ 1e-18 まで歩幅を縮める（ulp 未満になれば早く抜ける）

    detail::seed_result(res, x0, f(x0), o.max_iter);
    if (o.max_iter == 0 || !std::isfinite(res.f)) return;

    Vec    x  = x0;
    double fx = res.f;
    Vec    g  = detail::central_gradient(f, x);

    std::array<Vec, N> H{};
    const auto         reset_H = [&]() noexcept {
        for (std::size_t i = 0; i < N; ++i) {
            H[i].fill(0.0);
            H[i][i] = 1.0;
        }
    };
    reset_H();

    double last_df = std::numeric_limits<double>::infinity();
    double last_dx = std::numeric_limits<double>::infinity();

    while (res.iters < o.max_iter) {
        if (last_df <= o.tol_f && last_dx <= o.tol_x) {
            res.converged = true;
            break;
        }
        const double gnorm = detail::inf_norm(g);
        if (!std::isfinite(gnorm)) break;  // 勾配が非有限: 進めない（converged = false のまま）
        if (gnorm == 0.0) {                // 勾配ゼロ: 厳密な停留点
            res.converged = true;
            break;
        }

        // 探索方向 d = −H g。降下方向でなければ最急降下に戻す。
        Vec d{};
        for (std::size_t i = 0; i < N; ++i) d[i] = -detail::dot(H[i], g);
        double gd       = detail::dot(g, d);
        bool   steepest = false;
        if (!(gd < 0.0)) {
            reset_H();
            for (std::size_t i = 0; i < N; ++i) d[i] = -g[i];
            gd       = detail::dot(g, d);
            steepest = true;
        }

        // Armijo 後退直線探索。最初の反復だけ歩幅を initial_step に揃える（勾配のスケールが未知のため）。
        // x + α d が x と一致したら（歩幅が ulp 未満）それ以上縮めても同じ点なので打ち切る。
        Vec    xn{};
        double fn = 0.0;
        const auto line_search = [&]() {
            double alpha = (res.iters == 0) ? std::min(1.0, o.initial_step / detail::inf_norm(d)) : 1.0;
            for (int k = 0; k < kMaxHalvings; ++k) {
                bool moved = false;
                for (std::size_t i = 0; i < N; ++i) {
                    xn[i] = x[i] + alpha * d[i];
                    moved = moved || (xn[i] != x[i]);
                }
                if (!moved) return false;
                fn = f(xn);
                if (fn <= fx + kArmijo * alpha * gd) return true;
                alpha *= 0.5;
            }
            return false;
        };
        bool ok = line_search();
        if (!ok && !steepest) {
            reset_H();
            for (std::size_t i = 0; i < N; ++i) d[i] = -g[i];
            gd       = detail::dot(g, d);
            steepest = true;
            ok       = line_search();
        }
        if (!ok) {  // 最急降下でも減らない: 数値的な停留点
            res.converged = true;
            break;
        }

        // BFGS 更新: H ← (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ, ρ = 1/(sᵀy)
        const Vec gn = detail::central_gradient(f, xn);
        Vec       s{}, y{};
        for (std::size_t i = 0; i < N; ++i) {
            s[i] = xn[i] - x[i];
            y[i] = gn[i] - g[i];
        }
        const double sy = detail::dot(s, y);
        if (sy > 0.0 && std::isfinite(sy)) {  // 曲率条件
            if (res.iters == 0) {  // 初期スケーリング（Nocedal & Wright 6.20）
                const double yy = detail::dot(y, y);
                if (yy > 0.0 && std::isfinite(yy)) {
                    reset_H();
                    for (std::size_t i = 0; i < N; ++i) H[i][i] = sy / yy;
                }
            }
            Vec Hy{};
            for (std::size_t i = 0; i < N; ++i) Hy[i] = detail::dot(H[i], y);
            const double rho  = 1.0 / sy;
            const double coef = rho * (1.0 + rho * detail::dot(y, Hy));
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t j = 0; j < N; ++j)
                    H[i][j] += coef * s[i] * s[j] - rho * (Hy[i] * s[j] + s[i] * Hy[j]);
        }

        last_df = fx - fn;
        last_dx = detail::inf_norm(s);
        x       = xn;
        fx      = fn;
        g       = gn;
        ++res.iters;
        res.path.push_back(x);
    }

    res.x         = x;
    res.f         = fx;
    res.converged = res.converged && std::isfinite(res.f);
}

/// BFGS（値を返す版）。
template <std::size_t N, class F>
OptimResult<N> bfgs(F&& f, std::array<double, N> x0, OptimOptions o = {}) {
    OptimResult<N> res;
    bfgs_into(res, f, x0, o);
    return res;
}

// ---- 制約変換（往復が恒等） -------------------------------------------------------------

/// R → (0,1): sigmoid。x の符号で式を分けて exp のオーバーフローを避ける。
inline double to_unit(double x) noexcept {
    if (x >= 0.0) {
        const double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    }
    const double e = std::exp(x);
    return e / (1.0 + e);
}

/// (0,1) → R: logit。log1p で 1 − u の相殺を避ける。
inline double from_unit(double u) noexcept { return std::log(u) - std::log1p(-u); }

/// R → (0,∞): softplus log(1 + e^x)。x > 0 では x + log1p(e^{−x}) と書いてオーバーフローを避ける。
inline double to_positive(double x) noexcept {
    return x > 0.0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
}

/// (0,∞) → R: 逆 softplus log(e^p − 1)。p が大きいと e^p が溢れるので p + log(1 − e^{−p}) に書き換える。
inline double from_positive(double p) noexcept {
    return p > 1.0 ? p + std::log(-std::expm1(-p)) : std::log(std::expm1(p));
}

}  // namespace quantviz::core
