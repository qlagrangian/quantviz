#pragma once
// core/math/mat.hpp — 固定サイズの密行列（row-major, std::array ベース、ヒープ確保なし）。
// カルマンフィルタのような小さな線形代数（NX, NZ ≤ 3 程度）をホットパスで扱うための型で、
// 集成体（aggregate）のまま保つことで trivially copyable / default constructible を満たし、
// Snapshot に直接埋め込める。例外は投げない: 逆行列の失敗は std::optional で表現する
// （core は「例外なし・確保なし」が契約。CLAUDE.md「Hot paths」参照）。
//
// 許容誤差の方針: `inverse` / `is_symmetric` / `is_psd` の `tol` は**相対**量で、
// それぞれの式に現れる項の絶対値の和（＝打ち消し合う前の自然なスケール）に掛けて使う。
// 絶対量にすると、共分散のスケールが変わっただけで判定が変わってしまうため。
// どの関数も `tol = 0` は「丸め誤差を一切許さない厳密判定」を意味する。

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <type_traits>

namespace quantviz::core {

/// R 行 C 列の行列。要素は row-major（`a[i * C + j]`）。
/// 集成体なので `Mat<2,2> m{{1, 2, 3, 4}}` と初期化でき、既定値は全要素 0。
template <std::size_t R, std::size_t C>
struct Mat {
    static_assert(R > 0 && C > 0, "Mat requires positive dimensions");

    static constexpr std::size_t kRows = R;
    static constexpr std::size_t kCols = C;

    std::array<double, R * C> a{};

    double& operator()(std::size_t i, std::size_t j) noexcept { return a[i * C + j]; }
    double  operator()(std::size_t i, std::size_t j) const noexcept { return a[i * C + j]; }

    /// 単位行列（正方のときだけ実体化できる）。
    static Mat identity() noexcept {
        static_assert(R == C, "identity() requires a square matrix");
        Mat m{};
        for (std::size_t i = 0; i < R; ++i) m(i, i) = 1.0;
        return m;
    }

    /// 対角行列（正方のときだけ）。
    static Mat diagonal(double v) noexcept {
        static_assert(R == C, "diagonal() requires a square matrix");
        Mat m{};
        for (std::size_t i = 0; i < R; ++i) m(i, i) = v;
        return m;
    }

    [[nodiscard]] Mat<C, R> transpose() const noexcept {
        Mat<C, R> t{};
        for (std::size_t i = 0; i < R; ++i)
            for (std::size_t j = 0; j < C; ++j) t(j, i) = (*this)(i, j);
        return t;
    }

    /// 逆行列の閉形式（1×1 / 2×2 / 3×3 のみ。それ以外は static_assert でコンパイルエラー）。
    ///
    /// 失敗の通知方法: 例外を投げず `std::nullopt` を返す。特異とみなすのは
    ///   * 行列式が非有限、または
    ///   * `|det| <= tol · scale`、ここで scale は行列式を作る各項の絶対値の和
    ///     （1×1: |m00|、2×2: |ad| + |bc|、3×3: Σ|m(0,k)·cofactor_k|）、または
    ///   * `1/det` が非有限（det が非正規化数のとき逆数が overflow して全要素が inf になる）
    /// のいずれか。scale を掛けるので tol は相対量であり、`tol = 0`（既定）のときは
    /// 「det が厳密に 0 か非有限のときだけ失敗」という最も厳しい判定になる。
    /// 呼び出し側は `if (auto inv = m.inverse())` で判定する。
    [[nodiscard]] std::optional<Mat> inverse(double tol = 0.0) const noexcept {
        static_assert(R == C, "inverse() requires a square matrix");
        static_assert(R <= 3, "inverse() has a closed form only for 1x1, 2x2 and 3x3 matrices");

        const Mat& m = *this;
        Mat        out{};

        if constexpr (R == 1) {
            const double det = m(0, 0);
            if (!nonsingular(det, std::abs(det), tol)) return std::nullopt;
            const double inv_det = 1.0 / det;
            if (!std::isfinite(inv_det)) return std::nullopt;
            out(0, 0) = inv_det;
        } else if constexpr (R == 2) {
            const double ad  = m(0, 0) * m(1, 1);
            const double bc  = m(0, 1) * m(1, 0);
            const double det = ad - bc;
            if (!nonsingular(det, std::abs(ad) + std::abs(bc), tol)) return std::nullopt;
            const double inv_det = 1.0 / det;
            if (!std::isfinite(inv_det)) return std::nullopt;
            out(0, 0) = m(1, 1) * inv_det;
            out(0, 1) = -m(0, 1) * inv_det;
            out(1, 0) = -m(1, 0) * inv_det;
            out(1, 1) = m(0, 0) * inv_det;
        } else {
            // 余因子行列（adjugate）/ 行列式
            const double c00 = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);
            const double c01 = m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2);
            const double c02 = m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0);
            const double t0  = m(0, 0) * c00;
            const double t1  = m(0, 1) * c01;
            const double t2  = m(0, 2) * c02;
            const double det = t0 + t1 + t2;
            if (!nonsingular(det, std::abs(t0) + std::abs(t1) + std::abs(t2), tol)) return std::nullopt;
            const double inv_det = 1.0 / det;
            if (!std::isfinite(inv_det)) return std::nullopt;
            out(0, 0) = c00 * inv_det;
            out(1, 0) = c01 * inv_det;
            out(2, 0) = c02 * inv_det;
            out(0, 1) = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * inv_det;
            out(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * inv_det;
            out(2, 1) = (m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1)) * inv_det;
            out(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * inv_det;
            out(1, 2) = (m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2)) * inv_det;
            out(2, 2) = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) * inv_det;
        }
        return out;
    }

private:
    /// `det` が有限かつ `|det| > tol · scale` なら非特異。NaN が入ると必ず false になるよう
    /// 否定形（`!(... > ...)` ではなく肯定形を返す）で書く。
    static bool nonsingular(double det, double scale, double tol) noexcept {
        return std::isfinite(det) && std::abs(det) > tol * scale;
    }
};

// ---------------------------------------------------------------------------
// 算術
// ---------------------------------------------------------------------------

template <std::size_t R, std::size_t C>
Mat<R, C> operator+(const Mat<R, C>& lhs, const Mat<R, C>& rhs) noexcept {
    Mat<R, C> out{};
    for (std::size_t i = 0; i < R * C; ++i) out.a[i] = lhs.a[i] + rhs.a[i];
    return out;
}

template <std::size_t R, std::size_t C>
Mat<R, C> operator-(const Mat<R, C>& lhs, const Mat<R, C>& rhs) noexcept {
    Mat<R, C> out{};
    for (std::size_t i = 0; i < R * C; ++i) out.a[i] = lhs.a[i] - rhs.a[i];
    return out;
}

template <std::size_t R, std::size_t C>
Mat<R, C> operator-(const Mat<R, C>& m) noexcept {
    Mat<R, C> out{};
    for (std::size_t i = 0; i < R * C; ++i) out.a[i] = -m.a[i];
    return out;
}

template <std::size_t R, std::size_t C>
Mat<R, C> operator*(const Mat<R, C>& m, double s) noexcept {
    Mat<R, C> out{};
    for (std::size_t i = 0; i < R * C; ++i) out.a[i] = m.a[i] * s;
    return out;
}

template <std::size_t R, std::size_t C>
Mat<R, C> operator*(double s, const Mat<R, C>& m) noexcept {
    return m * s;
}

/// 行列積 (R×K) · (K×C)。
template <std::size_t R, std::size_t K, std::size_t C>
Mat<R, C> operator*(const Mat<R, K>& lhs, const Mat<K, C>& rhs) noexcept {
    Mat<R, C> out{};
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t k = 0; k < K; ++k) {
            const double v = lhs(i, k);
            for (std::size_t j = 0; j < C; ++j) out(i, j) += v * rhs(k, j);
        }
    return out;
}

// ---------------------------------------------------------------------------
// 判定
//
// 判定はすべて「合格条件を肯定形で書いて否定する」（`if (!(x >= y)) return false;`）。
// NaN はどの比較でも false になるので、この形なら NaN は必ず不合格になる。
// `x < y` と書くと NaN が合格してしまい、overflow で NaN になった小行列式を
// PSD と誤判定する（例: {{1e200, −1e300}, {−1e300, 1e200}}）。
// ---------------------------------------------------------------------------

/// 対称性（相対判定）: 全 (i,j) で |m(i,j) − m(j,i)| ≤ tol · (|m(i,j)| + |m(j,i)|)。
/// 非有限要素があれば false。tol = 0 は bit 一致の厳密対称を要求する。
template <std::size_t N>
[[nodiscard]] bool is_symmetric(const Mat<N, N>& m, double tol) noexcept {
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i; j < N; ++j) {
            if (!std::isfinite(m(i, j)) || !std::isfinite(m(j, i))) return false;
            const double scale = std::abs(m(i, j)) + std::abs(m(j, i));
            if (!(std::abs(m(i, j) - m(j, i)) <= tol * scale)) return false;
        }
    return true;
}

/// 半正定値性（N ≤ 3）: 対称かつ**全ての**主小行列式 ≥ −tol · (その小行列式のスケール)。
///
/// N ≤ 3 では主小行列式は最大 7 個（1 次 3 個・2 次 3 個・3 次 1 個）なので閉形式で全部数え
/// られる。首座小行列式だけでは PSD の十分条件にならない（例: diag(0, −1)）ため、全ての
/// 主小行列式を見るのが正しい判定（Sylvester の判定法）。
///
/// スケールの取り方: k 次の小行列式は ‖m‖^k のオーダーなので、絶対 tol を全次数に使うと
/// 意味が変わってしまう。2 次・3 次は「打ち消し合う前の項の絶対値の和」を、1 次（対角成分）は
/// 打ち消す相手が無いので行列の最大絶対値 ‖m‖∞ をスケールに使う。tol は対称性判定にも同じ
/// 相対量として渡る。tol = 0 は厳密判定。
template <std::size_t N>
[[nodiscard]] bool is_psd(const Mat<N, N>& m, double tol) noexcept {
    static_assert(N <= 3, "is_psd() checks all principal minors in closed form only for N <= 3");
    if (!is_symmetric(m, tol)) return false;

    double nrm = 0.0;  // ‖m‖∞（要素の最大絶対値）。ここに来る時点で全要素は有限。
    for (std::size_t i = 0; i < N * N; ++i) nrm = std::max(nrm, std::abs(m.a[i]));

    for (std::size_t i = 0; i < N; ++i)
        if (!(m(i, i) >= -tol * nrm)) return false;  // 1 次の主小行列式

    if constexpr (N >= 2) {
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = i + 1; j < N; ++j) {
                const double diag  = m(i, i) * m(j, j);
                const double off   = m(i, j) * m(j, i);
                const double scale = std::abs(diag) + std::abs(off);
                if (!(diag - off >= -tol * scale)) return false;  // 2 次
            }
    }
    if constexpr (N == 3) {
        const double t0    = m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1));
        const double t1    = m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0));
        const double t2    = m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
        const double scale = std::abs(t0) + std::abs(t1) + std::abs(t2);
        if (!(t0 - t1 + t2 >= -tol * scale)) return false;  // 3 次（行列式）
    }
    return true;
}

// Snapshot に直接埋め込めること（POD である）を型で固定する。集成体のまま保つ限り成り立つ。
static_assert(std::is_trivially_copyable_v<Mat<2, 2>>, "Mat must stay trivially copyable");
static_assert(std::is_default_constructible_v<Mat<2, 2>>, "Mat must stay default constructible");
static_assert(sizeof(Mat<2, 3>) == 6 * sizeof(double), "Mat must not carry any padding or vtable");

}  // namespace quantviz::core
