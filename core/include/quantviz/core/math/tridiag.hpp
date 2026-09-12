#pragma once
// core/math/tridiag.hpp — 三重対角系 A x = d の Thomas 法（ピボットなし Gauss 消去、O(n)）。
// Crank–Nicolson（pricing/fdm_cn.hpp）の各時間ステップで呼ばれるホットパスなので、
// 確保なし・例外なし・noexcept。作業配列は呼び手が渡す（長さ ≥ n）。
//
//   A = tridiag(a, b, c):  a: 下対角 (n−1), b: 対角 (n), c: 上対角 (n−1), d: 右辺 (n)
//
// 前進消去で c'_i = c_i / p_i, d'_i = (d_i − a_{i−1} d'_{i−1}) / p_i, p_i = b_i − a_{i−1} c'_{i−1}
// を作り、後退代入で x_i = d'_i − c'_i x_{i+1}。c' は work に、d' は x に置く。
// x は d と同じ領域でよい（in-place）: d_i を読んでから x_i を書く順序になっている。
//
// 戻り値: ピボット p_i が 0 または非有限（NaN / inf、あるいは 1/p_i が inf になる非正規化数）
// のとき false。長さが整合しない（a, c が n−1 でない、d / x が n でない、work が n 未満）ときも
// false で、その場合 x には触れない。n = 0 は自明に解けたとして true。
// 失敗時の x の内容は不定（前進消去の途中で止まる）。
// Thomas 法は対角優位な系で後退安定。FDM の CN 行列（1 + θΔt(σ²i² + r) が対角）は実用格子で
// 常にこの条件を満たす。

#include <cmath>
#include <cstddef>
#include <span>

namespace quantviz::core {

[[nodiscard]] inline bool tridiag_solve(std::span<const double> a, std::span<const double> b,
                                        std::span<const double> c, std::span<const double> d,
                                        std::span<double> x, std::span<double> work) noexcept {
    const std::size_t n = b.size();
    if (n == 0) return true;
    if (a.size() + 1 != n || c.size() + 1 != n || d.size() != n || x.size() != n || work.size() < n) return false;

    // ピボットは割る前に検査する（非有限・0 なら false）。非正規化数は 1/p が inf になるので逆数も検査。
    // NaN は比較が全て偽になるので肯定形で書く。
    const auto pivot_ok = [](double p) noexcept { return std::isfinite(p) && p != 0.0; };

    if (!pivot_ok(b[0])) return false;
    double inv = 1.0 / b[0];
    if (!std::isfinite(inv)) return false;
    x[0] = d[0] * inv;  // d[0] を読んでから x[0] を書く（in-place 安全）
    if (n > 1) work[0] = c[0] * inv;

    for (std::size_t i = 1; i < n; ++i) {
        const double p = b[i] - a[i - 1] * work[i - 1];
        if (!pivot_ok(p)) return false;
        inv = 1.0 / p;
        if (!std::isfinite(inv)) return false;
        x[i] = (d[i] - a[i - 1] * x[i - 1]) * inv;
        if (i + 1 < n) work[i] = c[i] * inv;
    }
    for (std::size_t i = n - 1; i-- > 0;) x[i] -= work[i] * x[i + 1];
    return true;
}

}  // namespace quantviz::core
