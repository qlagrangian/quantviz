#pragma once
// core/math/linsolve.hpp — 小さな密行列の連立一次方程式 A x = b（部分ピボット LU、n ≤ kLinsolveMaxN = 16）。
// Longstaff–Schwartz（pricing/lsm.hpp）の正規方程式 (ΦᵀΦ) c = Φᵀy を毎時点解くホットパス用で、
// 確保なし・例外なし・noexcept。作業配列（LU のピボット記録）は呼び手が渡す。
//
//   a: n×n row-major（a[i n + j]）。呼び出し後は LU 因子で上書きされる（U は対角以上、L の乗数は対角より下）。
//   b: 右辺 n。呼び出し後は解 x で上書きされる（ピボットに合わせて行交換済み）。
//   piv: n。piv[k] = 第 k 段で第 k 行と交換した行番号（交換なしなら k）。
//
// 各段で列 k の絶対値最大の行をピボットにとる（部分ピボット）。ピボット p_k が
//   * 非有限、または
//   * |p_k| ≤ rel_tol · scale（scale = 入力行列の max |a_ij|）、または
//   * 1/p_k が非有限（非正規化数）
// なら特異とみなして false を返す。rel_tol = 0（既定）は「厳密に 0 のピボットだけを特異とする」最も厳しい
// 判定（math/mat.hpp の inverse(tol) と同じ流儀）。丸めで 1e-16 程度のピボットが残る「ほぼランク落ち」
// （LSM で σ = 0 のとき全パスの S が一致し ΦᵀΦ = n φφᵀ になる場合）は rel_tol > 0 で弾く。入力に NaN / inf
// があれば scale が非有限になり必ず false。長さが足りない（a < n², b < n, piv < n）、n > kLinsolveMaxN、
// rel_tol が負または NaN のときも false で、その場合 a / b には触れない。n = 0 は自明に解けたとして true。
// 失敗時の a / b の内容は不定（消去の途中で止まる）。

#include <cmath>
#include <cstddef>
#include <span>
#include <utility>

namespace quantviz::core {

inline constexpr std::size_t kLinsolveMaxN = 16;

[[nodiscard]] inline bool linsolve(std::size_t n, std::span<double> a, std::span<double> b,
                                   std::span<std::size_t> piv, double rel_tol = 0.0) noexcept {
    if (n == 0) return true;
    if (n > kLinsolveMaxN || a.size() < n * n || b.size() < n || piv.size() < n) return false;
    if (!(rel_tol >= 0.0)) return false;  // NaN / 負

    // 特異判定のスケール。NaN は `!(v <= scale)` で伝播し、inf はそのまま残るので非有限入力はここで落ちる。
    double scale = 0.0;
    for (std::size_t i = 0; i < n * n; ++i) {
        const double v = std::abs(a[i]);
        if (!(v <= scale)) scale = v;
    }
    if (!std::isfinite(scale)) return false;
    const double thresh = rel_tol * scale;

    for (std::size_t k = 0; k < n; ++k) {
        // 列 k の絶対値最大の行をピボットに
        std::size_t p    = k;
        double      best = std::abs(a[k * n + k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            const double v = std::abs(a[i * n + k]);
            if (v > best) {
                best = v;
                p    = i;
            }
        }
        piv[k] = p;
        if (p != k) {
            for (std::size_t j = 0; j < n; ++j) std::swap(a[k * n + j], a[p * n + j]);
            std::swap(b[k], b[p]);
        }
        const double pv = a[k * n + k];
        // NaN は比較が全て偽になるので肯定形で書く: 有限かつ |p| > thresh のときだけ続行
        if (!(std::isfinite(pv) && std::abs(pv) > thresh)) return false;
        const double inv = 1.0 / pv;
        if (!std::isfinite(inv)) return false;

        for (std::size_t i = k + 1; i < n; ++i) {
            const double f = a[i * n + k] * inv;
            a[i * n + k]   = f;  // L の乗数を記録
            if (f == 0.0) continue;
            for (std::size_t j = k + 1; j < n; ++j) a[i * n + j] -= f * a[k * n + j];
            b[i] -= f * b[k];  // 前進代入を同時に進める
        }
    }

    // 後退代入 U x = y
    for (std::size_t i = n; i-- > 0;) {
        double s = b[i];
        for (std::size_t j = i + 1; j < n; ++j) s -= a[i * n + j] * b[j];
        b[i] = s / a[i * n + i];
        if (!std::isfinite(b[i])) return false;  // 有限なピボットでも積の overflow はありうる
    }
    return true;
}

}  // namespace quantviz::core
