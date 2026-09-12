#pragma once
// core/stats/kalman.hpp — 線形カルマンフィルタ（固定次元・ヒープ確保なし・例外なし）。
//
//   状態方程式  x_t = F x_{t−1} + w_t,  w ~ N(0, Q)
//   観測方程式  z_t = H x_t     + v_t,  v ~ N(0, R)
//
// predict/update ともホットパス（scenes の step から毎ステップ呼ばれる）なので、
// 確保も例外もロックも無い。共分散の更新は Joseph 形
//   P⁺ = (I − K H) P⁻ (I − K H)ᵀ + K R Kᵀ
// を使う。単純形 P⁺ = (I − K H) P⁻ より演算は多いが、丸め誤差が入っても対称・半正定値が
// 崩れにくい（KALMAN-05 が 1000 ステップ後の対称性と PSD を検査する）。さらに毎回
// 上三角だけを平均して下三角へ複製し（symmetrized）、厳密対称を構造的に保証する。
//
// 退化した入力（NaN 観測・特異なイノベーション共分散）は状態を壊さずスキップし、
// 回数を `skipped_updates()` で数える。UI は「フィルタが実質止まっている」ことを
// これで検知できる。

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "quantviz/core/math/mat.hpp"

namespace quantviz::core {

/// NX 次元の状態・NZ 次元の観測を持つ線形カルマンフィルタ。
/// NZ は `Mat::inverse()` の閉形式に合わせて 3 以下。NX に上限は無い。
template <std::size_t NX, std::size_t NZ>
class Kalman {
public:
    static_assert(NX > 0 && NZ > 0, "Kalman requires positive dimensions");
    static_assert(NZ <= 3, "Kalman needs a closed-form inverse of the NZ x NZ innovation covariance");

    using StateVec  = Mat<NX, 1>;
    using StateMat  = Mat<NX, NX>;
    using ObsVec    = Mat<NZ, 1>;
    using ObsMat    = Mat<NZ, NZ>;
    using DesignMat = Mat<NZ, NX>;

    /// 既定の事前分散。x = 0, P = I · 1e6 で「ほぼ無情報」な事前分布から始める
    /// （完全な拡散事前分布は表現できないので、観測スケールに対して十分大きい有限値を使う。
    ///  有限であるぶん最初の数ステップに 1/(1 + n·P0/R) 程度の相対バイアスが残る）。
    static constexpr double kDefaultPriorVariance = 1e6;

    /// 既定構築: x = 0, P = I · kDefaultPriorVariance。
    Kalman() noexcept : Kalman(StateVec{}, StateMat::diagonal(kDefaultPriorVariance)) {}

    /// P0 は対称化して保持する（上三角を採って下三角へ複製）。半正定値であることは
    /// **検証しない**（ホットパスに判定を置かない方針）。呼び出し側の責任。
    Kalman(const StateVec& x0, const StateMat& p0) noexcept : x_(x0), p_(symmetrized(p0)) {}

    /// 時間更新: x ← F x, P ← F P Fᵀ + Q。
    void predict(const StateMat& f, const StateMat& q) noexcept {
        x_ = f * x_;
        p_ = symmetrized(f * p_ * f.transpose() + q);
    }

    /// 観測更新。H は時変でよい（回帰なら H = [x_t]）。戻り値は事前残差（イノベーション）
    ///   y = z − H x⁻
    /// で、これは更新前の状態に対する残差。次の 2 つの場合は状態・共分散を一切変えずに
    /// y だけ返し、`skipped_updates()` を 1 増やす（例外を投げない方針のため）:
    ///   1. y に非有限成分がある（z か現在の状態が NaN/inf）。1 個の NaN 観測が状態を
    ///      恒久的に汚染するのを防ぐ。
    ///   2. S = H P⁻ Hᵀ + R の逆行列が取れない。NZ ≥ 2 では S が階数落ちしているだけでも
    ///      スキップするので、本来は情報のある部分空間まで捨てることに注意（部分空間だけを
    ///      使う擬似逆は閉形式の範囲を超えるため、ここでは採らない）。
    ObsVec update(const DesignMat& h, const ObsVec& z, const ObsMat& r) noexcept {
        const ObsVec y = z - h * x_;
        for (std::size_t i = 0; i < NZ; ++i)
            if (!std::isfinite(y(i, 0))) {
                ++skipped_updates_;
                return y;
            }

        const ObsMat s     = h * p_ * h.transpose() + r;
        const auto   s_inv = s.inverse();
        if (!s_inv) {
            ++skipped_updates_;
            return y;
        }

        const Mat<NX, NZ> k = p_ * h.transpose() * *s_inv;

        x_ = x_ + k * y;

        // Joseph 形: P⁺ = (I − K H) P⁻ (I − K H)ᵀ + K R Kᵀ
        const StateMat ikh = StateMat::identity() - k * h;
        p_                 = symmetrized(ikh * p_ * ikh.transpose() + k * r * k.transpose());
        return y;
    }

    [[nodiscard]] const StateVec& state() const noexcept { return x_; }
    [[nodiscard]] const StateMat& cov() const noexcept { return p_; }

    /// 非有限観測・特異な S でスキップした update の累計回数。reset で 0 に戻す。
    [[nodiscard]] std::uint64_t skipped_updates() const noexcept { return skipped_updates_; }

    /// 状態と共分散を初期値に戻す。P0 はコンストラクタと同じく対称化して保持し、
    /// 半正定値かどうかは検証しない。`skipped_updates()` も 0 に戻る。
    void reset(const StateVec& x0, const StateMat& p0) noexcept {
        x_               = x0;
        p_               = symmetrized(p0);
        skipped_updates_ = 0;
    }

    /// 既定の初期値（x = 0, P = I · kDefaultPriorVariance）に戻す。
    void reset() noexcept { reset(StateVec{}, StateMat::diagonal(kDefaultPriorVariance)); }

private:
    /// ½(P + Pᵀ)。上三角だけを計算して下三角へ複製するので、(i,j) と (j,i) は
    /// 同じ式の同じ結果になり、FMA 縮約や加算順序に関係なく厳密対称になる。
    static StateMat symmetrized(const StateMat& m) noexcept {
        StateMat out{};
        for (std::size_t i = 0; i < NX; ++i) {
            out(i, i) = m(i, i);
            for (std::size_t j = i + 1; j < NX; ++j) {
                const double v = 0.5 * (m(i, j) + m(j, i));
                out(i, j)      = v;
                out(j, i)      = v;
            }
        }
        return out;
    }

    StateVec      x_{};
    StateMat      p_{};
    std::uint64_t skipped_updates_ = 0;
};

// Snapshot やモデルに値として持てること（コピーが bit 単位で安全）を型で固定する。
static_assert(std::is_trivially_copyable_v<Kalman<2, 1>>, "Kalman must stay trivially copyable");

}  // namespace quantviz::core
