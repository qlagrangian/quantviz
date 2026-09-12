#pragma once
// viz/perf.hpp — パフォーマンスパネルの「計算」部分（vizcore: GUI 非依存・単体テスト可能）
//
// 責務: 描画スレッドが毎フレーム集める数値（フレーム時間、step 時間、dropped 数）を、そのまま
// 読める形に変換する純粋な道具を 4 つ提供する。
//
//   * FixedHistogram<N> — 境界を自分で決める固定ビンのヒストグラム（フレーム時間の分布用）
//   * percentile_sorted — 整列済み標本からの p 分位（p50 / p95 / p99）
//   * Ema               — 指数移動平均（毎フレームの値をそのまま出すと踊って読めないため）
//   * dropped_ratio     — 落とした Snapshot の割合
//
// どれも時間源・乱数・状態共有を持たない（`RateMeter` と同じ方針）ので、合成入力で決定的に
// テストできる（PERF-01..04）。ゼロアロケーション・例外なし。Runner 側の step 時間ヒストグラム
// （`bridge::StepHistogram`、対数ビン固定）とは別物で、こちらは描画側が境界を決める汎用品。

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace quantviz::viz {

/// 固定ビンのヒストグラム。`edges` は単調増加の N+1 個の境界（呼び手が与える）。
///
/// ビンは `[edges[i], edges[i+1])`、最後のビンだけ上端を含む閉区間。範囲外の値は捨てずに
/// 端のビンへクランプする（パネルは「外れ値が何個あったか」ではなく「全部で何個来たか」を
/// 見せたいので、黙って落とすと total が合わなくなる）。NaN だけはどのビンにも入れられない
/// ので `dropped` で数える（比較が全部 false になる値をビンに入れる意味がないため）。
///
/// 集約体（aggregate）なので `FixedHistogram<4> h{{0.0, 1.0, 2.0, 3.0, 4.0}};` と境界だけ与えて作る。
template <std::size_t N>
struct FixedHistogram {
    static_assert(N >= 1, "FixedHistogram needs at least one bin");

    std::array<double, N + 1>    edges{};    ///< 単調増加の境界（呼び手が設定する）
    std::array<std::uint64_t, N> counts{};   ///< 各ビンの個数
    std::uint64_t                dropped = 0;  ///< NaN で入れられなかった個数

    static constexpr std::size_t bins() noexcept { return N; }

    /// 値を 1 つ入れる。NaN は `dropped` へ、範囲外は端のビンへ。
    void add(double v) noexcept {
        if (std::isnan(v)) {
            ++dropped;
            return;
        }
        ++counts[bin_of(v)];
    }

    /// `v` が入るビンの番号（範囲外はクランプ）。NaN を渡すとビン 0 を返すが、`add` は NaN を
    /// ここへ回さない（比較が全て false になる値をビンに入れても意味が無いため）。
    [[nodiscard]] std::size_t bin_of(double v) const noexcept {
        if (!(v >= edges[0])) return 0;       // 下端未満（および -inf）はビン 0
        if (v >= edges[N]) return N - 1;      // 上端以上（および +inf）は最後のビン
        // edges[lo] <= v < edges[hi] を保ちながら幅 1 まで詰める（境界が等間隔とは限らない）
        std::size_t lo = 0;
        std::size_t hi = N;
        while (hi - lo > 1) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (v >= edges[mid]) lo = mid;
            else                 hi = mid;
        }
        return lo;
    }

    /// 個数だけ 0 に戻す（境界は保つ）。
    void reset() noexcept {
        counts.fill(0);
        dropped = 0;
    }

    /// ビンに入った総数（`dropped` は含まない）。
    [[nodiscard]] std::uint64_t total() const noexcept {
        std::uint64_t sum = 0;
        for (const std::uint64_t c : counts) sum += c;
        return sum;
    }
};

/// 昇順に整列済みの標本から p 分位（0 ≤ p ≤ 1）を線形補間で返す。
///
/// 定義は numpy の既定（`linear`）と同じ: pos = p·(n−1) として、その前後の標本を按分する。
/// p50 は偶数個なら中央 2 点の平均、p = 0 は最小値、p = 1 は最大値。
/// 範囲外の p はクランプする。**NaN の p は 0 として扱う**（`!(p > 0)` を 0 に倒すので NaN もここに
/// 落ちる）。空の span は NaN を返す（「分位が無い」を表せる唯一の値）。整列は呼び手の責任。
[[nodiscard]] inline double percentile_sorted(std::span<const double> sorted, double p) noexcept {
    if (sorted.empty()) return std::nan("");
    if (!(p > 0.0)) p = 0.0;  // 負・NaN
    if (p > 1.0) p = 1.0;

    const double pos = p * static_cast<double>(sorted.size() - 1);
    const double lo  = std::floor(pos);
    const auto   i   = static_cast<std::size_t>(lo);
    if (i + 1 >= sorted.size()) return sorted[sorted.size() - 1];
    return sorted[i] + (pos - lo) * (sorted[i + 1] - sorted[i]);
}

/// 指数移動平均。`value += alpha·(sample − value)`。最初のサンプルで初期化する（0 から
/// 立ち上がる過渡を見せないため）。毎フレームの生値は揺れて読めないので、パネルの数字は
/// これを通す（`RateMeter` が内部でやっているのと同じことを、単体で使える形にしたもの）。
class Ema {
public:
    /// alpha は新しいサンプルの取り分。(0, 1] の外（0 以下・1 超・NaN）は **1.0 = 平滑化なし**に
    /// 倒す（0 に倒すと値が永久に固まり「壊れている」ように見えるため）。
    explicit Ema(double alpha) noexcept : alpha_((alpha > 0.0 && alpha <= 1.0) ? alpha : 1.0) {}

    /// サンプルを 1 つ。NaN は無視する（1 つの NaN で以後ずっと NaN になるのを防ぐ）。
    void add(double v) noexcept {
        if (std::isnan(v)) return;
        if (!initialised_) {
            value_       = v;
            initialised_ = true;
            return;
        }
        value_ += alpha_ * (v - value_);
    }

    /// 現在値。サンプルが 1 つも入っていなければ 0。
    [[nodiscard]] double value() const noexcept { return value_; }
    [[nodiscard]] bool   initialised() const noexcept { return initialised_; }
    [[nodiscard]] double alpha() const noexcept { return alpha_; }

    /// 次の `add` が「最初のサンプル」に戻る（シーン切り替えや Reset 用）。
    void reset() noexcept {
        value_       = 0.0;
        initialised_ = false;
    }

private:
    double alpha_;
    double value_       = 0.0;
    bool   initialised_ = false;
};

/// 落とした割合 = dropped / (dropped + received)。まだ 1 つも来ていない（0/0）なら 0。
[[nodiscard]] inline double dropped_ratio(std::uint64_t dropped, std::uint64_t received) noexcept {
    const std::uint64_t total = dropped + received;
    if (total == 0) return 0.0;
    return static_cast<double>(dropped) / static_cast<double>(total);
}

}  // namespace quantviz::viz
