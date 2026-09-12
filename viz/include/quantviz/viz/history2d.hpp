#pragma once
// viz/history2d.hpp — 価格×時間の 2D 循環履歴（ヒートマップ用、vizcore: GUI・GL ヘッダ非依存）。
//
// 責務: 固定サイズ（Rows 行 × Cols 列）の列循環バッファ。1 列 = ある時刻のスナップショットから
// 縮約した Rows 個の値（例: mid を中心にした価格ビンごとの数量）を `push_column` で追加する。
// `History`（viz/history.hpp）と同じ「列方向に循環」の規約だが、こちらは 2 次元なので、内部は
// 列優先（column-major: 1 列 = Rows 個が連続）で保持し、`ordered()` が呼ばれたときだけ 1 回の
// コピーで「最古→最新」の列順・行優先（row-major、[row][col]）に並べ替える。これは
// `ImPlot::PlotHeatmap(values, rows=Rows, cols=count())` にそのまま渡せる形。
//
// `ordered()` の戻り値はインスタンスが 1 つだけ持つ可変（mutable）スクラッチ配列を指しており、
// 呼び出しのたびに上書きされる。よって **`ordered()` は描画（レンダー）スレッド専用** ——
// 複数スレッドから同時に呼んだり、前回の戻り値を次の呼び出しの後まで保持したりしない。
//
// `push_column` は Rows 要素の span を想定する: 短い span は（範囲外読み出しを避けるため）
// 何もせず無視する。長い span は先頭 Rows 要素だけを使う。ヒープも例外も使わない
// （コンストラクタで確保する `std::array` のみ）。

#include <array>
#include <cstddef>
#include <span>

namespace quantviz::viz {

template <std::size_t Rows, std::size_t Cols>
class History2D {
    static_assert(Rows >= 1, "History2D needs at least one row");
    static_assert(Cols >= 1, "History2D needs at least one column");

public:
    /// column は Rows 要素であること。短い span は無視し、長い span は先頭 Rows 要素だけを使う。
    void push_column(std::span<const float> column) noexcept {
        if (column.size() < Rows) return;
        float* dst = &ring_[head_ * Rows];
        for (std::size_t r = 0; r < Rows; ++r) dst[r] = column[r];
        head_ = (head_ + 1) % Cols;
        if (count_ < Cols) ++count_;
    }

    /// 埋まった列数（Cols で飽和）。
    std::size_t count() const noexcept { return count_; }

    /// row-major [row][col]（col は最古→最新、count() 列分、要素数 Rows*count()）。
    /// 戻り値は内部スクラッチ配列を指す（描画スレッド専用。次の呼び出しで上書きされる）。
    /// コストは Rows*count() 個の float のコピー（64×256 なら 64 KB、毎フレーム呼んでも 60 fps で十分軽い）。
    std::span<const float> ordered() const noexcept {
        const std::size_t n      = count_;
        const std::size_t oldest = (count_ < Cols) ? 0 : head_;
        for (std::size_t c = 0; c < n; ++c) {
            const std::size_t slot = (oldest + c) % Cols;
            const float*      src  = &ring_[slot * Rows];
            for (std::size_t r = 0; r < Rows; ++r) scratch_[r * n + c] = src[r];
        }
        return std::span<const float>(scratch_.data(), Rows * n);
    }

    /// 全ゼロ・count 0 に戻す。再アロケーションなし（次の push は先頭スロットから）。
    void clear() noexcept {
        ring_.fill(0.0f);
        scratch_.fill(0.0f);
        head_  = 0;
        count_ = 0;
    }

private:
    std::array<float, Rows * Cols>         ring_{};
    mutable std::array<float, Rows * Cols> scratch_{};
    std::size_t                            head_  = 0;
    std::size_t                            count_ = 0;
};

}  // namespace quantviz::viz
