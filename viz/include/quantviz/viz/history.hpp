#pragma once
// viz/history.hpp — 描画側の固定長循環履歴。ImPlot の offset 引数と組で使う。
//
// コアは「今の状態」しか吐かない（Snapshot は 1 枚）。時系列として並べるのは描画の責務。
// 描画スレッド専用・ゼロアロケーション。

#include <array>
#include <cstddef>

namespace quantviz::viz {

template <std::size_t N>
class History {
    static_assert(N >= 2, "History needs at least two points to draw a line");

public:
    static constexpr std::size_t capacity() noexcept { return N; }

    void push(double x, double y) noexcept {
        xs_[head_] = x;
        ys_[head_] = y;
        head_      = (head_ + 1) % N;
        if (count_ < N) ++count_;
    }

    /// ImPlot::PlotLine(label, xs(), ys(), count(), flags, offset())
    int           count() const noexcept { return static_cast<int>(count_); }
    int           offset() const noexcept { return count_ < N ? 0 : static_cast<int>(head_); }
    const double* xs() const noexcept { return xs_.data(); }
    const double* ys() const noexcept { return ys_.data(); }

    bool   empty() const noexcept { return count_ == 0; }
    double latest_x() const noexcept { return empty() ? 0.0 : xs_[(head_ + N - 1) % N]; }
    double latest_y() const noexcept { return empty() ? 0.0 : ys_[(head_ + N - 1) % N]; }
    double oldest_x() const noexcept { return empty() ? 0.0 : xs_[static_cast<std::size_t>(offset())]; }

    void clear() noexcept {
        head_  = 0;
        count_ = 0;
    }

private:
    std::array<double, N> xs_{};
    std::array<double, N> ys_{};
    std::size_t           head_  = 0;
    std::size_t           count_ = 0;
};

}  // namespace quantviz::viz
