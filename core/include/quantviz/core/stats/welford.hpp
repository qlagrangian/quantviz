#pragma once
// core/stats/welford.hpp — 逐次平均・分散（Welford 法, O(1) 更新, 数値的に安定）

#include <cmath>
#include <cstdint>

namespace quantviz::core {

class Welford {
public:
    void push(double x) noexcept {
        ++n_;
        const double delta = x - mean_;
        mean_ += delta / static_cast<double>(n_);
        m2_ += delta * (x - mean_);  // 更新後の mean_ を使うのが Welford の要点
    }

    std::uint64_t count() const noexcept { return n_; }
    double        mean() const noexcept { return mean_; }

    /// 標本分散（n-1 で割る）。n < 2 のときは 0。
    double variance() const noexcept { return n_ > 1 ? m2_ / static_cast<double>(n_ - 1) : 0.0; }
    /// 母分散（n で割る）。n == 0 のときは 0。
    double population_variance() const noexcept { return n_ > 0 ? m2_ / static_cast<double>(n_) : 0.0; }
    double stddev() const noexcept { return std::sqrt(variance()); }

    void reset() noexcept {
        n_    = 0;
        mean_ = 0.0;
        m2_   = 0.0;
    }

private:
    std::uint64_t n_    = 0;
    double        mean_ = 0.0;
    double        m2_   = 0.0;
};

}  // namespace quantviz::core
