#pragma once
// core/stats/ewma.hpp — 指数加重移動分散（RiskMetrics 型, 平均ゼロ仮定）
//
//   var_1 = r_1^2
//   var_n = lambda * var_{n-1} + (1 - lambda) * r_n^2
//
// 閉形式: var_n = lambda^{n-1} r_1^2 + (1-lambda) sum_{i=2}^{n} lambda^{n-i} r_i^2

#include <cmath>
#include <cstdint>

namespace quantviz::core {

class EwmaVariance {
public:
    static constexpr double kMinLambda = 1e-9;
    static constexpr double kMaxLambda = 1.0 - 1e-9;

    explicit EwmaVariance(double lambda = 0.94) noexcept { set_lambda(lambda); }

    void push(double r) noexcept {
        const double r2 = r * r;
        var_            = (n_ == 0) ? r2 : lambda_ * var_ + (1.0 - lambda_) * r2;
        ++n_;
    }

    double        variance() const noexcept { return var_; }
    double        volatility() const noexcept { return std::sqrt(var_); }
    double        lambda() const noexcept { return lambda_; }
    std::uint64_t count() const noexcept { return n_; }

    /// (0,1) にクランプ。UI からの不正値でも壊れないようにする。NaN はデフォルトに戻す。
    void set_lambda(double lambda) noexcept {
        if (!(lambda == lambda)) lambda = 0.94;
        lambda_ = lambda < kMinLambda ? kMinLambda : (lambda > kMaxLambda ? kMaxLambda : lambda);
    }

    void reset() noexcept {
        n_   = 0;
        var_ = 0.0;
    }

private:
    double        lambda_ = 0.94;
    double        var_    = 0.0;
    std::uint64_t n_      = 0;
};

}  // namespace quantviz::core
