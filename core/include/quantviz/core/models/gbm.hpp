#pragma once
// core/models/gbm.hpp — 幾何ブラウン運動（厳密離散化）
//
//   dS = mu S dt + sigma S dW
//   S_{t+dt} = S_t * exp( (mu - sigma^2/2) dt + sigma sqrt(dt) Z ),  Z ~ N(0,1)
//
// 離散化誤差ゼロ（対数正規の厳密解）なので、テストは統計量を直接検証できる。

#include <cmath>
#include <cstdint>

#include "quantviz/core/rng.hpp"

namespace quantviz::core {

struct GbmParams {
    double s0    = 100.0;  ///< 初期価格
    double mu    = 0.05;   ///< ドリフト（年率）
    double sigma = 0.20;   ///< ボラティリティ（年率）
};

class Gbm {
public:
    explicit Gbm(GbmParams p = {}, std::uint64_t seed = 42) : p_(p), s_(p.s0), t_(0.0), rng_(seed) {}

    /// 1 ステップ進め、そのステップの対数リターン log(S_{t+dt}/S_t) を返す。
    double step(double dt) {
        const double z = rng_.normal();
        const double r = (p_.mu - 0.5 * p_.sigma * p_.sigma) * dt + p_.sigma * std::sqrt(dt) * z;
        s_ *= std::exp(r);
        t_ += dt;
        return r;
    }

    double           spot() const noexcept { return s_; }
    double           time() const noexcept { return t_; }
    const GbmParams& params() const noexcept { return p_; }

    /// パラメータ変更は「次のステップから」効く。現在の S と t は不変。
    void set_mu(double mu) noexcept { p_.mu = mu; }
    void set_sigma(double sigma) noexcept { p_.sigma = sigma < 0.0 ? 0.0 : sigma; }

    /// パスを初期状態に戻す。mu / sigma は保持する（UI で変えた値を失わない）。
    void reset(std::uint64_t seed) {
        rng_.reseed(seed);
        s_ = p_.s0;
        t_ = 0.0;
    }
    void reset() { reset(rng_.seed()); }

private:
    GbmParams p_;
    double    s_;
    double    t_;
    Rng       rng_;
};

}  // namespace quantviz::core
