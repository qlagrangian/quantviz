#pragma once
// core/rng.hpp — 決定性のための乱数ラッパ。
// 依存: <random> のみ。seed が同じなら同じ系列（同一ツールチェーン内）。

#include <cstdint>
#include <random>

namespace quantviz::core {

class Rng {
public:
    explicit Rng(std::uint64_t seed = 42) : seed_(seed), engine_(seed) {}

    /// N(0,1)
    double normal() { return normal_(engine_); }
    /// U(0,1)
    double uniform() { return uniform_(engine_); }

    /// 同じ seed で reseed すると、以後の系列は最初から同一になる。
    void reseed(std::uint64_t seed) {
        seed_ = seed;
        engine_.seed(seed);
        normal_.reset();  // Box–Muller のキャッシュを捨てる
        uniform_.reset();
    }

    std::uint64_t seed() const noexcept { return seed_; }

private:
    std::uint64_t                          seed_;
    std::mt19937_64                        engine_;
    std::normal_distribution<double>       normal_{0.0, 1.0};
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};
};

}  // namespace quantviz::core
