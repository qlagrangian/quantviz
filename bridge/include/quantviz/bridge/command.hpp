#pragma once
// bridge/command.hpp — 描画 → コア への唯一の入力経路
//
// 時計系（Pause/Resume/StepOnce/SetSpeed）は Runner が SimClock に適用し、
// それ以外（SetParam/Reset）は Model::apply() に転送される。

#include <cstdint>
#include <type_traits>

namespace quantviz::bridge {

enum class CommandType : std::uint8_t {
    SetParam = 0,  ///< param_id, value    → Model::apply
    Reset    = 1,  ///< seed (0 = 現在の seed) → Model::apply
    Pause    = 2,  ///< → SimClock
    Resume   = 3,  ///< → SimClock
    StepOnce = 4,  ///< → SimClock（一時停止中でも 1 ステップだけ進める）
    SetSpeed = 5,  ///< value = 倍率 → SimClock
};

struct Command {
    CommandType   type     = CommandType::SetParam;
    std::uint32_t param_id = 0;
    double        value    = 0.0;
    std::uint64_t seed     = 0;

    static constexpr Command set_param(std::uint32_t id, double v) noexcept {
        return Command{CommandType::SetParam, id, v, 0};
    }
    static constexpr Command reset(std::uint64_t seed = 0) noexcept {
        return Command{CommandType::Reset, 0, 0.0, seed};
    }
    static constexpr Command pause() noexcept { return Command{CommandType::Pause, 0, 0.0, 0}; }
    static constexpr Command resume() noexcept { return Command{CommandType::Resume, 0, 0.0, 0}; }
    static constexpr Command step_once() noexcept { return Command{CommandType::StepOnce, 0, 0.0, 0}; }
    static constexpr Command set_speed(double multiplier) noexcept {
        return Command{CommandType::SetSpeed, 0, multiplier, 0};
    }
};

static_assert(std::is_trivially_copyable_v<Command>, "Command must stay a POD (it travels through SpscRing)");

constexpr bool is_clock_command(CommandType t) noexcept {
    return t == CommandType::Pause || t == CommandType::Resume || t == CommandType::StepOnce ||
           t == CommandType::SetSpeed;
}

}  // namespace quantviz::bridge
