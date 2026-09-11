#pragma once
// bridge/model_concept.hpp — 「コアと描画の双対」を型で表す契約
//
//   Model  : step(dt) で時間発展し、snapshot() で POD の状態を吐き、apply(Command) で入力を受ける
//   Snapshot: trivially copyable かつ default constructible（リングバッファに載せる条件）
//
// 描画層は Snapshot の純関数として実装される。Model を直接参照してはならない。

#include <concepts>
#include <type_traits>

#include "quantviz/bridge/command.hpp"

namespace quantviz::bridge {

template <class S>
concept SnapshotType = std::is_trivially_copyable_v<S> && std::is_default_constructible_v<S>;

template <class M>
concept Model = requires(M& m, const M& cm, double dt, const Command& c) {
    typename M::Snapshot;
    requires SnapshotType<typename M::Snapshot>;
    { m.step(dt) };
    { cm.snapshot() } -> std::same_as<typename M::Snapshot>;
    { m.apply(c) };
};

}  // namespace quantviz::bridge
