#pragma once
// bridge/model_concept.hpp — 「コアと描画の双対」を型で表す契約
//
//   Model       : step(dt) で時間発展し、snapshot() で POD の状態を吐き、apply(Command) で入力を受ける
//   Snapshot    : trivially copyable かつ default constructible（リングバッファに載せる条件）
//   SurfaceModel: Model の任意拡張。グリッド大の「面」を持つ Model が名乗る（M2）。
//                 面は履歴が要らない・1 枚が大きいので、リングではなく TripleBuffer で最新 1 枚だけ渡す
//
// 描画層は Snapshot（と面）の純関数として実装される。Model を直接参照してはならない。

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

/// 面チャネルを持つ Model。Runner は SurfaceModel のときだけ TripleBuffer<Surface> を生やす。
/// surface(s) は呼ばれたその場で現在の面を s に書く（確保も例外も無いこと＝ noexcept を要求）。
///
/// 【s は空ではない】s は TripleBuffer の使い回しスロットへの参照で、中身は「2 回前に publish した
/// 面」。ゼロ初期化も前回内容の保持も期待できないので、surface() は毎回**全フィールド**を書くこと。
/// 「変わった所だけ書く」実装は 2 世代前の値が混ざった面を描画側へ渡す。
///
/// Surface には Snapshot と同じ SnapshotType（trivially copyable かつ default constructible）を課す。
/// 計画書の字面は trivially copyable だけだが、TripleBuffer のスロットが T{} を要求するので、
/// 契約側で揃えて「概念は満たすのに TripleBuffer の static_assert で落ちる」状態を無くす。
template <class M>
concept SurfaceModel = Model<M> && requires(const M& cm, typename M::Surface& s) {
    typename M::Surface;
    requires SnapshotType<typename M::Surface>;
    { cm.surface(s) } noexcept;
};

}  // namespace quantviz::bridge
