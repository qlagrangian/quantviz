#pragma once
// core/aad/tape.hpp — テープ式の随伴自動微分（reverse-mode AAD）。`Tape` が演算の記録（テープ）を持ち、
// `Var` が「値 + ノード番号 + テープへのポインタ」を運ぶ。逆伝播 1 回で全入力の偏微分が同時に得られるので、
// 入力数 n に対して n + 1 回の再評価が要るバンプ法と違い、コストは記録した演算数にしか比例しない。
//
// 記録と逆伝播:
//   y = f(x0, …, x_{n-1}) を Var で普通に計算すると、演算ごとに「親ノード番号と局所偏微分」の組が
//   テープに 1 行追記される（親は最大 2 つ = 二項演算まで。これで BS カーネルの語彙は足りる）。
//   propagate(y.node(), adj) は adj[y] = 1 として番号の大きい方から 1 回走査し、各行で
//   adj[parent] += adj[i] * partial を加える。テープは常に位相順（親の番号 < 子の番号）なので
//   後ろ向きの単一ループで済み、計算量は O(size)、仮想関数もソートも要らない。
//
// 設計上の決定（すべてホットパスを確保なし・例外なしに保つため）:
//   * ノードは SoA（親 index 2 本 + 偏微分 2 本の並列配列）。バイト数は AoS（32 B/node）と同じだが、
//     随伴が 0 のノードを飛ばすとき偏微分配列のキャッシュラインに触れずに済み、将来 index を
//     std::uint32_t に落とす変更も配列 1 本の差し替えで済む。
//   * 確保するのは reserve() ただ 1 つ（= 唯一の非 noexcept 関数）。記録は resize 済み領域への
//     添字代入だけで、std::vector が伸びることはない（AAD-08）。したがって容量を超えた記録は
//     「記録失敗」になる: push_* は番兵 kNoNode を返し、Var は値の計算だけ続ける（NaN にも UB にもしない）。
//     失敗した Var を使った下流の演算も kNoNode を伝播し、kNoNode に対する propagate は何もしない。
//     つまり「テープが足りなければ勾配が全部 0 になる」だけで、値は常に正しい。
//   * ノードは値を持たない（値は Var 側）。逆伝播に要るのは局所偏微分だけなので、テープは 1 ノード
//     32 B（index 2 + double 2）に収まる。
//   * テープはスコープ内で明示的に渡す。thread_local のグローバルテープは使わない（どの Var が
//     どのテープに載っているかが型と引数から追えなくなり、並列化の邪魔にもなるため）。
//
// 数値の約束:
//   * 値の算術は素の + − × ÷ と <cmath> で、fma には融合させない（pricing/black_scholes.hpp の
//     ScalarOps と同じ演算列を AadOps（M5 Task 2）が再現できるようにするため）。
//   * 定義域は呼び手の責任。log(0) の偏微分は +inf、sqrt(0) の偏微分は +inf、負の引数は NaN になる
//     （IEEE のまま伝播させ、トラップも置き換えもしない）。
//   * 定数（Var::constant / スカラとの演算）はノードを作らない。テープ長 = 入力 leaf 数 + 演算回数。
//
// 前提と未定義動作:
//   * Tape は、それに載る Var より長生きすること（Tape のコピー・move は delete してあるので、
//     Var の Tape* が別のオブジェクトに付け替わることはない）。
//   * 別々の Tape に載った Var どうしの二項演算は Debug では assert、Release では未定義。

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace quantviz::core::aad {

/// 「テープ上のノードではない」ことを表す番兵。記録失敗（容量超過・無効な親）と定数の node() に使う。
inline constexpr std::size_t kNoNode = std::numeric_limits<std::size_t>::max();

// pricing/black_scholes.hpp と同じ値を持つが、このヘッダは std だけに依存させたいので複製してある
// （AAD が BS を include する筋合いはない）。ずれの見張りは AAD-02 の
// `norm_cdf(Var) == core::norm_cdf(double)` のビット一致検査（片方だけ桁を落とせば必ず落ちる）。
namespace detail {
inline constexpr double kInvSqrt2      = 0.70710678118654752440084436210485;
inline constexpr double kInvSqrt2Pi    = 0.39894228040143267793994605993438;
inline constexpr double kTwoOverSqrtPi = 1.1283791670955125738961589031215;
}  // namespace detail

/// 演算の記録（テープ）。ノードは値を持たず、親ノード番号 ≤ 2 と局所偏微分だけを持つ。
class Tape {
public:
    Tape() = default;

    /// コピーも move もできない。Var は生の `Tape*` を持つので、move されると Var のポインタが
    /// 静かに「中身を抜かれた元のテープ」を指し続け、読むと解放済みの配列に当たる。モデルのメンバ
    /// として持つのは問題ない（そのモデル自体を move しない限り。M5 Task 4 の AadModel は
    /// Runner の中で in-place に構築されるのでこの制限に当たらない）。
    Tape(const Tape&)            = delete;
    Tape& operator=(const Tape&) = delete;
    Tape(Tape&&)                 = delete;
    Tape& operator=(Tape&&)      = delete;

    /// n_nodes 個のノードを記録できるようにする。**ここだけが確保する**（したがってここだけが
    /// 例外を投げうる）。容量は縮まない（n_nodes ≤ capacity() なら何もしない）。既に記録した内容と
    /// size() は保たれるので、記録の途中で足りなくなったときに呼んでも構わない。
    void reserve(std::size_t n_nodes) {
        if (n_nodes <= parent0_.size()) return;
        parent0_.resize(n_nodes, kNoNode);
        parent1_.resize(n_nodes, kNoNode);
        partial0_.resize(n_nodes, 0.0);
        partial1_.resize(n_nodes, 0.0);
    }

    /// テープ長を 0 に戻す（容量は保持 = 次の記録も確保なし）。rewind をまたいで持ち越した Var は
    /// **同じ番号に載る別のノードの別名**になる（値だけは正しいまま）。そのまま propagate すると
    /// 無関係なノードの随伴が返るので、rewind したら入力 Var も作り直すこと。
    void rewind() noexcept { n_ = 0; }

    /// 記録済みノード数 = 入力 leaf の数 + 記録した演算の回数（定数・スカラ定数はノードを作らない）。
    [[nodiscard]] std::size_t size() const noexcept { return n_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return parent0_.size(); }

    /// 入力変数（親なし）。戻り値はノード番号、容量超過なら kNoNode。
    [[nodiscard]] std::size_t push_leaf() noexcept { return emplace(kNoNode, 0.0, kNoNode, 0.0); }

    /// 単項演算 y = f(x): parent = x のノード、partial = df/dx。親が未記録・kNoNode なら kNoNode。
    [[nodiscard]] std::size_t push_unary(std::size_t parent, double partial) noexcept {
        if (parent >= n_) return kNoNode;
        return emplace(parent, partial, kNoNode, 0.0);
    }

    /// 二項演算 y = f(a, b): (p1, d1) = (a のノード, df/da)、(p2, d2) = (b のノード, df/db)。
    [[nodiscard]] std::size_t push_binary(std::size_t p1, double d1, std::size_t p2, double d2) noexcept {
        if (p1 >= n_ || p2 >= n_) return kNoNode;
        return emplace(p1, d1, p2, d2);
    }

    /// 逆伝播。result_node の随伴を 1 として全ノードの随伴を adjoints に書く（adjoints[i] = dy/dnode_i）。
    /// adjoints は size() 要素以上を渡すこと。足りなければ **何もしない**（呼び手のバッファを壊さない）。
    /// result_node が無効（kNoNode = 記録失敗、または size() 以上）のときも何もしない。
    /// 親のノード番号は必ず子より小さいので、走査は result_node から 0 へ下る 1 ループで足りる
    /// （result_node より後のノードは結果に寄与しないので触れない）。随伴が 0 のノードは飛ばす
    /// ＝ 偏微分の読み出しを省くと同時に、0 × inf で NaN を作らない。
    void propagate(std::size_t result_node, std::span<double> adjoints) const noexcept {
        if (adjoints.size() < n_ || result_node >= n_) return;
        for (std::size_t i = 0; i < n_; ++i) adjoints[i] = 0.0;
        adjoints[result_node] = 1.0;
        for (std::size_t i = result_node + 1; i-- > 0;) {
            const double a = adjoints[i];
            if (a == 0.0) continue;
            const std::size_t q0 = parent0_[i];
            if (q0 != kNoNode) adjoints[q0] += a * partial0_[i];
            const std::size_t q1 = parent1_[i];
            if (q1 != kNoNode) adjoints[q1] += a * partial1_[i];
        }
    }

private:
    /// 1 行追記して番号を返す。容量超過は kNoNode（確保しない）。
    std::size_t emplace(std::size_t p0, double d0, std::size_t p1, double d1) noexcept {
        if (n_ == parent0_.size()) return kNoNode;
        const std::size_t i = n_++;
        parent0_[i]         = p0;
        partial0_[i]        = d0;
        parent1_[i]         = p1;
        partial1_[i]        = d1;
        return i;
    }

    std::vector<std::size_t> parent0_;
    std::vector<std::size_t> parent1_;
    std::vector<double>      partial0_;
    std::vector<double>      partial1_;
    std::size_t              n_ = 0;
};

/// テープに載る実数。値・ノード番号・テープの 3 つだけを持つ小さな値型（trivially copyable）で、
/// コピーは自由（同じノードを指す別名になる）。
///
/// 3 つの状態がある:
///   * 通常 — tape() != nullptr かつ node() < tape()->size()。
///   * 定数 — tape() == nullptr（Var::constant、既定構築、定数どうしの演算）。ノードを作らず、
///     微分に寄与しない。AadOps::splat はこれを返す（トレイトは Tape を持てないため）。
///   * 記録失敗 — tape() != nullptr かつ node() == kNoNode（テープ容量超過）。値は正しく、
///     下流の演算にも伝播する。この Var で propagate しても何も起きない。
class Var {
public:
    /// 定数 0。
    Var() noexcept = default;

    /// 入力変数（leaf）としてテープに載せる。
    Var(Tape& t, double v) noexcept : tape_(&t), value_(v), node_(t.push_leaf()) {}

    /// テープに載らない定数（微分は 0）。
    [[nodiscard]] static Var constant(double v) noexcept { return Var(nullptr, v, kNoNode); }

    [[nodiscard]] double      value() const noexcept { return value_; }
    [[nodiscard]] std::size_t node() const noexcept { return node_; }
    [[nodiscard]] Tape*       tape() const noexcept { return tape_; }
    [[nodiscard]] bool        is_constant() const noexcept { return tape_ == nullptr; }
    /// テープに載るはずだったのに記録できなかった（容量超過）か。
    [[nodiscard]] bool recording_failed() const noexcept { return tape_ != nullptr && node_ == kNoNode; }

    // -------------------------------------------------------------------------------------------
    // 算術。Var どうし・Var とスカラ（左右どちらでも）。スカラは定数なのでノードは 1 つ（単項）だけ増える。
    // -------------------------------------------------------------------------------------------
    friend Var operator+(const Var& a, const Var& b) noexcept {
        return binary(a, b, a.value_ + b.value_, 1.0, 1.0);
    }
    friend Var operator-(const Var& a, const Var& b) noexcept {
        return binary(a, b, a.value_ - b.value_, 1.0, -1.0);
    }
    friend Var operator*(const Var& a, const Var& b) noexcept {
        return binary(a, b, a.value_ * b.value_, b.value_, a.value_);
    }
    friend Var operator/(const Var& a, const Var& b) noexcept {
        const double v = a.value_ / b.value_;  // 値は素の除算（ScalarOps::div とビット一致）
        return binary(a, b, v, 1.0 / b.value_, -v / b.value_);
    }

    friend Var operator+(const Var& a) noexcept { return a; }
    friend Var operator-(const Var& a) noexcept { return unary(a, -a.value_, -1.0); }

    friend Var operator+(const Var& a, double s) noexcept { return unary(a, a.value_ + s, 1.0); }
    friend Var operator+(double s, const Var& b) noexcept { return unary(b, s + b.value_, 1.0); }
    friend Var operator-(const Var& a, double s) noexcept { return unary(a, a.value_ - s, 1.0); }
    friend Var operator-(double s, const Var& b) noexcept { return unary(b, s - b.value_, -1.0); }
    friend Var operator*(const Var& a, double s) noexcept { return unary(a, a.value_ * s, s); }
    friend Var operator*(double s, const Var& b) noexcept { return unary(b, s * b.value_, s); }
    friend Var operator/(const Var& a, double s) noexcept { return unary(a, a.value_ / s, 1.0 / s); }
    friend Var operator/(double s, const Var& b) noexcept {
        const double v = s / b.value_;
        return unary(b, v, -v / b.value_);
    }

    Var& operator+=(const Var& b) noexcept { return *this = *this + b; }
    Var& operator-=(const Var& b) noexcept { return *this = *this - b; }
    Var& operator*=(const Var& b) noexcept { return *this = *this * b; }
    Var& operator/=(const Var& b) noexcept { return *this = *this / b; }
    Var& operator+=(double s) noexcept { return *this = *this + s; }
    Var& operator-=(double s) noexcept { return *this = *this - s; }
    Var& operator*=(double s) noexcept { return *this = *this * s; }
    Var& operator/=(double s) noexcept { return *this = *this / s; }

    // -------------------------------------------------------------------------------------------
    // 初等関数。演算子と違って **名前空間スコープの関数**（クラス内の宣言は friend 宣言だけ）に
    // してある: hidden friend だと ADL でしか見つからず、`aad::log(x)` と修飾して呼べないため。
    // M5 Task 2 の AadOps は `static V log(const V& a) { return aad::log(a); }` のように自分と
    // 同名の静的メンバから呼ぶので、修飾できないと無限再帰になる。
    // -------------------------------------------------------------------------------------------
    friend Var exp(const Var& a) noexcept;
    friend Var log(const Var& a) noexcept;
    friend Var sqrt(const Var& a) noexcept;
    friend Var erfc(const Var& a) noexcept;
    friend Var norm_cdf(const Var& a) noexcept;

private:
    Var(Tape* t, double v, std::size_t n) noexcept : tape_(t), value_(v), node_(n) {}

    /// 単項の記録。定数はノードを作らず定数のまま、記録失敗（kNoNode）はそのまま下流に伝播する。
    static Var unary(const Var& a, double value, double partial) noexcept {
        if (a.tape_ == nullptr) return Var(nullptr, value, kNoNode);
        return Var(a.tape_, value, a.tape_->push_unary(a.node_, partial));
    }

    /// 二項の記録。片方が定数なら単項に落ちる（ノードは 1 つ）。両方が定数なら結果も定数。
    static Var binary(const Var& a, const Var& b, double value, double da, double db) noexcept {
        if (a.tape_ == nullptr) return unary(b, value, db);
        if (b.tape_ == nullptr) return unary(a, value, da);
        assert(a.tape_ == b.tape_ && "quantviz::core::aad::Var: operands live on different tapes");
        return Var(a.tape_, value, a.tape_->push_binary(a.node_, da, b.node_, db));
    }

    Tape*       tape_  = nullptr;
    double      value_ = 0.0;
    std::size_t node_  = kNoNode;
};

inline Var exp(const Var& a) noexcept {
    const double v = std::exp(a.value_);
    return Var::unary(a, v, v);
}

inline Var log(const Var& a) noexcept { return Var::unary(a, std::log(a.value_), 1.0 / a.value_); }

inline Var sqrt(const Var& a) noexcept {
    const double v = std::sqrt(a.value_);
    return Var::unary(a, v, 0.5 / v);  // x = 0 では +inf（定義域は呼び手の責任）
}

/// 相補誤差関数。d/dx erfc(x) = −2/√π · e^{−x²}
inline Var erfc(const Var& a) noexcept {
    const double d = -detail::kTwoOverSqrtPi * std::exp(-a.value_ * a.value_);
    return Var::unary(a, std::erfc(a.value_), d);
}

/// 標準正規の累積分布 Φ(x) = 0.5 erfc(−x/√2)（BS カーネルの bs_cdf と同じ演算順 = 値はビット一致）。
/// 微分は密度 φ(x)。
inline Var norm_cdf(const Var& a) noexcept {
    const double v = 0.5 * std::erfc(-a.value_ * detail::kInvSqrt2);
    const double d = detail::kInvSqrt2Pi * std::exp(-0.5 * a.value_ * a.value_);
    return Var::unary(a, v, d);
}

static_assert(std::is_trivially_copyable_v<Var>, "Var must stay a small trivially copyable value");
static_assert(sizeof(Var) <= 24, "Var must stay pointer + double + index");

}  // namespace quantviz::core::aad
