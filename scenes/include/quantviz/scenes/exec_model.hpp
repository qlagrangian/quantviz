#pragma once
// scenes/exec_model.hpp — シーン「Optimal execution」: Almgren–Chriss (2000) の最適執行を
// 「λ（リスク回避）別の残量軌道の束 + 効率的フロンティア + 面 x(t, λ)」として見せる Model。
//
// 責務: `core/exec/almgren_chriss.hpp` の純関数を決まった格子で引き直し、描画側に 2 系統を渡す:
//
//   1. `Snapshot`（リング）: λ の梯子 kL = 8 本ぶんの軌道（各 kN = 64 点）とそのコスト、現在の λ の
//      軌道とコスト、フロンティア格子 kFrontier = 32 点、κ、再生ヘッド t とそこでの残量。≈ 5.6 KB。
//   2. `Surface`（`bridge::TripleBuffer`）: 同じ軌道を λ について 32 本に細かくした面
//      x(t; λ)（kT = 64 × kL = 32 の float、row-major [iL][iT]）。≈ 8.4 KB。
//
// 【このシーンに「掃引」は無い】FDM や LSM と違い、軌道もコストも面もパラメータの**純関数**で、
// 時間発展する状態は持たない。`step(dt)` が動かすのは再生ヘッド t（パネルが軌道の上に打つマーカー）と
// 通番 seq だけで、t は T を法として**巻き戻る**（= 執行プログラムを繰り返し再生する）。したがって
//   * `SetParam` は FDM と同じく**その場で**全部を引き直す（次の step を待たない）。ただし FDM と違って
//     seq は巻き戻さない: 掃引の途中ではないので「やり直し」は起きていない。R10 により一時停止中でも
//     Snapshot と面が 1 組 publish され、次のフレームで絵が変わる。
//   * `Reset` は t と seq を 0 に戻すだけ（パラメータは保持する。他シーンと同じ規約）。乱数は無い。
//
// 【λ の梯子と面の λ 軸】どちらも**現在の λ を中心に ±kLadderDecades 桁**の対数等間隔で張る
// （梯子 8 本、面 32 本）。固定レンジにすると λ を動かしても面が変わらず、「λ がプログラムの形を
// 決める」というこのシーンの主題が 3D 側に出ない。フロンティアだけは逆に**固定レンジ**
// [kMinLambda, kMaxLambda] で張る: 曲線は λ を動かしても動かず、その上をマーカー（現在の λ）が
// 滑るのが効率的フロンティアの読み方だから。
//
// 【格子の対応】n_steps は kSteps = kN − 1 = 63 に固定してある。`ac_trajectory` が書く点数は
// n_steps + 1 なので、Snapshot の 1 行と面の 1 行がちょうど「コアの出力そのもの」になり
// （ACSCENE-01 はこれをビット一致で見る）、面の t 軸は執行の区間端 t_j = j·T/63 に一致する。
//
// 【κT の大きさ】コアの sinh 比はオーバーフローしない形（e^{a−b}·expm1 比）で書かれているので、
// κT がいくら大きくても軌道は有限（十分大きければ内点が 0 へアンダーフローするだけ）。それでも
// スライダーのクランプ域は λ ≤ 1e-3・σ ≤ 2・T ≤ 10・η ≥ 1e-6・γ ≤ 2.5e-6 に切ってある: κT が数百に
// なると軌道は「初日に全部売る」に潰れて絵として読めなくなるからで、この域なら梯子の上端
// （λ·10^1.5 = 3.16e-2）でも κT = 2N·asinh(κ̃T/2N) ≈ 126·asinh(31.5) ≈ 522 までしか行かず、内点は
// 潰れても単調非増加のまま残る。面を埋めるときの非有限値ガード（0 に落とす）は保険として残す:
// メッシュは NaN を渡されると法線が壊れる（`viz/gl/surface_mesh.hpp` の set_z は Debug で assert）。
// 同じクランプ域では γτ/2 ≤ 2.0e-7 < η なので η~ = η − γτ/2 の床（kAcMinEtaTilde）も効かない。

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/core/exec/almgren_chriss.hpp"

namespace quantviz::scenes {

/// リングに載る方。面（λ を細かく刻んだ格子）は Surface が運ぶ。
struct ExecSnapshot {
    static constexpr std::size_t kL        = 8;   ///< 重ね描きする λ の本数（梯子）
    static constexpr std::size_t kN        = 64;  ///< 軌道 1 本の点数（= n_steps + 1）
    static constexpr std::size_t kFrontier = 32;  ///< フロンティアの λ 格子点数

    double t     = 0.0;  ///< 再生ヘッド（[0, T) を巻き戻しながら進む。面には効かない）
    double x_now = 0.0;  ///< 現在の λ の軌道を t で線形補間した残量（マーカー・テレメトリ用）
    double kappa = 0.0;  ///< 現在の λ の κ（= ac_kappa(params)）

    core::AcParams params{};  ///< 現在の（クランプ済み）パラメータ = 参照値。n_steps は常に kSteps
    core::AcCost   cost{};    ///< 現在の λ の E[C], V[C]

    std::array<double, kN>      trajectory{};  ///< 現在の λ の残量 x_j（パネルが太線で描く）
    std::array<double, kL>      lambdas{};     ///< 梯子（昇順・対数等間隔、現在の λ を挟む）
    std::array<core::AcCost, kL> costs{};      ///< 梯子 1 本ごとの E[C], V[C]
    /// 梯子の軌道。**row-major [l][n]** = `trajectories[l * kN + n]`（1 本が連続する）。
    std::array<double, kL * kN> trajectories{};

    std::array<double, kFrontier>       frontier_lambdas{};  ///< 固定レンジの λ 格子（昇順）
    std::array<core::AcCost, kFrontier> frontier{};          ///< その E[C], V[C]

    std::uint64_t seq = 0;  ///< ステップ通番（0 = 未ステップ／Reset 直後）
};

static_assert(std::is_trivially_copyable_v<ExecSnapshot>);
static_assert(std::is_standard_layout_v<ExecSnapshot>);
static_assert(sizeof(ExecSnapshot) <= 8192, "Snapshot 上限（docs/02『Snapshot のサイズ方針』）");

/// 3D 面。`bridge::TripleBuffer` の使い回しスロットに書くので `surface()` は毎回全フィールドを書く。
struct ExecSurface {
    static constexpr std::size_t kT = 64;  ///< t 方向の点数（= 執行の区間端 t_j、昇順）
    static constexpr std::size_t kL = 32;  ///< λ 方向の本数（対数等間隔、昇順）

    std::array<float, kT>      ts{};       ///< t 軸 [0, T]（昇順・等間隔）
    std::array<float, kL>      lambdas{};  ///< λ 軸（昇順・対数等間隔。メッシュには log10 で渡す）
    /// 残量 x(t; λ)。**row-major [iL][iT]** = `x[iL * kT + iT]`。`gl::SurfaceMesh`（index = iy·n_x + ix）
    /// と一致する: x = t、y = λ。非有限値は 0 に落としてある（メッシュに NaN を渡さない）。
    std::array<float, kT * kL> x{};
};

static_assert(std::is_trivially_copyable_v<ExecSurface>);
static_assert(std::is_standard_layout_v<ExecSurface>);
static_assert(sizeof(ExecSurface) <= 16384, "Surface 上限（TripleBuffer は 3 枚持つ）");

class ExecModel {
public:
    using Snapshot = ExecSnapshot;
    using Surface  = ExecSurface;

    /// 区間数 N。`ac_trajectory` の出力（N + 1 点）が Snapshot / Surface の 1 行に丁度収まる値に固定。
    static constexpr std::size_t kSteps = ExecSnapshot::kN - 1;

    /// 0 は予約（`Command` の既定値と衝突させない）。
    enum Param : std::uint32_t {
        kLambda = 1,  ///< λ（リスク回避。0 で TWAP、大きいほど前倒し）
        kEta    = 2,  ///< η（一時インパクト）
        kGamma  = 3,  ///< γ（恒久インパクト）
        kSigma  = 4,  ///< σ（価格ボラティリティ）
        kT      = 5,  ///< T（執行期間）
    };

    // クランプ域（ヘッダ冒頭「κT の上限」がこの組み合わせで成り立つ）。
    static constexpr double kMinLambda = 1e-9;
    static constexpr double kMaxLambda = 1e-3;
    static constexpr double kMinEta    = 1e-6;
    static constexpr double kMaxEta    = 2.5e-5;
    static constexpr double kMinGamma  = 0.0;
    static constexpr double kMaxGamma  = 2.5e-6;
    static constexpr double kMinSigma  = 0.05;
    static constexpr double kMaxSigma  = 2.0;
    static constexpr double kMinT      = 0.5;
    static constexpr double kMaxT      = 10.0;

    /// 梯子と面の λ 軸が現在の λ の周りに張る片側の桁数。
    static constexpr double kLadderDecades = 1.5;

    struct Config {
        /// 初期パラメータ。既定は Almgren–Chriss (2000) §3 の数値例（X = 1e6 株、T = 5 日、
        /// σ = 0.95 $/√日、η = 2.5e-6、γ = 2.5e-7、ε = 0.0625 $、λ = 2e-6）。n_steps は kSteps に固定。
        core::AcParams params{};
    };

    // GCC のバグ回避: 既定メンバ初期化子を持つ入れ子 Config を既定引数にしない（CLAUDE.md）。
    ExecModel() : ExecModel(Config{}) {}

    explicit ExecModel(Config cfg) : params_(clamp_all(cfg.params)) { recompute(); }

    /// 1 ステップ = 再生ヘッドを dt だけ進める（T を法として巻き戻る）。軌道も面も動かない。
    void step(double dt) {
        if (std::isfinite(dt) && dt > 0.0) {
            t_ += dt;
            if (t_ >= params_.T) t_ = std::fmod(t_, params_.T);
        }
        ++seq_;
        publish_time();
    }

    [[nodiscard]] Snapshot snapshot() const noexcept { return snap_; }

    /// 現在の面を s に書く。TripleBuffer の使い回しスロットなので**全フィールド**を書く
    /// （`s = cache_` の丸ごとコピー）。パラメータが変わっていればここで 1 回だけ作り直す。
    void surface(Surface& s) const noexcept {
        if (dirty_) {
            rebuild_surface();
            dirty_ = false;
        }
        s = cache_;
    }

    void apply(const bridge::Command& c) {
        switch (c.type) {
            case bridge::CommandType::SetParam: apply_param(c.param_id, c.value); break;
            case bridge::CommandType::Reset:    reset(c.seed); break;
            default:                            break;  // 時計系は Runner が処理済み
        }
    }

    /// 再生ヘッドと通番を巻き戻す。パラメータは保持する（他シーンの Reset と同じ規約）。
    /// 軌道・コスト・面は t に依らないので作り直さない。乱数が無いので seed は使わない。
    void reset(std::uint64_t /*seed*/ = 0) noexcept {
        t_   = 0.0;
        seq_ = 0;
        publish_time();
    }

    // ------------------------------------------------------------------ テスト・パネル用アクセサ
    [[nodiscard]] const core::AcParams& params() const noexcept { return params_; }
    [[nodiscard]] double                time() const noexcept { return t_; }
    /// `recompute()`（軌道・コスト・フロンティアの引き直し）を走らせた回数。構築時の 1 回から始まる。
    /// 「同じ値の SetParam は仕事をしない」を**結果**ではなく**呼ばれた回数**で見るために露出する。
    [[nodiscard]] std::uint64_t recompute_count() const noexcept { return recomputes_; }
    /// `rebuild_surface()` を走らせた回数（dirty フラグの遅延が効いているかを見る）。0 から始まる。
    [[nodiscard]] std::uint64_t surface_builds() const noexcept { return surface_builds_; }

private:
    // ------------------------------------------------------------------ 入力の丸め（例外なし）
    /// 有限値は [lo, hi] に丸め、NaN / ∞ は**拒否**して現在値を返す（FDM シーンと同じ流儀:
    /// UI の事故で λ や σ が下限へ飛ぶと別のプログラムになってしまう）。
    static double clamp_param(double v, double lo, double hi, double current) noexcept {
        if (!std::isfinite(v)) return current;
        return v < lo ? lo : (v > hi ? hi : v);
    }

    /// Config 由来のパラメータを定義域へ。X と ε はスライダーを持たない（Config で決め打ち）ので、
    /// 金融的に意味のある範囲だけ入れて `ac_sanitize` に委ねる。
    static core::AcParams clamp_all(core::AcParams p) noexcept {
        constexpr core::AcParams d{};
        p.X       = clamp_param(p.X, 1.0, 1e9, d.X);
        p.T       = clamp_param(p.T, kMinT, kMaxT, d.T);
        p.sigma   = clamp_param(p.sigma, kMinSigma, kMaxSigma, d.sigma);
        p.eta     = clamp_param(p.eta, kMinEta, kMaxEta, d.eta);
        p.gamma   = clamp_param(p.gamma, kMinGamma, kMaxGamma, d.gamma);
        p.epsilon = clamp_param(p.epsilon, 0.0, 10.0, d.epsilon);
        p.lambda  = clamp_param(p.lambda, kMinLambda, kMaxLambda, d.lambda);
        p.n_steps = kSteps;  // 格子は固定（ヘッダ冒頭「格子の対応」）
        return p;
    }

    /// 値が動いたときだけ引き直す（同じ値の再送で 2000 点以上を舐め直さない）。
    /// ここに来る値はクランプ済みで NaN は無いので == で判定してよい。
    void apply_param(std::uint32_t id, double v) noexcept {
        core::AcParams next = params_;
        switch (id) {
            case kLambda: next.lambda = clamp_param(v, kMinLambda, kMaxLambda, params_.lambda); break;
            case kEta:    next.eta = clamp_param(v, kMinEta, kMaxEta, params_.eta); break;
            case kGamma:  next.gamma = clamp_param(v, kMinGamma, kMaxGamma, params_.gamma); break;
            case kSigma:  next.sigma = clamp_param(v, kMinSigma, kMaxSigma, params_.sigma); break;
            case kT:      next.T = clamp_param(v, kMinT, kMaxT, params_.T); break;
            default:      return;  // 0 と未知の param_id は無視
        }
        if (same(next, params_)) return;
        params_ = next;
        recompute();
    }

    static bool same(const core::AcParams& a, const core::AcParams& b) noexcept {
        return a.X == b.X && a.T == b.T && a.sigma == b.sigma && a.eta == b.eta && a.gamma == b.gamma &&
               a.epsilon == b.epsilon && a.lambda == b.lambda && a.n_steps == b.n_steps;
    }

    // ------------------------------------------------------------------ 引き直し
    /// λ を差し替えたパラメータ（梯子・面・フロンティアの各点で使う）。
    [[nodiscard]] core::AcParams with_lambda(double lambda) const noexcept {
        core::AcParams p = params_;
        p.lambda         = lambda;
        return p;
    }

    /// 現在の λ を中心に ±kLadderDecades 桁を n 等分した i 番目（昇順・対数等間隔）。
    [[nodiscard]] double ladder_lambda(std::size_t i, std::size_t n) const noexcept {
        const double u = static_cast<double>(i) / static_cast<double>(n - 1);  // [0, 1]
        return params_.lambda * std::pow(10.0, kLadderDecades * (2.0 * u - 1.0));
    }

    /// フロンティアの λ 格子（固定レンジ [kMinLambda, kMaxLambda] の対数等間隔）。
    static double frontier_lambda(std::size_t i) noexcept {
        const double u = static_cast<double>(i) / static_cast<double>(ExecSnapshot::kFrontier - 1);
        return kMinLambda * std::pow(kMaxLambda / kMinLambda, u);
    }

    /// パラメータが変わったときだけ走る（確保も例外も無い: 書き先は固定長）。
    void recompute() noexcept {
        ++recomputes_;
        // T が縮んだら再生ヘッドを新しい区間へ畳む（[0, T) の外に置かない）。
        if (t_ >= params_.T) t_ = std::fmod(t_, params_.T);

        snap_.params = params_;
        snap_.kappa  = core::ac_kappa(params_);

        for (std::size_t l = 0; l < ExecSnapshot::kL; ++l) {
            const double lambda = ladder_lambda(l, ExecSnapshot::kL);
            snap_.lambdas[l]    = lambda;
            const core::AcParams p = with_lambda(lambda);
            (void)core::ac_trajectory(
                p, std::span<double>(snap_.trajectories.data() + l * ExecSnapshot::kN, ExecSnapshot::kN));
            snap_.costs[l] = core::ac_cost(p);
        }

        (void)core::ac_trajectory(params_, std::span<double>(snap_.trajectory));
        snap_.cost = core::ac_cost(params_);

        for (std::size_t i = 0; i < ExecSnapshot::kFrontier; ++i)
            snap_.frontier_lambdas[i] = frontier_lambda(i);
        (void)core::ac_frontier(params_, std::span<const double>(snap_.frontier_lambdas),
                                std::span<core::AcCost>(snap_.frontier));

        dirty_ = true;  // 面は次の surface() で 1 回だけ作り直す
        publish_time();
    }

    /// 時刻まわりだけを Snapshot へ写す（step / reset / recompute の共通尾）。
    void publish_time() noexcept {
        snap_.t     = t_;
        snap_.x_now = inventory_at(t_);
        snap_.seq   = seq_;
    }

    /// 現在の λ の軌道を t で線形補間した残量。t は [0, T] に丸めて読む。
    [[nodiscard]] double inventory_at(double t) const noexcept {
        const double tau = params_.T / static_cast<double>(kSteps);
        const double u   = (t > 0.0 ? t : 0.0) / tau;
        if (!(u < static_cast<double>(kSteps))) return snap_.trajectory[ExecSnapshot::kN - 1];
        const double      fl = std::floor(u);
        const std::size_t j  = static_cast<std::size_t>(fl);
        const double      w  = u - fl;
        return snap_.trajectory[j] * (1.0 - w) + snap_.trajectory[j + 1] * w;
    }

    /// 面を作り直す。const な surface() の中から 1 回だけ呼ばれる（計算スレッド専用）。
    void rebuild_surface() const noexcept {
        ++surface_builds_;
        const double tau = params_.T / static_cast<double>(kSteps);
        for (std::size_t i = 0; i < ExecSurface::kT; ++i)
            cache_.ts[i] = static_cast<float>(static_cast<double>(i) * tau);

        std::array<double, ExecSurface::kT> row{};  // スタックの作業領域（512 B、確保なし）
        for (std::size_t l = 0; l < ExecSurface::kL; ++l) {
            const double lambda = ladder_lambda(l, ExecSurface::kL);
            cache_.lambdas[l]   = static_cast<float>(lambda);
            (void)core::ac_trajectory(with_lambda(lambda), std::span<double>(row));
            float* const dst = cache_.x.data() + l * ExecSurface::kT;
            for (std::size_t i = 0; i < ExecSurface::kT; ++i) {
                // 非有限値はメッシュの法線を壊すので 0 に落とす（クランプ域では出ないはずの保険）。
                dst[i] = std::isfinite(row[i]) ? static_cast<float>(row[i]) : 0.f;
            }
        }
    }

    core::AcParams params_{};
    double         t_   = 0.0;
    std::uint64_t  seq_ = 0;
    Snapshot       snap_{};

    /// 面のキャッシュと「作り直しが要るか」。触るのは計算スレッドだけで、描画スレッドは
    /// `TripleBuffer` 越しのコピーしか見ない（`vol_surface_model.hpp` と同じ約束）。
    mutable Surface cache_{};
    mutable bool    dirty_ = true;

    std::uint64_t         recomputes_     = 0;  ///< recompute() を走らせた回数（構築時の 1 回を含む）
    mutable std::uint64_t surface_builds_ = 0;  ///< rebuild_surface() を走らせた回数
};

static_assert(bridge::Model<ExecModel>, "ExecModel must satisfy the Model contract");
static_assert(bridge::SurfaceModel<ExecModel>,
              "ExecModel must satisfy the SurfaceModel contract (the 3D panel needs the channel)");

}  // namespace quantviz::scenes
