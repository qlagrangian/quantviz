#pragma once
// scenes/vol_surface_model.hpp — シーン「Vol surface」: SSVI のパラメトリックなボラ面
//
// 責務: `core::ssvi_implied_vol`（Gatheral–Jacquier SSVI）を (k, T) の 64×32 格子で評価し、
// 面チャネル（`bridge::SurfaceModel`）に流す。数値そのものはコアの純関数で、ここが持つのは
// 「どの格子で評価するか」「UI のパラメータをどうクランプして反映するか」だけ。
//
// このシーンに時間発展は無い: 面は (k, T) の関数であって sim 時刻 t の関数ではないので、
// `step(dt)` は t と通番を進めるだけで面には触れない（時計・Pause/Step/Reset が他シーンと同じ形で
// 効くこと自体には意味があるので、時計は素通しで残す）。面が変わるのはパラメータを変えたときだけ。
//
// 面の再計算は dirty フラグで遅延させる:
//   * `apply(SetParam)` は値をクランプして保存し、実際に変わったときだけ dirty を立てる
//     （UI が同じ値を送り直しても 2048 点を舐め直さない）
//   * 実際の再構築は次の `surface()` の中で 1 回だけ行う。1 tick に 4 本のスライダーが届いても
//     再構築は 1 回で済み、面を誰も見ない（Snapshot だけ読む）使い方なら 1 回も走らない
//   * `Reset` は t と通番を巻き戻し、面を作り直す（パラメータは保持する）
// そのため cache_ / dirty_ は mutable。呼ぶのは計算スレッドだけ（`Runner::publish_surface`）で、
// 描画スレッドは `TripleBuffer` 越しのコピーしか見ない。
//
// Snapshot と Surface の分担: パラメータ・時刻・軸のレンジのような「小さくて毎ステップ欲しい」値は
// Snapshot（リング）、格子は Surface（TripleBuffer で最新 1 枚）。

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/core/pricing/vol_surface.hpp"

namespace quantviz::scenes {

/// リングに載る小さい方。面（格子）は Surface が運ぶ。
struct VolSurfaceSnapshot {
    double           t = 0.0;  ///< sim 時刻（年）。面には効かない（時計が動いている証拠）
    core::SsviParams params{};  ///< 現在の（クランプ済み）SSVI パラメータ = 参照値
    double           k_min = 0.0;  ///< k 軸の下端（log-moneyness）
    double           k_max = 0.0;
    double           t_min = 0.0;  ///< T 軸の下端（年）
    double           t_max = 0.0;
    std::uint64_t    seq   = 0;    ///< ステップ通番（0 = 未ステップ／Reset 直後）
};

static_assert(std::is_trivially_copyable_v<VolSurfaceSnapshot>);
static_assert(std::is_standard_layout_v<VolSurfaceSnapshot>);
static_assert(sizeof(VolSurfaceSnapshot) <= 128, "Snapshot は 128 B 以内（docs/01_design.md §4）");

/// TripleBuffer で最新 1 枚だけ渡す格子。`gl::SurfaceMesh` にそのまま載せられる float 配列。
struct VolSurfaceSurface {
    static constexpr std::size_t kK = 64;  ///< k（log-moneyness）方向の格子点数
    static constexpr std::size_t kT = 32;  ///< T（満期）方向の格子点数

    std::array<float, kK>      ks{};  ///< k 軸（昇順・等間隔）
    std::array<float, kT>      ts{};  ///< T 軸（昇順・等間隔、年）
    /// インプライド・ボラ。**row-major [iT][iK]** = `iv[iT * kK + iK]`。
    /// 行（固定 T）が連続するので、スマイル 1 本は `&iv[iT * kK]` から kK 個で取れる。
    /// `gl::SurfaceMesh`（index = iy * n_x + ix）とも一致する: x = k, y = T。
    std::array<float, kK * kT> iv{};
};

static_assert(std::is_trivially_copyable_v<VolSurfaceSurface>);
static_assert(std::is_standard_layout_v<VolSurfaceSurface>);

class VolSurfaceModel {
public:
    using Snapshot = VolSurfaceSnapshot;
    using Surface  = VolSurfaceSurface;

    /// 0 は予約（`Command` の既定値と衝突させない）。
    enum Param : std::uint32_t {
        kSigmaAtm = 1,  ///< σ_atm（ATM ボラ）
        kRho      = 2,  ///< ρ（スキュー）
        kEta      = 3,  ///< η（翼の水準）
        kGamma    = 4,  ///< γ（φ の減衰指数）
    };

    struct Config {
        core::SsviParams params{};        ///< 初期パラメータ（ssvi_clamp を通してから使う）
        double           k_min = -1.0;    ///< k ∈ [−1, 1] ≒ 現物比 0.37〜2.7 倍
        double           k_max = 1.0;
        double           t_min = 0.05;    ///< T ∈ [0.05, 3] 年（≒ 2 週間〜3 年）
        double           t_max = 3.0;
    };

    // GCC のバグ回避: 既定メンバ初期化子を持つ入れ子 Config を既定引数にしない（CLAUDE.md）。
    VolSurfaceModel() : VolSurfaceModel(Config{}) {}

    explicit VolSurfaceModel(Config cfg) : params_(core::ssvi_clamp(cfg.params)) {
        sanitize(cfg);
        k_min_ = cfg.k_min;
        k_max_ = cfg.k_max;
        t_min_ = cfg.t_min;
        t_max_ = cfg.t_max;
        // 面は最初の surface() で作る（dirty_ の初期値が true）。
    }

    /// 1 ステップ: 時刻と通番を進めるだけ。面は (k, T) の関数なので t に依らない。
    void step(double dt) {
        if (std::isfinite(dt)) t_ += dt;
        ++seq_;
    }

    [[nodiscard]] Snapshot snapshot() const noexcept {
        Snapshot s;
        s.t      = t_;
        s.params = params_;
        s.k_min  = k_min_;
        s.k_max  = k_max_;
        s.t_min  = t_min_;
        s.t_max  = t_max_;
        s.seq    = seq_;
        return s;
    }

    /// 現在の面を out へ書く。out は TripleBuffer の使い回しスロット（＝ 2 世代前の面）なので、
    /// 全フィールドを書く（丸ごとコピーする）。パラメータが変わっていれば、ここで 1 回だけ作り直す。
    void surface(Surface& out) const noexcept {
        if (dirty_) {
            rebuild();
            dirty_ = false;
        }
        out = cache_;
    }

    void apply(const bridge::Command& c) {
        switch (c.type) {
            case bridge::CommandType::SetParam: {
                core::SsviParams p = params_;
                switch (c.param_id) {
                    case kSigmaAtm: p.sigma_atm = c.value; break;
                    case kRho:      p.rho = c.value; break;
                    case kEta:      p.eta = c.value; break;
                    case kGamma:    p.gamma = c.value; break;
                    default:        return;  // 0 と未知の param_id は無視
                }
                set_params(core::ssvi_clamp(p));  // UI 由来の値は必ずクランプを通す
                break;
            }
            case bridge::CommandType::Reset: reset(c.seed); break;
            default:                         break;  // 時計系は Runner が処理済み
        }
    }

    /// 時刻と通番を巻き戻し、面を作り直す。パラメータは保持する（他シーンの Reset と同じ規約）。
    /// このモデルに乱数は無いので seed は使わない。
    void reset(std::uint64_t /*seed*/) noexcept {
        t_     = 0.0;
        seq_   = 0;
        dirty_ = true;
    }

    // テスト・パネル用の読み取りアクセサ（Snapshot にも同じ値が載る）
    [[nodiscard]] double time() const noexcept { return t_; }
    [[nodiscard]] const core::SsviParams& params() const noexcept { return params_; }

private:
    /// 軸レンジの健全化: 非有限・逆順・T ≤ 0 は既定へ戻す（T は w/T の割り算に使う）。
    static void sanitize(Config& cfg) noexcept {
        const Config d{};
        if (!std::isfinite(cfg.k_min) || !std::isfinite(cfg.k_max) || !(cfg.k_min < cfg.k_max)) {
            cfg.k_min = d.k_min;
            cfg.k_max = d.k_max;
        }
        if (!std::isfinite(cfg.t_min) || !std::isfinite(cfg.t_max) || !(cfg.t_min > 0.0) ||
            !(cfg.t_min < cfg.t_max)) {
            cfg.t_min = d.t_min;
            cfg.t_max = d.t_max;
        }
    }

    /// 実際に変わったときだけ dirty を立てる（同じ値の再送で 2048 点を舐め直さない）。
    /// ここに来る p はクランプ済みなので NaN は無く、== の比較で判定してよい。
    void set_params(const core::SsviParams& p) noexcept {
        if (p.sigma_atm == params_.sigma_atm && p.rho == params_.rho && p.eta == params_.eta &&
            p.gamma == params_.gamma)
            return;
        params_ = p;
        dirty_  = true;
    }

    /// 軸と格子を作り直す。ヒープも例外も無い（cache_ は固定長）。
    ///
    /// iv は **float に丸めた後の軸の値**で評価する: パネルが描く x 座標（ks）と、その点の高さ
    /// （iv）が厳密に同じ (k, T) に対応するようにするため（テストの突き合わせもこの形で行う）。
    void rebuild() const noexcept {
        for (std::size_t ik = 0; ik < Surface::kK; ++ik) cache_.ks[ik] = static_cast<float>(axis_k(ik));
        for (std::size_t it = 0; it < Surface::kT; ++it) cache_.ts[it] = static_cast<float>(axis_t(it));

        for (std::size_t it = 0; it < Surface::kT; ++it) {
            const double T = static_cast<double>(cache_.ts[it]);
            for (std::size_t ik = 0; ik < Surface::kK; ++ik) {
                const double k = static_cast<double>(cache_.ks[ik]);
                cache_.iv[it * Surface::kK + ik] =
                    static_cast<float>(core::ssvi_implied_vol(k, T, params_));
            }
        }
    }

    [[nodiscard]] double axis_k(std::size_t i) const noexcept {
        const double u = static_cast<double>(i) / static_cast<double>(Surface::kK - 1);
        return k_min_ + (k_max_ - k_min_) * u;
    }
    [[nodiscard]] double axis_t(std::size_t j) const noexcept {
        const double u = static_cast<double>(j) / static_cast<double>(Surface::kT - 1);
        return t_min_ + (t_max_ - t_min_) * u;
    }

    double           t_   = 0.0;
    std::uint64_t    seq_ = 0;
    core::SsviParams params_{};
    double           k_min_ = 0.0;
    double           k_max_ = 0.0;
    double           t_min_ = 0.0;
    double           t_max_ = 0.0;

    /// 面のキャッシュと「作り直しが要るか」。const な surface() の中で 1 回だけ更新する
    /// （触るのは計算スレッドだけ。描画スレッドは TripleBuffer のコピーしか見ない）。
    mutable Surface cache_{};
    mutable bool    dirty_ = true;
};

static_assert(bridge::Model<VolSurfaceModel>, "VolSurfaceModel must satisfy the Model contract");
static_assert(bridge::SurfaceModel<VolSurfaceModel>,
              "VolSurfaceModel must satisfy the SurfaceModel contract (the 3D panel needs the channel)");

}  // namespace quantviz::scenes
