#pragma once
// scenes/kalman_pair_model.hpp — シーン 4「Kalman pair」: 共和分ペアの動的ヘッジ比推定
//
// 生成（真のデータ生成過程）:
//   x_t  : μ = 0 の GBM を厳密離散化した価格（E[x_t] = x_0 のマルチンゲール、年率ボラ `x_vol`）
//   β_t  : `kBetaTrue` を中心とする平均回帰付きランダムウォーク
//            β_t = β_{t−1} + κ(β_center − β_{t−1}) + state_noise · Z
//          κ = 0 なら純粋なランダムウォーク。κ > 0 にしておくと β が中心から離れ続けず、
//          UI で kBetaTrue を動かしたときに真値が追従するのが見える。
//   y_t  = β_t x_t + obs_noise · Z
//
// 推定: 状態を β、観測行列を H = [x_t]、F = 1、Q = state_noise²、R = obs_noise² とした
// `core::Kalman<1,1>`。フィルタは β を純粋なランダムウォークとみなす（κ の分だけ軽く誤特定
// だが、局所レベルモデルとしての追従性は保たれる）。
//
// Snapshot に載せるのは「今の 1 点」だけ（時系列を作るのは描画側の責務）。乱数は Config の
// seed から作った 1 本の `core::Rng` に集約し、同じ seed + 同じコマンド列なら bit 一致する。

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/core/math/mat.hpp"
#include "quantviz/core/rng.hpp"
#include "quantviz/core/stats/kalman.hpp"

namespace quantviz::scenes {

struct KalmanPairSnapshot {
    double        t           = 0.0;  ///< sim 時刻（年）
    double        x           = 0.0;  ///< 説明変数側の価格
    double        y           = 0.0;  ///< 被説明変数側の価格 y = β_t x + ε
    double        beta_true   = 0.0;  ///< 真のヘッジ比 β_t（参照線用）
    double        beta_hat    = 0.0;  ///< カルマン推定 β̂_t（事後平均）
    double        beta_var    = 0.0;  ///< その事後分散 P_t（±2√P の帯を描くため）
    double        spread      = 0.0;  ///< ヘッジ後スプレッド y_t − β̂_t x_t（事後残差）
    double        innovation  = 0.0;  ///< イノベーション y_t − β̂⁻_t x_t（事前残差）
    double        obs_noise   = 0.0;  ///< 現在の観測ノイズ σ_ε（参照）
    double        state_noise = 0.0;  ///< 現在の状態ノイズ σ_β（参照）
    std::uint64_t seq         = 0;    ///< ステップ通番（0 = 未ステップ／Reset 直後）
    std::uint32_t skipped     = 0;    ///< S が特異で観測更新を捨てた回数（0 でない = 入力が壊れている）
};
static_assert(std::is_trivially_copyable_v<KalmanPairSnapshot>);
static_assert(std::is_default_constructible_v<KalmanPairSnapshot>);
static_assert(sizeof(KalmanPairSnapshot) <= 128, "Snapshot must stay within two cache lines");

class KalmanPairModel {
public:
    using Snapshot = KalmanPairSnapshot;

    enum Param : std::uint32_t {
        kObsNoise   = 1,
        kStateNoise = 2,
        kBetaTrue   = 3,
    };

    struct Config {
        double        x0             = 100.0;  ///< x の初期値
        double        x_vol          = 0.20;   ///< x の年率ボラティリティ
        double        beta_center    = 1.20;   ///< β の中心（kBetaTrue）
        double        beta_reversion = 0.002;  ///< κ: 1 ステップあたりの平均回帰係数
        double        state_noise    = 0.003;  ///< σ_β: β の 1 ステップ標準偏差（dt 非依存）
        double        obs_noise      = 2.0;    ///< σ_ε: 観測ノイズ標準偏差
        double        beta_prior     = 1.0;    ///< フィルタの初期 β̂（真値とは別に置く）
        double        prior_var      = 0.25;   ///< フィルタの初期 P
        std::uint64_t seed           = 42;
    };

    /// GCC の「既定メンバ初期化子を持つ入れ子 Config を既定引数にする」バグ回避のため委譲する。
    KalmanPairModel() : KalmanPairModel(Config{}) {}

    explicit KalmanPairModel(Config cfg)
        : cfg_(sanitized(cfg)),
          rng_(cfg_.seed),
          kf_(core::Mat<1, 1>{{cfg_.beta_prior}}, core::Mat<1, 1>{{cfg_.prior_var}}),
          x_(cfg_.x0),
          beta_(cfg_.beta_center),
          y_(cfg_.beta_center * cfg_.x0) {}

    void step(double dt) {
        // --- 生成 ------------------------------------------------------------------
        const double sqrt_dt = std::sqrt(dt);
        x_ *= std::exp(-0.5 * cfg_.x_vol * cfg_.x_vol * dt + cfg_.x_vol * sqrt_dt * rng_.normal());
        beta_ += cfg_.beta_reversion * (cfg_.beta_center - beta_) + cfg_.state_noise * rng_.normal();
        y_ = beta_ * x_ + cfg_.obs_noise * rng_.normal();

        // --- 推定（F = 1, H = [x_t] は時変, Q = σ_β², R = σ_ε²） --------------------
        const core::Mat<1, 1> q{{cfg_.state_noise * cfg_.state_noise}};
        const core::Mat<1, 1> r{{cfg_.obs_noise * cfg_.obs_noise}};
        const core::Mat<1, 1> h{{x_}}, z{{y_}};

        kf_.predict(core::Mat<1, 1>::identity(), q);

        // core::Kalman::update は S = H P⁻ Hᵀ + R が特異（非有限 または 0）なら状態も共分散も
        // 変えずに戻る。1×1 では S はスカラなので、同じ判定をここでも行って回数を数える
        // （Snapshot の `skipped` が「パラメータが壊れている」ことの唯一の可視化になる）。
        const double s_scalar = x_ * kf_.cov()(0, 0) * x_ + r(0, 0);
        if (!(std::isfinite(s_scalar) && s_scalar != 0.0)) ++skipped_updates_;

        innovation_ = kf_.update(h, z, r)(0, 0);
        spread_     = y_ - kf_.state()(0, 0) * x_;  // 事後残差（トレード対象のスプレッド）

        t_ += dt;
        ++seq_;
    }

    Snapshot snapshot() const noexcept {
        Snapshot s;
        s.t           = t_;
        s.x           = x_;
        s.y           = y_;
        s.beta_true   = beta_;
        s.beta_hat    = kf_.state()(0, 0);
        s.beta_var    = kf_.cov()(0, 0);
        s.spread      = spread_;
        s.innovation  = innovation_;
        s.obs_noise   = cfg_.obs_noise;
        s.state_noise = cfg_.state_noise;
        s.seq         = seq_;
        s.skipped     = skipped_updates_;
        return s;
    }

    void apply(const bridge::Command& c) {
        switch (c.type) {
            case bridge::CommandType::SetParam:
                switch (c.param_id) {
                    case kObsNoise:   set_obs_noise(c.value); break;
                    case kStateNoise: set_state_noise(c.value); break;
                    case kBetaTrue:   set_beta_true(c.value); break;
                    default:          break;  // 未知の param_id は無視
                }
                break;
            case bridge::CommandType::Reset: reset(c.seed != 0 ? c.seed : cfg_.seed); break;
            default:                         break;  // 時計系は Runner が処理済み
        }
    }

    /// パス・フィルタ・統計を初期状態へ。ノイズ / β 中心などのパラメータは保持する。
    void reset(std::uint64_t seed) {
        cfg_.seed = seed;
        rng_.reseed(seed);
        kf_.reset(core::Mat<1, 1>{{cfg_.beta_prior}}, core::Mat<1, 1>{{cfg_.prior_var}});
        x_          = cfg_.x0;
        beta_       = cfg_.beta_center;
        y_          = cfg_.beta_center * cfg_.x0;
        t_          = 0.0;
        spread_          = 0.0;
        innovation_      = 0.0;
        seq_             = 0;
        skipped_updates_ = 0;
    }

    /// R = 0 も R = NaN/∞ も S = x²P⁻ + R を特異にし、`core::Kalman::update` が観測更新を
    /// まるごと捨てる（しかも Reset は cfg_ を保持するので二度と復帰しない）。
    /// NaN はどの比較でも false になるため `std::max(v, lo)` では素通りする。下限を第 1 引数に
    /// 置いたうえで非有限を明示的に弾く。
    void set_obs_noise(double v) noexcept { cfg_.obs_noise = floored(v, kMinObsNoise); }
    /// Q < 0 は共分散を壊すので 0 でクランプする（Q = 0 は「β は定数」という正当なモデル）。
    void set_state_noise(double v) noexcept { cfg_.state_noise = floored(v, 0.0); }
    /// β の中心。非有限は意味を持たないので黙って捨て、直前の値を残す。
    void set_beta_true(double v) noexcept {
        if (std::isfinite(v)) cfg_.beta_center = v;
    }

private:
    static constexpr double kMinObsNoise = 1e-9;
    static constexpr double kFallbackX0  = 100.0;  ///< Config の x0 が非有限／非正のときの代替

    /// 非有限なら lo、そうでなければ max(lo, v)。下限を第 1 引数に置くのが要点（NaN 対策）。
    static double floored(double v, double lo) noexcept {
        return std::isfinite(v) ? std::max(lo, v) : lo;
    }
    /// 非有限なら fallback、そうでなければ [lo, hi] に丸める。
    static double bounded(double v, double lo, double hi, double fallback) noexcept {
        if (!std::isfinite(v)) return fallback;
        return v < lo ? lo : (v > hi ? hi : v);
    }

    /// 構築時に Config を安全な範囲へ。UI 以外（テスト・将来の設定ファイル）からの経路も塞ぐ。
    static Config sanitized(Config c) noexcept {
        c.obs_noise      = floored(c.obs_noise, kMinObsNoise);
        c.state_noise    = floored(c.state_noise, 0.0);
        c.x_vol          = floored(c.x_vol, 0.0);
        c.prior_var      = floored(c.prior_var, 0.0);
        c.beta_center    = std::isfinite(c.beta_center) ? c.beta_center : 1.0;
        c.beta_prior     = std::isfinite(c.beta_prior) ? c.beta_prior : 0.0;
        c.beta_reversion = bounded(c.beta_reversion, 0.0, 1.0, 0.0);  // κ > 1 は振動、κ < 0 は発散
        if (!(c.x0 > 0.0) || !std::isfinite(c.x0)) c.x0 = kFallbackX0;
        return c;
    }

    Config             cfg_;
    core::Rng          rng_;
    core::Kalman<1, 1> kf_;

    double        x_          = 0.0;
    double        beta_       = 0.0;
    double        y_          = 0.0;
    double        t_          = 0.0;
    double        spread_          = 0.0;
    double        innovation_      = 0.0;
    std::uint64_t seq_             = 0;
    std::uint32_t skipped_updates_ = 0;
};

static_assert(bridge::Model<KalmanPairModel>, "KalmanPairModel must satisfy the Model contract");

}  // namespace quantviz::scenes
