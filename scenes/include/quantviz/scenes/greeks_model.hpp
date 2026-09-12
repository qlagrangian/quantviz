#pragma once
// scenes/greeks_model.hpp — シーン 2「Greeks」: GBM で動くスポットに対する Black–Scholes の解析 Greeks
//
// core の `Gbm`（パス）と `bs_greeks` / `bs_price_strip`（価格付け）を 1 つの Model に束ね、描画が必要と
// する状態だけを固定長 POD の Snapshot として吐く。載せるのは 2 つの縮約図形だけ:
//
//   1. ストライク軸 K ∈ [S0(1-span), S0(1+span)] 上の Greeks ストリップ（kStrikes 点）
//   2. Γ(S, T) の格子（S ∈ [0.5 S0, 1.5 S0] × T ∈ (0, T_max]、ストライクは S0 に固定）
//
// どちらも固定サイズなので Snapshot は POD のままだが 16 KB 強あり、M0 の「128 B 目安」を超える
// （M1 の明示的な例外。M2 で triple buffer に移すまでは Runner の SnapCap を 256 に、
// steps_per_second を 100 に抑えて帯域を稼がない。契約テスト GREEKS-01 の上限は 32 KiB）。
//
// パラメータの反映タイミング: `apply()` は値をペンディングに積み、その場で `commit_pending()` +
// `publish()` を行う。つまり SetParam 直後の Snapshot は新しいパラメータで計算し直した配列と真値
// フィールドを持ち（自己整合的）、seq は進まない。Runner は一時停止中の SetParam の直後に Snapshot を
// 1 回だけ再送する（R10）ので、Step を押さなくてもスライダーやスポットショックの効果が画面に出る。
// step() の先頭でも `commit_pending()` を呼ぶが、apply で確定済みなら差分が無く何もしない。
// GREEKS-03 はこの契約を検査する。`reset()` も同様にその場で確定・張り直しを行う。
//
// σ について: このシーンには 2 つの σ がある。パスのボラティリティは `Config::gbm.sigma`（固定）、
// 価格付けのボラティリティは `Config::sigma`（`kSigma` で可変）。前者を動かさないので、スポット列は
// パラメータ操作に依らず seed だけで決まる（GREEKS-05 の dual-run）。
// オプションは既定でコール（`kOptionType`）。Put 版は Snapshot を 2 倍にしないため M1 では持たない。

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/core/models/gbm.hpp"
#include "quantviz/core/pricing/black_scholes.hpp"

namespace quantviz::scenes {

struct GreeksSnapshot {
    static constexpr std::size_t kStrikes = 64;  ///< ストライク軸の点数（昇順・等間隔）
    static constexpr std::size_t kGridS   = 48;  ///< Γ 格子の S 方向（M2 で 3D 化）
    static constexpr std::size_t kGridT   = 32;  ///< Γ 格子の T 方向

    double t     = 0.0;  ///< sim 時刻（年）
    double spot  = 0.0;  ///< 現在の S
    double r     = 0.0;  ///< 現在の無リスク金利（真値・参照線用）
    double sigma = 0.0;  ///< 現在の価格付けボラティリティ（真値）
    double T     = 0.0;  ///< 現在の満期までの年数（真値）

    std::array<double, kStrikes> strikes{};  ///< K 軸（固定。span 変更と Reset でのみ張り直す）
    std::array<double, kStrikes> price{};    ///< コール価格（bs_price_strip: ベクトル化経路）
    std::array<double, kStrikes> delta{};
    std::array<double, kStrikes> gamma{};
    std::array<double, kStrikes> vega{};   ///< ボラティリティ 1.0 あたり
    std::array<double, kStrikes> theta{};  ///< 年あたり
    std::array<double, kStrikes> rho{};    ///< 金利 1.0 あたり

    std::array<double, kGridS> grid_s{};  ///< Γ 格子の S 座標（昇順）
    std::array<double, kGridT> grid_t{};  ///< Γ 格子の T 座標（昇順、すべて > 0）
    /// Γ(S, T)。row-major [iS][iT] = gamma_surface[iS * kGridT + iT]。ストライクは S0 に固定。
    std::array<double, kGridS * kGridT> gamma_surface{};

    std::uint64_t seq = 0;  ///< ステップ通番（0 = 未ステップ／Reset 直後）
};

static_assert(std::is_trivially_copyable_v<GreeksSnapshot>);
static_assert(std::is_standard_layout_v<GreeksSnapshot>);
static_assert(sizeof(GreeksSnapshot) <= 32 * 1024, "M1 の Snapshot 上限（docs/02 の『Snapshot のサイズ方針』）");

class GreeksModel {
public:
    using Snapshot = GreeksSnapshot;

    /// 既定はコール。Put 版を出すなら Snapshot を分けるのではなく別シーンにする。
    static constexpr core::OptionType kOptionType = core::OptionType::Call;

    enum Param : std::uint32_t {
        /// 現在のスポットを value 倍する手動ショック（apply の場で適用、非正・非有限は無視）。
        /// value <= 0 と非有限値は「無視」（クランプではなく破棄。0 倍や負のスポットは意味を持たず、
        /// 既定値に落とすと UI の誤操作が静かにパスを壊すため）。掛け算の結果が非有限になる場合も同様。
        kSpotJump   = 1,
        kRate       = 2,  ///< r（非有限値は無視）
        kSigma      = 3,  ///< 価格付けの sigma（< 0 と NaN は 0 にクランプ）
        kMaturity   = 4,  ///< T（kMinMaturity 未満と NaN は kMinMaturity にクランプ）
        kStrikeSpan = 5,  ///< ストライク軸の半幅（(0, kMaxSpan] にクランプ）
    };

    static constexpr double kMinMaturity = 1.0 / 365.0;  ///< 1 日。T > 0 を保つ下限
    static constexpr double kMinSpan     = 0.01;
    static constexpr double kMaxSpan     = 0.90;

    struct Config {
        core::GbmParams gbm{};            ///< s0 / mu / sigma。sigma は**パスの**ボラティリティ
        double          r           = 0.02;
        double          sigma       = 0.20;  ///< **価格付けの**ボラティリティ（kSigma で可変）
        double          maturity    = 1.00;  ///< T_max（年）
        double          strike_span = 0.40;  ///< ストライク軸 [S0(1-span), S0(1+span)]
        std::uint64_t   seed        = 42;
    };

    // GCC のバグ回避: 既定メンバ初期化子を持つ入れ子 Config を既定引数にすると壊れるので委譲する。
    GreeksModel() : GreeksModel(Config{}) {}

    explicit GreeksModel(Config cfg)
        : gbm_(sanitize_gbm(cfg.gbm), cfg.seed),
          s0_(sanitize_s0(cfg.gbm.s0)),
          spot_(s0_),
          seed_(cfg.seed),
          pending_{clamp_rate(cfg.r, 0.0), clamp_sigma(cfg.sigma), clamp_maturity(cfg.maturity),
                   clamp_span(cfg.strike_span)},
          active_(pending_) {
        rebuild_strike_axis();
        rebuild_surface();
        publish();
    }

    /// 1 ステップ: ペンディングのパラメータを確定 → S を進める → ストリップを張り直す。
    /// 面（Γ 格子）は (r, sigma, T_max) が変わったときだけ作り直す（per-step コストを小さく保つ）。
    void step(double dt) {
        commit_pending();
        const double log_return = gbm_.step(dt);
        // Gbm と同じ演算（s *= exp(r)）を同じ順序で行うので、ショックが無い限り bit 一致する。
        spot_ *= std::exp(log_return);
        ++seq_;
        publish();
    }

    Snapshot snapshot() const noexcept { return snap_; }

    void apply(const bridge::Command& c) {
        switch (c.type) {
            case bridge::CommandType::SetParam:
                switch (c.param_id) {
                    case kSpotJump:   queue_spot_jump(c.value); break;
                    case kRate:       pending_.r = clamp_rate(c.value, pending_.r); break;
                    case kSigma:      pending_.sigma = clamp_sigma(c.value); break;
                    case kMaturity:   pending_.maturity = clamp_maturity(c.value); break;
                    case kStrikeSpan: pending_.span = clamp_span(c.value); break;
                    default:          return;  // 未知の param_id は無視（再計算もしない）
                }
                // 一時停止中でも Snapshot が新しい値を映すよう、その場で確定して張り直す（R10 と対）。
                // seq は動かさない。step() 側の commit_pending() は差分が無ければ何もしない。
                commit_pending();
                publish();
                break;
            case bridge::CommandType::Reset: reset(c.seed != 0 ? c.seed : seed_); break;
            default:                         break;  // 時計系は Runner が処理済み
        }
    }

    /// パスを初期状態へ。r / sigma / T / span は保持し、ペンディング分はここで確定する。
    void reset(std::uint64_t seed) {
        seed_ = seed;
        gbm_.reset(seed);
        spot_      = s0_;
        jump_      = 1.0;
        seq_       = 0;
        active_    = pending_;
        rebuild_strike_axis();
        rebuild_surface();
        publish();
    }

    // テスト用の読み取りアクセサ
    const core::Gbm& gbm() const noexcept { return gbm_; }
    double           spot() const noexcept { return spot_; }

private:
    struct Params {
        double r        = 0.0;
        double sigma    = 0.0;
        double maturity = 0.0;
        double span     = 0.0;
    };

    // ------------------------------------------------------------------ 入力のクランプ（例外なし）
    /// S0 は全ての軸（ストライク軸・Γ 格子）と価格付けの基準なので、非有限 / 非正なら既定の 100 に落とす。
    static double sanitize_s0(double s0) noexcept { return (s0 > 0.0 && std::isfinite(s0)) ? s0 : 100.0; }

    /// Gbm に渡す前に s0 だけ健全化する（mu / sigma は Gbm 側の責務）。
    static core::GbmParams sanitize_gbm(core::GbmParams p) noexcept {
        p.s0 = sanitize_s0(p.s0);
        return p;
    }

    /// 非有限値は「変更なし」として現在値を返す（r には自然な範囲が無いので下限を置かない）。
    static double clamp_rate(double v, double current) noexcept { return std::isfinite(v) ? v : current; }
    /// sigma >= 0。NaN も 0 に落ちる（`v > 0.0` が false）。
    static double clamp_sigma(double v) noexcept { return (v > 0.0) ? v : 0.0; }
    /// T > 0。NaN は下限へ。
    static double clamp_maturity(double v) noexcept { return (v > kMinMaturity) ? v : kMinMaturity; }
    /// span ∈ (0, kMaxSpan]。NaN は下限へ。
    static double clamp_span(double v) noexcept {
        if (!(v > kMinSpan)) return kMinSpan;
        return (v < kMaxSpan) ? v : kMaxSpan;
    }

    /// スポットショックは倍率として積み上げ、commit_pending()（apply の場、および step の先頭）で消費する。
    void queue_spot_jump(double v) noexcept {
        if (v > 0.0 && std::isfinite(v)) jump_ *= v;
    }

    // ------------------------------------------------------------------ パラメータ確定とキャッシュ
    void commit_pending() noexcept {
        if (jump_ != 1.0) {
            const double shocked = spot_ * jump_;
            if (shocked > 0.0 && std::isfinite(shocked)) spot_ = shocked;
            jump_ = 1.0;
        }
        if (pending_.span != active_.span) {
            active_.span = pending_.span;
            rebuild_strike_axis();
        }
        // 面は (r, sigma, T_max) にしか依らない（S 軸は S0 固定）→ 変わったときだけ張り直す。
        const bool surface_stale = pending_.r != active_.r || pending_.sigma != active_.sigma ||
                                   pending_.maturity != active_.maturity;
        active_.r        = pending_.r;
        active_.sigma    = pending_.sigma;
        active_.maturity = pending_.maturity;
        if (surface_stale) rebuild_surface();
    }

    /// 区間 [lo, hi] を N 点で等分する（i = N-1 が厳密に hi になる形）。
    template <std::size_t N>
    static double axis_point(double lo, double hi, std::size_t i) noexcept {
        static_assert(N >= 2, "axis_point: 等間隔軸には少なくとも 2 点必要（N-1 で割るため）");
        const double u = static_cast<double>(i) / static_cast<double>(N - 1);
        return lo + (hi - lo) * u;
    }

    void rebuild_strike_axis() noexcept {
        const double lo = s0_ * (1.0 - active_.span);
        const double hi = s0_ * (1.0 + active_.span);
        for (std::size_t i = 0; i < Snapshot::kStrikes; ++i)
            snap_.strikes[i] = axis_point<Snapshot::kStrikes>(lo, hi, i);
    }

    /// Γ(S, T) 格子。S ∈ [0.5 S0, 1.5 S0]、T ∈ (0, T_max]（T = 0 を含めないので値は有限）。
    void rebuild_surface() noexcept {
        for (std::size_t i = 0; i < Snapshot::kGridS; ++i)
            snap_.grid_s[i] = axis_point<Snapshot::kGridS>(0.5 * s0_, 1.5 * s0_, i);
        for (std::size_t j = 0; j < Snapshot::kGridT; ++j)
            snap_.grid_t[j] = active_.maturity * static_cast<double>(j + 1) /
                              static_cast<double>(Snapshot::kGridT);

        for (std::size_t i = 0; i < Snapshot::kGridS; ++i) {
            for (std::size_t j = 0; j < Snapshot::kGridT; ++j) {
                const core::BsGreeks g = core::bs_greeks(snap_.grid_s[i], s0_, snap_.grid_t[j], active_.r,
                                                         active_.sigma, kOptionType);
                snap_.gamma_surface[i * Snapshot::kGridT + j] = g.gamma;
            }
        }
    }

    /// 現在の (S, r, sigma, T) でストリップを張り直し、真値フィールドを更新する。
    void publish() noexcept {
        snap_.t     = gbm_.time();
        snap_.spot  = spot_;
        snap_.r     = active_.r;
        snap_.sigma = active_.sigma;
        snap_.T     = active_.maturity;
        snap_.seq   = seq_;

        // 価格だけはベクトル化されたストリップ版で計算する。数値としては bs_greeks().price と同じ
        // （BS-10: 相対 1e-15 以内、実測は bit 一致）なので「速いから」ではなく、`bs_price_strip` を
        // 本番経路に置き続けて SIMD 実装が退行したら気付けるようにするためである（パネルは価格を
        // 描かないが、Snapshot には載るので契約テストが毎回スカラ版と突き合わせる）。
        core::bs_price_strip(spot_, std::span<const double>(snap_.strikes), active_.maturity, active_.r,
                             active_.sigma, kOptionType, std::span<double>(snap_.price));
        for (std::size_t i = 0; i < Snapshot::kStrikes; ++i) {
            const core::BsGreeks g = core::bs_greeks(spot_, snap_.strikes[i], active_.maturity, active_.r,
                                                     active_.sigma, kOptionType);
            snap_.delta[i]         = g.delta;
            snap_.gamma[i]         = g.gamma;
            snap_.vega[i]          = g.vega;
            snap_.theta[i]         = g.theta;
            snap_.rho[i]           = g.rho;
        }
    }

    core::Gbm     gbm_;
    double        s0_;             ///< 軸を張る基準（Config の s0 で固定）
    double        spot_;           ///< 現在の S（ショックがあるので Gbm の内部値とは別に持つ）
    std::uint64_t seed_;
    double        jump_ = 1.0;     ///< 未適用のスポットショック倍率
    std::uint64_t seq_  = 0;
    Params        pending_;        ///< UI から来た値（apply の場で確定。step 先頭の commit は差分ゼロ）
    Params        active_;         ///< snap_ の中身を計算したときの値
    Snapshot      snap_{};
};

static_assert(bridge::Model<GreeksModel>, "GreeksModel must satisfy the Model contract");

}  // namespace quantviz::scenes
