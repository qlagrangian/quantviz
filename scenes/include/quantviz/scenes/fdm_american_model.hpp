#pragma once
// scenes/fdm_american_model.hpp — シーン「FDM American」: Crank–Nicolson（+ PSOR）の後ろ向き反復を
// 1 ステップずつ手送りし、満期から今日へ向かって伸びていく V(S, t) の面を吐く Model。
//
// 責務: `core::FdmCn` を 1 台抱え、`step()` 1 回で `step_backward()` 1 回を回す。描画側には 2 系統を渡す:
//
//   1. `Snapshot`（リング）: **現在の時刻断面**だけ。V(S) を 256 点に補間した曲線、S0 での値と Δ、
//      行使境界 S*、反復数・残り・PSOR の状態、参照線に要るパラメータの真値。≈ 6 KB。
//   2. `Surface`（`bridge::TripleBuffer`）: **これまでに解けた時間レベル全部**を 200×200 の float 格子で。
//      row-major [iT][iS]、行 0 が満期。まだ解いていない行はゼロで埋める（下記「未計算行」）。
//
// 面の出所はモデルが持つ「レベル・バッファ」= 反復が 1 つ終わるごとに V(S) を 200 点サンプルして
// 1 行ぶん追記する (n_time + 1) × 200 の double 配列（既定で 201×200 = 約 314 KiB）。確保は
// コンストラクタの一度きりで、`step` / `snapshot` / `surface` はどれも確保しない。
//
// 【時間レベルと面の行の対応】掃引は n_time + 1 個のレベル（レベル 0 = 満期 … レベル n_time = t 0）を
// 生むが、面は kT = 200 行しか持てない。そこで行 j にはレベル `round(j · n_time / (kT − 1))` を載せる:
// 単調非減少で、行 0 は必ずレベル 0（= 満期ペイオフ）、最終行は必ずレベル n_time（= t = 0）になる。
// 既定（n_time = 200）では 1 レベルだけが間引かれ、n_time < kT では同じレベルが複数行に写る。
// 「行 0 = ペイオフ、最後に埋まった行 = 現在の解」という読み方がどの n_time でも崩れないことを優先した。
//
// 【未計算行】まだ解いていない行はゼロで埋める（計画書の指定どおり）。最後に埋まった行を前方へ
// 押し出す（extrude）案もあるが、それは「無いデータを有るように見せる」ことになる。ゼロ平面に
// 対して計算済みの領域が楔形に伸びるほうが、後ろ向き反復が今日へ進んでいく様子が一目で分かる。
//
// 【パラメータの反映】他シーンの「ペンディング → 次の step で確定」とは**違う**。PDE の掃引は途中で
// 係数を差し替えられない（満期から途中までは古い σ、そこから先は新しい σ、という面は何の解でもない）
// ので、`SetParam` は**その場で `init` をやり直す** = 満期ペイオフに巻き戻して反復 0 から始める。
// Reset まで待たせる案も採り得るが、スライダーを動かした人が期待するのは「今の面が新しい値で描き直る」
// ことであり、待たせると「効いていない」と読まれる。巻き戻すので `seq` も 0 に戻る（= Reset と同じ扱い）。
// 値が実際に変わらない SetParam（同じ値の再送・クランプで同着）では巻き戻さない。
//
// 【S_max】S_max = kSmaxMultiple · K に固定する。こうすると K は常に節点 n_space / 4 に乗り、
// `fdm_cn.hpp` が言う「ペイオフのキンクが格子点に乗る」状態が K を動かしても保たれる
// （セル平均による 2 次収束と、満期断面が本源的価値そのものになる性質の両方がこれに依る）。

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/core/pricing/fdm_cn.hpp"

namespace quantviz::scenes {

/// 現在の時刻断面。配列 3 本で ≈ 6 KB（M1 の Greeks と同じ「32 KiB まで」の例外枠）。
struct FdmSceneSnapshot {
    static constexpr std::size_t kCurve = 256;  ///< V(S) 曲線と行使境界の履歴の点数

    static constexpr std::uint32_t kStatusOk            = 0;  ///< 直前の反復は収束した
    static constexpr std::uint32_t kStatusNotConverged  = 1;  ///< PSOR / 三重対角ソルバが収束しなかった

    /// 現在の時刻 t（= `FdmCn::time()`）。「掃引が今日 t = 0 に届くまでに残っている年数」であって、
    /// 満期までの残存期間ではない: 満期（反復 0）で T、解き終わりで 0。残存期間は τ = maturity − t_remaining
    /// で、パネルはこの τ で European 参照価格を引き、行使境界を τ 軸に描く。
    double t_remaining = 0.0;
    double maturity    = 0.0;  ///< T（真値。行使境界の時間軸を張るのに使う）
    double s0          = 0.0;  ///< 参照スポット（縦線・テレメトリ用。[0, S_max] に丸めてある）
    double v_at_s0     = 0.0;
    double delta_at_s0 = 0.0;
    /// 現在の時刻断面での行使境界 S*。European と該当節点なしは NaN。
    double exercise_boundary_now = 0.0;

    double K     = 0.0;  ///< 以下 4 つは参照線用のパラメータ真値
    double r     = 0.0;
    double sigma = 0.0;
    double q     = 0.0;

    std::array<double, kCurve> spots{};   ///< [0, S_max] の等間隔 256 点（昇順）
    std::array<double, kCurve> values{};  ///< 現在時刻の V(S)（線形補間）
    /// 完了した反復ごとの S*（index i = i + 1 回目の反復の直後）。該当なし・European は NaN。
    /// 【打ち切り】高々 kCurve = 256 反復ぶん。n_time > 256 の格子では 257 回目以降の S* は記録されない
    /// （`boundary_len` もそこで止まる）。面と V(S) は全反復ぶん正しく、切れるのはこの履歴だけ。
    std::array<double, kCurve> boundary_t{};

    std::uint64_t seq = 0;  ///< ステップ通番（0 = 未ステップ／Reset 直後／パラメータ変更直後）

    std::uint32_t iteration       = 0;  ///< 完了した後ろ向き反復数
    std::uint32_t remaining       = 0;  ///< 残り反復数（0 = t = 0 まで解けた）
    std::uint32_t n_time          = 0;  ///< 時間分割数 M（boundary_t の時間軸に使う）
    std::uint32_t boundary_len    = 0;  ///< boundary_t の有効長（= min(iteration, kCurve)。上の「打ち切り」）
    std::uint32_t psor_iterations = 0;  ///< 直前の反復で回した PSOR スイープ数（European は 0）
    std::uint32_t status          = kStatusOk;

    core::OptionType type     = core::OptionType::Put;
    bool             american = true;
};

static_assert(std::is_trivially_copyable_v<FdmSceneSnapshot>);
static_assert(std::is_standard_layout_v<FdmSceneSnapshot>);
static_assert(sizeof(FdmSceneSnapshot) <= 32 * 1024, "Snapshot 上限（docs/02『Snapshot のサイズ方針』）");

/// 3D 面。`bridge::TripleBuffer` の使い回しスロットに書くので、`surface()` は毎回全フィールドを書く。
struct FdmSceneSurface {
    static constexpr std::size_t kS = 200;  ///< S 方向の点数
    static constexpr std::size_t kT = 200;  ///< 時間方向の行数（行 0 = 満期）

    std::array<float, kS>      spots{};  ///< S 軸（昇順・等間隔）
    std::array<float, kT>      times{};  ///< 各行の t（`t_remaining` と同じ規約。降順、行 0 = T、最終行 = 0）
    std::array<float, kS * kT> values{}; ///< V。row-major、index = iT * kS + iS。未計算行は 0
    std::uint32_t              filled_rows = 0;  ///< 先頭から何行が計算済みか（≥ 1）
};

static_assert(std::is_trivially_copyable_v<FdmSceneSurface>);
static_assert(std::is_standard_layout_v<FdmSceneSurface>);

class FdmAmericanModel {
public:
    using Snapshot = FdmSceneSnapshot;
    using Surface  = FdmSceneSurface;

    /// S_max = kSmaxMultiple · K（ヘッダ冒頭「S_max」参照）。
    static constexpr double kSmaxMultiple = 4.0;

    enum Param : std::uint32_t {
        kStrike     = 1,  ///< K（[kMinStrike, kMaxStrike]）
        kRate       = 2,  ///< r（[kMinRate, kMaxRate]）
        kSigma      = 3,  ///< σ（[kMinSigma, kMaxSigma]）
        kDividend   = 4,  ///< q（[kMinDividend, kMaxDividend]）
        kAmerican   = 5,  ///< 0 = European、それ以外 = American
        kOptionType = 6,  ///< < 0.5 = Call、≥ 0.5 = Put（core::OptionType の並びと同じ）
        kOmega      = 7,  ///< PSOR の緩和係数 ω（[kMinOmega, kMaxOmega]）
    };

    static constexpr double kMinStrike   = 1.0;
    static constexpr double kMaxStrike   = 1000.0;
    static constexpr double kMinRate     = -0.5;
    static constexpr double kMaxRate     = 0.5;
    /// σ = 0 は PDE が退化し PSOR が収束しないので、下限は小さな正の値にする。
    static constexpr double kMinSigma    = 0.005;
    static constexpr double kMaxSigma    = 2.0;
    static constexpr double kMinDividend = 0.0;
    static constexpr double kMaxDividend = 0.5;
    static constexpr double kMinOmega    = 1.0;
    static constexpr double kMaxOmega    = 1.99;

    /// 格子の上限（レベル・バッファのメモリを縛る: (kMaxTime + 1) × kS × 8 B ≈ 8 MB）。
    static constexpr std::size_t kMinSpace = 4;
    static constexpr std::size_t kMaxSpace = 2000;
    static constexpr std::size_t kMinTime  = 1;
    static constexpr std::size_t kMaxTime  = 5000;

    struct Config {
        std::size_t      n_space  = 200;    ///< 空間分割数 N（節点は N + 1）
        /// 時間分割数 M（= 後ろ向き反復の回数）。M > FdmSceneSnapshot::kCurve (256) にすると
        /// 行使境界の履歴が先頭 256 反復で打ち切られる（面と V(S) は影響を受けない）。
        std::size_t      n_time   = 200;
        double           K        = 100.0;
        double           T        = 1.0;
        double           r        = 0.05;
        double           sigma    = 0.20;
        double           q        = 0.0;
        bool             american = true;
        core::OptionType type     = core::OptionType::Put;
        double           omega    = 1.2;   ///< PSOR の緩和係数
        double           s0       = 100.0; ///< 参照スポット（縦線・テレメトリ）
    };

    // GCC のバグ回避: 既定メンバ初期化子を持つ入れ子 Config を既定引数にしない（CLAUDE.md）。
    FdmAmericanModel() : FdmAmericanModel(Config{}) {}

    explicit FdmAmericanModel(Config cfg)
        : n_space_(std::clamp(cfg.n_space, kMinSpace, kMaxSpace)),
          n_time_(std::clamp(cfg.n_time, kMinTime, kMaxTime)),
          maturity_(clamp_maturity(cfg.T)),
          s0_config_(cfg.s0),
          active_{clamp_param(cfg.K, kMinStrike, kMaxStrike, 100.0),
                  clamp_param(cfg.r, kMinRate, kMaxRate, 0.05),
                  clamp_param(cfg.sigma, kMinSigma, kMaxSigma, 0.20),
                  clamp_param(cfg.q, kMinDividend, kMaxDividend, 0.0),
                  clamp_param(cfg.omega, kMinOmega, kMaxOmega, 1.2),
                  cfg.american,
                  cfg.type},
          level_spots_(Surface::kS),
          levels_((n_time_ + 1) * Surface::kS) {
        init_solver();
    }

    /// 1 ステップ = 後ろ向き反復 1 回。`dt` は使わない（時間の刻みは格子が決める）。
    /// t = 0 まで解き終わったあとは何もしないが、`seq` は進める: Runner から見れば「1 ステップ実行した」
    /// のは事実で、ここで seq を止めると描画側のレート計とテレメトリが嘘をつく。
    void step(double /*dt*/) {
        if (fdm_.step_backward()) {
            ++iteration_;
            store_level(iteration_);
            record_boundary();
            status_ = fdm_.last_psor_converged() ? Snapshot::kStatusOk : Snapshot::kStatusNotConverged;
        }
        ++seq_;
        publish();
    }

    Snapshot snapshot() const noexcept { return snap_; }

    /// 現在の面を s に書く。TripleBuffer の使い回しスロットなので**全フィールド**を書く。
    void surface(Surface& s) const noexcept {
        const std::size_t filled = filled_rows();
        for (std::size_t j = 0; j < Surface::kS; ++j) s.spots[j] = static_cast<float>(level_spots_[j]);

        const double inv_m = 1.0 / static_cast<double>(n_time_);
        for (std::size_t row = 0; row < Surface::kT; ++row) {
            const std::size_t level = level_of_row(row);
            s.times[row] = static_cast<float>(maturity_ * static_cast<double>(n_time_ - level) * inv_m);

            float* const dst = s.values.data() + row * Surface::kS;
            if (row < filled) {
                const double* const src = levels_.data() + level * Surface::kS;
                for (std::size_t j = 0; j < Surface::kS; ++j) dst[j] = static_cast<float>(src[j]);
            } else {
                for (std::size_t j = 0; j < Surface::kS; ++j) dst[j] = 0.f;
            }
        }
        s.filled_rows = static_cast<std::uint32_t>(filled);
    }

    void apply(const bridge::Command& c) {
        switch (c.type) {
            case bridge::CommandType::SetParam: apply_param(c.param_id, c.value); break;
            case bridge::CommandType::Reset:    reset(c.seed); break;
            default:                            break;  // 時計系は Runner が処理済み
        }
    }

    /// 満期ペイオフへ巻き戻す（= `FdmCn::init` のやり直し）。パラメータは保持する。
    /// このシーンに乱数は無いので seed は使わない。
    void reset(std::uint64_t /*seed*/ = 0) { init_solver(); }

    // ------------------------------------------------------------------ テスト用の読み取りアクセサ
    [[nodiscard]] const core::FdmCn& solver() const noexcept { return fdm_; }
    [[nodiscard]] std::size_t        iteration() const noexcept { return iteration_; }

private:
    struct Params {
        double           K     = 0.0;
        double           r     = 0.0;
        double           sigma = 0.0;
        double           q     = 0.0;
        double           omega = 0.0;
        bool             american = true;
        core::OptionType type     = core::OptionType::Put;

        friend bool operator==(const Params&, const Params&) = default;
    };

    // ------------------------------------------------------------------ 入力の丸め（例外なし）
    /// 有限値は [lo, hi] に丸め、NaN / ∞ は**拒否**して現在値を返す。
    /// （下限に倒す流儀もあるが、K や σ が UI の事故で最小値に飛ぶと面が別物になるので拒否を選ぶ。）
    static double clamp_param(double v, double lo, double hi, double current) noexcept {
        if (!std::isfinite(v)) return current;
        return v < lo ? lo : (v > hi ? hi : v);
    }
    static double clamp_maturity(double t) noexcept {
        if (!(t > 1e-6)) return 1.0;  // NaN もここへ落ちる
        return t < 100.0 ? t : 100.0;
    }

    /// noexcept は付けない: `init_solver()` が `FdmCn::init` を呼び、その中は `std::vector::resize`。
    /// 現状は格子（n_space_ / n_time_）が生涯不変なので resize は必ず「同じ大きさ」= 確保も例外も起きないが、
    /// その不変条件はこのクラスの都合であって `init` の契約ではない。noexcept を約束すると、将来 UI から
    /// 格子を変えられるようにした瞬間に「確保失敗 → std::terminate」に化ける。Command の処理は
    /// ホットパス（step / snapshot / リング）ではないので、noexcept を外しても失うものは無い。
    void apply_param(std::uint32_t id, double v) {
        Params next = active_;
        switch (id) {
            case kStrike:   next.K = clamp_param(v, kMinStrike, kMaxStrike, active_.K); break;
            case kRate:     next.r = clamp_param(v, kMinRate, kMaxRate, active_.r); break;
            case kSigma:    next.sigma = clamp_param(v, kMinSigma, kMaxSigma, active_.sigma); break;
            case kDividend: next.q = clamp_param(v, kMinDividend, kMaxDividend, active_.q); break;
            case kOmega:    next.omega = clamp_param(v, kMinOmega, kMaxOmega, active_.omega); break;
            case kAmerican:
                if (!std::isfinite(v)) return;
                next.american = (v != 0.0);
                break;
            case kOptionType:
                if (!std::isfinite(v)) return;
                next.type = (v >= 0.5) ? core::OptionType::Put : core::OptionType::Call;
                break;
            default: return;  // 未知の param_id は無視
        }
        if (next == active_) return;  // 値が動かないなら掃引を巻き戻さない
        active_ = next;
        init_solver();
    }

    // ------------------------------------------------------------------ 掃引の開始 / レベルの蓄積
    /// 格子とパラメータを組み直し、満期ペイオフ（レベル 0）を置いて反復 0 に戻す。
    /// `FdmCn::init` の `resize` は格子サイズが変わらない限り確保しない（N と M はこのモデルでは不変）。
    void init_solver() {
        const double s_max = kSmaxMultiple * active_.K;

        core::FdmParams p{};
        p.K          = active_.K;
        p.T          = maturity_;
        p.r          = active_.r;
        p.sigma      = active_.sigma;
        p.q          = active_.q;
        p.type       = active_.type;
        p.american   = active_.american;
        p.psor_omega = active_.omega;
        fdm_.init(core::FdmGrid{s_max, n_space_, n_time_}, p);

        for (std::size_t j = 0; j < Surface::kS; ++j)
            level_spots_[j] = axis_point(s_max, j, Surface::kS);
        // s0 は格子の外に出さない（K を大きく下げると S_max が s0 を下回りうる。value_at は外で NaN）。
        s0_ = std::clamp(std::isfinite(s0_config_) && s0_config_ > 0.0 ? s0_config_ : active_.K, 0.0, s_max);

        std::fill(levels_.begin(), levels_.end(), 0.0);
        boundary_.fill(std::numeric_limits<double>::quiet_NaN());
        boundary_len_ = 0;
        iteration_    = 0;
        status_       = Snapshot::kStatusOk;
        seq_          = 0;
        store_level(0);
        publish();
    }

    /// 区間 [0, hi] を n 点で等分した i 番目（i = n − 1 が厳密に hi）。
    static double axis_point(double hi, std::size_t i, std::size_t n) noexcept {
        return hi * (static_cast<double>(i) / static_cast<double>(n - 1));
    }

    /// レベル level の V(S) を面のサンプル点 200 個で控える（確保なし）。
    void store_level(std::size_t level) noexcept {
        double* const dst = levels_.data() + level * Surface::kS;
        for (std::size_t j = 0; j < Surface::kS; ++j) dst[j] = fdm_.value_at(level_spots_[j]);
    }

    void record_boundary() noexcept {
        if (boundary_len_ >= Snapshot::kCurve) return;  // 溢れた分は捨てる（先頭 256 反復だけ持つ）
        boundary_[boundary_len_++] = fdm_.exercise_boundary();
    }

    /// 面の行 j に載せる時間レベル（ヘッダ冒頭「時間レベルと面の行の対応」）。単調非減少、
    /// level_of_row(0) = 0、level_of_row(kT − 1) = n_time。
    [[nodiscard]] std::size_t level_of_row(std::size_t row) const noexcept {
        constexpr std::size_t d = Surface::kT - 1;
        return (row * n_time_ + d / 2) / d;
    }

    /// 先頭から何行が計算済みか（level_of_row は単調なので最初に超えた所で止まる）。
    [[nodiscard]] std::size_t filled_rows() const noexcept {
        std::size_t n = 0;
        while (n < Surface::kT && level_of_row(n) <= iteration_) ++n;
        return n;
    }

    // ------------------------------------------------------------------ Snapshot 更新
    void publish() noexcept {
        const double s_max = fdm_.grid().s_max;
        for (std::size_t i = 0; i < Snapshot::kCurve; ++i) {
            snap_.spots[i]  = axis_point(s_max, i, Snapshot::kCurve);
            snap_.values[i] = fdm_.value_at(snap_.spots[i]);
        }
        snap_.boundary_t = boundary_;

        snap_.t_remaining          = fdm_.time();
        snap_.maturity             = maturity_;
        snap_.s0                   = s0_;
        snap_.v_at_s0              = fdm_.value_at(s0_);
        snap_.delta_at_s0          = fdm_.delta_at(s0_);
        snap_.exercise_boundary_now = fdm_.exercise_boundary();

        snap_.K     = active_.K;
        snap_.r     = active_.r;
        snap_.sigma = active_.sigma;
        snap_.q     = active_.q;

        snap_.seq             = seq_;
        snap_.iteration       = static_cast<std::uint32_t>(iteration_);
        snap_.remaining       = static_cast<std::uint32_t>(fdm_.remaining());
        snap_.n_time          = static_cast<std::uint32_t>(n_time_);
        snap_.boundary_len    = static_cast<std::uint32_t>(boundary_len_);
        snap_.psor_iterations = static_cast<std::uint32_t>(fdm_.last_psor_iterations());
        // 発散（非有限値）も「収束しなかった」として扱う（設計書 §10: コアは止めず status で知らせる）。
        snap_.status = (status_ == Snapshot::kStatusOk && std::isfinite(snap_.v_at_s0))
                           ? Snapshot::kStatusOk
                           : Snapshot::kStatusNotConverged;

        snap_.type     = active_.type;
        snap_.american = active_.american;
    }

    std::size_t n_space_;
    std::size_t n_time_;
    double      maturity_;
    double      s0_config_;  ///< Config が指定した参照スポット（格子に丸める前）
    double      s0_ = 0.0;   ///< [0, S_max] に丸めた参照スポット

    Params      active_;
    core::FdmCn fdm_;

    std::vector<double> level_spots_;  ///< 面の S 軸（kS 点、double 精度で保持）
    std::vector<double> levels_;       ///< (n_time + 1) × kS の V。行 = 時間レベル（行 0 = 満期）

    std::array<double, Snapshot::kCurve> boundary_{};  ///< 反復ごとの S*
    std::size_t                          boundary_len_ = 0;

    std::size_t   iteration_ = 0;
    std::uint64_t seq_       = 0;
    std::uint32_t status_    = Snapshot::kStatusOk;
    Snapshot      snap_{};
};

static_assert(bridge::Model<FdmAmericanModel>, "FdmAmericanModel must satisfy the Model contract");
static_assert(bridge::SurfaceModel<FdmAmericanModel>, "FdmAmericanModel must satisfy SurfaceModel");

}  // namespace quantviz::scenes
