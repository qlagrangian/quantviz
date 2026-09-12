#pragma once
// scenes/lsm_model.hpp — シーン「LSM American」: Longstaff–Schwartz の後ろ向き回帰を 1 行使時点ずつ
// 手送りし、パス・分位帯・継続価値フィットを吐く Model。
//
// 責務: `core::Lsm` を 1 台抱え、`step()` 1 回で `step_backward()` 1 回（= 1 行使時点の回帰 + 行使判定）を
// 回す。面（`SurfaceModel`）は持たない: 見せたい格子は「パス × 時点」だが、N = 4000 本を全部渡すのは
// 無意味（描いても潰れる）なので、Snapshot の中で **16 本 + 5 本の分位** に縮約する。
//
// 【Snapshot の中身は 2 種類】
//   1. init で決まり、後ろ向き反復では**変わらない**もの: `shown_paths`（16 本 × 65 点）と
//      `quantiles`（5 水準 × 65 点）。パスは init で生成したきり動かないので、`publish()` では書かない
//      （init が 1 回書いた値がそのまま残る）。
//   2. 1 時点ごとに変わるもの: `t_index` / `remaining` / `price` / `std_error` / `european` /
//      `exercised_paths` / `itm_now` / `shown_exercise`（行使時点は反復のたびに前へ動く）と、
//      下の「保持するフィット」。
//
// 【フィットは保持する】ある時点で ITM パスが 1 本も無ければ core は回帰せず、`continuation_coeffs()`
// は全 0、`last_fit_rank()` は 0、`continuation_value()` は NaN になる。既定の ATM（S0 = K）では
// **最後の時点 k = 0 がまさにこれ**（全パスが S0 なので h(S0) = 0）で、素直に写すと「掃引を終えた
// 瞬間に継続価値の窓が空になる」= 一番見たい所で絵が消える。そこで Snapshot は
// **直近に回帰できた時点のフィット**（`coeffs` / `fit_s` / `fit_v` / `fit_rank` / `fit_fallback` /
// `itm_paths`）を保持し、それが属する時点を `fit_t_index` で、保持に回った事実を `fit_stale` で示す。
// 現在の時点の ITM 本数は `itm_now` で別に出す（`fit_stale && itm_now == 0` なら「この時点は ITM が
// 無い」、`fit_stale && itm_now > 0` なら「回帰が発散した」）。フィットを 1 つも持たない状態
// （init 直後 = 満期）だけ `fit_valid` が false で、`fit_v` は全 NaN になる。
//
// 【価格の意味】`price` / `std_error` は `Lsm::result()` の **部分結果**をそのまま出す。満期から k 番目の
// 時点まで処理した状態では「t ≥ t_k でのみ行使できる Bermudan」の価格で、掃引の開始時（t_index =
// kSteps）は European MC そのもの、t_index = 0 まで進めて初めて LSM の American 価格になる。NaN で
// 隠す案もあったが、European から American へ価格がせり上がっていく様子こそがこのシーンの見どころ
// なので出す（パネルは remaining > 0 の間「partial」と明示する）。t = 0 の即時行使も core の規約に従う
// （h(S0) > 0 なら最後の時点で rank 1 の回帰になり「即時行使 vs 割引継続価値の平均」の判定が入る）。
//
// 【パラメータの反映】FDM シーンと同じ: `SetParam` はその場で `Lsm::init` をやり直す（= パスを同じ
// seed で生成し直し、満期へ巻き戻す）。後ろ向き回帰は途中で σ や K を差し替えられない（満期から
// 途中までは古い σ、そこから先は新しい σ、という価格は何の解でもない）ので、掃引の途中で効かせる
// 選択肢が無い。巻き戻すので `seq` も 0 に戻る（= Reset と同じ扱い）。クランプ後の値が動かない
// SetParam（同じ値の再送・クランプで同着）では巻き戻さない。
//
// 【確保】`Lsm::init` だけが確保する（パス 2 面 + 作業配列。n_paths = 20000 では
// 2 × 20000 × 65 × 8 B ≈ 20.8 MB）。`step` / `snapshot` / `apply`（= 未知 ID・同値・NaN で弾かれる経路）は
// 確保しない。確保が起きるのはコンストラクタと「SetParam / Reset による init やり直し」だけで、これは
// FDM シーンと同じ扱い（Command の処理はホットパスではない）。
//
// 【1 ステップの費用】`result()` が O(N) を 4 回走る（価格と European の平均 + 分散）ので、1 ステップは
// O(N)（N = 4000 で数万フロップ。掃引 64 時点ぜんぶで約 4 ms）。パネルの既定 8 steps/s では無視できる。
// パラメータ変更のたびに走る init は N = 4000 で ≈ 10–15 ms、N = 20000 で ≈ 45–80 ms（実測、機械依存）。
// 分位帯（O(N log N) 相当の nth_element を 5 × 65 回）は init でだけ払う。
//
// 【スレッド】`Lsm::quantile_at()` は const だが mutable な作業配列を使うので再入不可。呼ぶのは
// このモデルの init だけ（= Runner の計算スレッド）で、描画側には Snapshot に写した値しか渡さない。

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/core/pricing/lsm.hpp"

namespace quantviz::scenes {

/// LSM シーンの縮約 Snapshot（≈ 12 KB。M1 の Greeks と同じ「32 KiB まで」の例外枠）。
struct LsmSceneSnapshot {
    static constexpr std::size_t kShow     = 16;           ///< 描画するパスの本数（8 組のアンチセティック対）
    static constexpr std::size_t kSteps    = 64;           ///< 行使時点数 M（= 後ろ向きステップの総数）
    /// パス 1 本の点数。行使時点 M に対して時点は t_0 … t_M の M + 1 個ある（t_0 = 0 は全パス共通の S0）。
    /// 計画書の字面（`kShow * kSteps`）より 1 点だけ長いのは、t = 0 を落とすとパスの出発点が消え、
    /// 「現在の時点」の縦線が t_index = 0 まで引けなくなるため。
    static constexpr std::size_t kPts      = kSteps + 1;
    static constexpr std::size_t kQuant    = 5;            ///< 分位の本数（5 / 25 / 50 / 75 / 95 %）
    static constexpr std::size_t kFit      = 64;           ///< 継続価値カーブの点数
    static constexpr std::size_t kMaxBasis = core::Lsm::kMaxBasis;  ///< 係数配列の長さ（8）

    /// `quantiles` の行の水準（行 q の点 k = 時点 k の S の q 分位）。
    static constexpr std::array<double, kQuant> kQuantLevels{0.05, 0.25, 0.50, 0.75, 0.95};

    double t_now     = 0.0;  ///< 現在の時刻 t = T · t_index / kSteps（満期 T → 0 へ向かう）
    /// 現時点までの部分結果（ヘッダ冒頭「価格の意味」）。remaining == 0 で初めて LSM の American 価格。
    double price     = 0.0;
    double std_error = 0.0;  ///< アンチセティック対平均を標本単位とした SE
    double european  = 0.0;  ///< 同じパスの European MC（参照線）

    double K     = 0.0;  ///< 以下 5 つは参照線・軸に要るパラメータ真値
    double r     = 0.0;
    double sigma = 0.0;
    double T     = 0.0;
    double s0    = 0.0;

    /// 描画用に抜いた 16 本（row-major、index = i * kPts + k）。init で固定、反復では変わらない。
    std::array<double, kShow * kPts>  shown_paths{};
    /// 分位帯（row-major、index = q * kPts + k）。init で固定、反復では変わらない。
    std::array<double, kQuant * kPts> quantiles{};
    /// 現在の時点の継続価値カーブ。fit_s は昇順の S、fit_v は c(S)（フィットが無ければ全て NaN）。
    std::array<double, kFit>          fit_s{};
    std::array<double, kFit>          fit_v{};
    /// 直近の回帰係数（x = S/K の基底に対して）。n_basis 個より後ろは 0。
    std::array<double, kMaxBasis>     coeffs{};
    /// 16 本それぞれの行使時点。kSteps は「満期まで持った（早期行使なし）」を意味する。
    std::array<std::int32_t, kShow>   shown_exercise{};

    std::uint64_t seq = 0;  ///< ステップ通番（0 = 未ステップ／Reset 直後／パラメータ変更直後）

    /// 掃引が今いる時点の番号（kSteps = 満期、0 = 今日）。`remaining` と同じ値だが、片方は
    /// 「時間軸上の位置」、もう片方は「あと何回回帰するか」として読む（計画書の型契約が両方を挙げている）。
    std::uint32_t t_index         = 0;
    std::uint32_t remaining       = 0;
    std::uint32_t n_paths         = 0;  ///< 実際に使っているパス数（偶数）
    std::uint32_t n_basis         = 0;  ///< 回帰子の数（定数を含む）
    std::uint32_t exercised_paths = 0;  ///< 満期より前に行使したパス数
    std::uint32_t itm_now         = 0;  ///< **現在の時点**の ITM パス数（0 = フィットが作れない時点）

    // ---- 以下 4 つと coeffs / fit_s / fit_v は「保持しているフィット」の属性（ヘッダ冒頭を参照）。
    std::uint32_t fit_t_index = 0;      ///< そのフィットが属する時点（`fit_valid` のときだけ意味を持つ）
    std::uint32_t fit_rank    = 0;      ///< 実際に使った基底数（< n_basis ならランク打ち切り）
    std::uint32_t itm_paths   = 0;      ///< その回帰に使った ITM パス数

    core::LsmBasis   basis        = core::LsmBasis::Laguerre;
    core::OptionType type         = core::OptionType::Put;
    bool             fit_valid    = false;  ///< 保持しているフィットがあるか（掃引開始直後は無い）
    bool             fit_fallback = false;  ///< その回帰で正規方程式が特異でランクを落としたか
    /// 保持しているフィットが現在の時点のものではない（= この時点では回帰できなかった）。
    bool             fit_stale    = false;
};

static_assert(std::is_trivially_copyable_v<LsmSceneSnapshot>);
static_assert(std::is_standard_layout_v<LsmSceneSnapshot>);
static_assert(sizeof(LsmSceneSnapshot) <= 16 * 1024, "Snapshot 上限（docs/02『Snapshot のサイズ方針』）");

class LsmModel {
public:
    using Snapshot = LsmSceneSnapshot;

    enum Param : std::uint32_t {
        kStrike = 1,  ///< K（[kMinStrike, kMaxStrike]）
        kSigma  = 2,  ///< σ（[kMinSigma, kMaxSigma]。0 = 決定的パス）
        kRate   = 3,  ///< r（[kMinRate, kMaxRate]）
        kBasis  = 4,  ///< < 0.5 = Power、≥ 0.5 = Laguerre（core::LsmBasis の並びと同じ）
        kNBasis = 5,  ///< 回帰子の数（[kMinBasis, kMaxBasis]）
        kNPaths = 6,  ///< パス数（[kMinPaths, kMaxPaths]、偶数に切り上げ）
    };

    static constexpr double      kMinStrike = 1.0;
    static constexpr double      kMaxStrike = 1000.0;
    /// σ = 0 も許す（決定的なパス 1 本ぶんの正気度チェック: S_k = s0 e^{r t_k}、SE = 0）。ただし
    /// 既定の ATM put では r > 0 が S を K の上へ運ぶので ITM パスが 1 本も出ず、回帰は 1 度も
    /// 走らない（`fit_valid` が false のまま = パネルは「どの行使日にも ITM パスが無い」と出す）。
    /// σ = 0 で rank 1 フォールバックを見たいなら K > s0 e^{rT} にする（全パスが ITM だが S が 1 点に
    /// 潰れるので基底が線形従属になる）。σ > 0 でも ITM パスが n_basis 本未満の時点では同じように
    /// ランクが落ちる（深い OTM の put で実際に起きる）。
    static constexpr double      kMinSigma  = 0.0;
    static constexpr double      kMaxSigma  = 2.0;
    static constexpr double      kMinRate   = -0.5;
    static constexpr double      kMaxRate   = 0.5;
    static constexpr std::size_t kMinBasis  = 1;
    static constexpr std::size_t kMaxBasis  = core::Lsm::kMaxBasis;
    /// パス数の下限・上限。上限はメモリ（2 面 × N × (kSteps + 1) × 8 B ≈ 20.8 MB）と
    /// init のやり直しにかかる時間（スライダーを動かすたびに払う）で決めた。
    static constexpr std::size_t kMinPaths  = 256;
    static constexpr std::size_t kMaxPaths  = 20000;

    struct Config {
        double           K       = 100.0;
        double           T       = 1.0;
        double           r       = 0.05;
        double           sigma   = 0.20;
        double           s0      = 100.0;
        core::OptionType type    = core::OptionType::Put;
        /// 既定は 4000 本: SE ≈ 0.02 で価格が読め、init のやり直し（スライダー操作）が数 ms で済む。
        std::size_t      n_paths = 4000;
        std::size_t      n_basis = 3;  ///< Longstaff–Schwartz 2001 と同じ「定数 + Laguerre 2 本」
        core::LsmBasis   basis   = core::LsmBasis::Laguerre;
        std::uint64_t    seed    = 20260912;
    };

    // GCC のバグ回避: 既定メンバ初期化子を持つ入れ子 Config を既定引数にしない（CLAUDE.md）。
    LsmModel() : LsmModel(Config{}) {}

    explicit LsmModel(Config cfg)
        : maturity_(clamp_maturity(cfg.T)),
          s0_(clamp_spot(cfg.s0, cfg.K)),
          type_(cfg.type),
          seed_(cfg.seed),
          active_{clamp_param(cfg.K, kMinStrike, kMaxStrike, 100.0),
                  clamp_param(cfg.sigma, kMinSigma, kMaxSigma, 0.20),
                  clamp_param(cfg.r, kMinRate, kMaxRate, 0.05),
                  even_paths(clamp_count(static_cast<double>(cfg.n_paths), kMinPaths, kMaxPaths, 4000)),
                  clamp_count(static_cast<double>(cfg.n_basis), kMinBasis, kMaxBasis, 3),
                  cfg.basis} {
        init_solver();
    }

    /// 1 ステップ = 後ろ向き回帰 1 時点。`dt` は使わない（時間の刻みは行使時点が決める）。
    /// t = 0 まで解き終わったあとは何もしないが、`seq` は進める（FDM シーンと同じ: Runner から見れば
    /// 「1 ステップ実行した」のは事実で、ここで止めると描画側のレート計とテレメトリが嘘をつく）。
    void step(double /*dt*/) {
        lsm_.step_backward();
        ++seq_;
        publish();
    }

    Snapshot snapshot() const noexcept { return snap_; }

    void apply(const bridge::Command& c) {
        switch (c.type) {
            case bridge::CommandType::SetParam: apply_param(c.param_id, c.value); break;
            case bridge::CommandType::Reset:    reset(c.seed); break;
            default:                            break;  // 時計系は Runner が処理済み
        }
    }

    /// 満期へ巻き戻す（= `Lsm::init` のやり直し）。パラメータは保持する。
    /// seed 0 は「今の seed でやり直す」（= 同じパスの再生）。0 以外はその seed に差し替える。
    void reset(std::uint64_t seed = 0) {
        if (seed != 0) seed_ = seed;
        init_solver();
    }

    // ------------------------------------------------------------------ テスト用の読み取りアクセサ
    [[nodiscard]] const core::Lsm& lsm() const noexcept { return lsm_; }
    [[nodiscard]] std::uint64_t    seed() const noexcept { return seed_; }

private:
    struct Params {
        double         K       = 0.0;
        double         sigma   = 0.0;
        double         r       = 0.0;
        std::size_t    n_paths = 0;
        std::size_t    n_basis = 0;
        core::LsmBasis basis   = core::LsmBasis::Laguerre;

        friend bool operator==(const Params&, const Params&) = default;
    };

    // ------------------------------------------------------------------ 入力の丸め（例外なし）
    /// 有限値は [lo, hi] に丸め、NaN / ∞ は**拒否**して現在値を返す（FDM シーンと同じ方針:
    /// UI の事故で K や σ が最小値に飛ぶと別のシーンになってしまう）。
    static double clamp_param(double v, double lo, double hi, double current) noexcept {
        if (!std::isfinite(v)) return current;
        return v < lo ? lo : (v > hi ? hi : v);
    }
    /// 個数系（n_paths / n_basis）: [lo, hi] に丸めてから四捨五入。NaN / ∞ は拒否。
    static std::size_t clamp_count(double v, std::size_t lo, std::size_t hi, std::size_t current) noexcept {
        if (!std::isfinite(v)) return current;
        const double c = std::clamp(v, static_cast<double>(lo), static_cast<double>(hi));
        return static_cast<std::size_t>(c + 0.5);
    }
    static double clamp_maturity(double t) noexcept {
        if (!(t > 1e-6)) return 1.0;  // NaN もここへ落ちる
        return t < 100.0 ? t : 100.0;
    }
    static double clamp_spot(double s, double k) noexcept {
        if (!(s > 0.0) || !std::isfinite(s)) return (k > 0.0 && std::isfinite(k)) ? k : 100.0;
        return s;
    }
    /// パス数は必ず偶数（アンチセティック対）。`Lsm::init` も +1 に丸めるが、こちらでも揃えて
    /// 「同じ値の SetParam は巻き戻さない」判定が core の丸めとずれないようにする。
    static std::size_t even_paths(std::size_t n) noexcept { return (n % 2 == 0) ? n : n + 1; }

    void apply_param(std::uint32_t id, double v) {
        Params next = active_;
        switch (id) {
            case kStrike: next.K     = clamp_param(v, kMinStrike, kMaxStrike, active_.K); break;
            case kSigma:  next.sigma = clamp_param(v, kMinSigma, kMaxSigma, active_.sigma); break;
            case kRate:   next.r     = clamp_param(v, kMinRate, kMaxRate, active_.r); break;
            case kNBasis: next.n_basis = clamp_count(v, kMinBasis, kMaxBasis, active_.n_basis); break;
            case kNPaths:
                next.n_paths = even_paths(clamp_count(v, kMinPaths, kMaxPaths, active_.n_paths));
                break;
            case kBasis:
                if (!std::isfinite(v)) return;
                next.basis = (v >= 0.5) ? core::LsmBasis::Laguerre : core::LsmBasis::Power;
                break;
            default: return;  // 未知の param_id は無視
        }
        if (next == active_) return;  // 値が動かないなら掃引を巻き戻さない
        active_ = next;
        init_solver();
    }

    // ------------------------------------------------------------------ 掃引の開始
    /// パスを生成し直して満期（t_index = kSteps）に戻し、変わらない側の Snapshot（パスと分位帯）を書く。
    /// ここだけが確保する（`Lsm::init`）。
    void init_solver() {
        core::LsmParams p{};
        p.s0      = s0_;
        p.K       = active_.K;
        p.T       = maturity_;
        p.r       = active_.r;
        p.sigma   = active_.sigma;
        p.type    = type_;
        p.n_paths = even_paths(active_.n_paths);
        p.n_steps = Snapshot::kSteps;
        p.n_basis = active_.n_basis;
        p.basis   = active_.basis;
        p.seed    = seed_;
        lsm_.init(p);

        select_shown_paths();
        store_paths();
        store_quantiles();

        // 保持していたフィットは前のパラメータのものなので捨てる（軸だけ満期の帯で張り、値は NaN）。
        snap_.coeffs.fill(0.0);
        snap_.fit_t_index  = static_cast<std::uint32_t>(Snapshot::kSteps);
        snap_.fit_rank     = 0;
        snap_.itm_paths    = 0;
        snap_.fit_valid    = false;
        snap_.fit_fallback = false;
        snap_.fit_stale    = false;
        store_fit_curve(Snapshot::kSteps);

        seq_ = 0;
        publish();
    }

    /// 描画する 16 本を選ぶ: アンチセティック対 8 組を母集団に等間隔で散らす（対で採るので、
    /// 「+Z と −Z が鏡像になる」という生成器の構造がそのまま見える）。
    void select_shown_paths() noexcept {
        const std::size_t pairs      = lsm_.params().n_paths / 2;
        const std::size_t show_pairs = Snapshot::kShow / 2;
        for (std::size_t j = 0; j < show_pairs; ++j) {
            const std::size_t pair = std::min(j * pairs / show_pairs, pairs - 1);
            shown_index_[2 * j]     = 2 * pair;
            shown_index_[2 * j + 1] = 2 * pair + 1;
        }
    }

    void store_paths() noexcept {
        for (std::size_t i = 0; i < Snapshot::kShow; ++i) {
            const std::span<const double> path = lsm_.path(shown_index_[i]);
            double* const                 dst  = snap_.shown_paths.data() + i * Snapshot::kPts;
            for (std::size_t k = 0; k < Snapshot::kPts; ++k)
                dst[k] = k < path.size() ? path[k] : std::numeric_limits<double>::quiet_NaN();
        }
    }

    void store_quantiles() noexcept {
        for (std::size_t q = 0; q < Snapshot::kQuant; ++q) {
            double* const dst = snap_.quantiles.data() + q * Snapshot::kPts;
            for (std::size_t k = 0; k < Snapshot::kPts; ++k)
                dst[k] = lsm_.quantile_at(k, Snapshot::kQuantLevels[q]);
        }
    }

    // ------------------------------------------------------------------ Snapshot 更新
    /// 1 時点ごとに変わる側だけを書く（パスと分位帯は init が書いたまま）。
    void publish() noexcept {
        const core::LsmResult res = lsm_.result();
        snap_.price               = res.price;
        snap_.std_error           = res.std_error;
        snap_.european            = res.european_price;
        snap_.exercised_paths     = static_cast<std::uint32_t>(res.exercised_paths);

        snap_.t_index   = static_cast<std::uint32_t>(lsm_.current_step());
        snap_.remaining = static_cast<std::uint32_t>(lsm_.remaining());
        snap_.t_now     = lsm_.time();

        snap_.n_paths = static_cast<std::uint32_t>(lsm_.params().n_paths);
        snap_.n_basis = static_cast<std::uint32_t>(lsm_.params().n_basis);
        snap_.itm_now = static_cast<std::uint32_t>(lsm_.last_itm_paths());

        // この時点で回帰できたときだけフィットを差し替える（できなければ直近のものを保持する。
        // ヘッダ冒頭「フィットは保持する」）。core は rank 0 を「フィット無し」の印にしている。
        if (lsm_.last_fit_rank() > 0) {
            const std::span<const double> c = lsm_.continuation_coeffs();
            for (std::size_t j = 0; j < Snapshot::kMaxBasis; ++j)
                snap_.coeffs[j] = j < c.size() ? c[j] : 0.0;
            snap_.fit_t_index  = snap_.t_index;
            snap_.fit_rank     = static_cast<std::uint32_t>(lsm_.last_fit_rank());
            snap_.itm_paths    = snap_.itm_now;
            snap_.fit_fallback = lsm_.last_fit_fallback();
            snap_.fit_valid    = true;
            store_fit_curve(lsm_.current_step());
        }
        snap_.fit_stale = snap_.fit_valid && snap_.fit_t_index != snap_.t_index;

        for (std::size_t i = 0; i < Snapshot::kShow; ++i)
            snap_.shown_exercise[i] = static_cast<std::int32_t>(lsm_.exercise_step(shown_index_[i]));

        snap_.K     = active_.K;
        snap_.r     = active_.r;
        snap_.sigma = active_.sigma;
        snap_.T     = maturity_;
        snap_.s0    = s0_;
        snap_.basis = active_.basis;
        snap_.type  = type_;
        snap_.seq   = seq_;
    }

    /// 時点 k の継続価値カーブ。S の範囲は「その時点の 5–95 % 帯」と固定枠 [0.8 K, 1.1 K] の**和**:
    /// lo = min(q05, 0.8 K), hi = max(q95, 1.1 K)。帯だけだと行使境界（フィットと本源的価値の交点）が
    /// 端に寄って読めず、固定枠だけだと σ を上げたときにパスが枠外へ出る。K の周りに必ず余白が
    /// 残るので、本源的価値の折れ点と交点は常に枠の中に入る。lo/hi が潰れる・非有限になる
    /// （σ = 0、または t = 0 で全パスが S0）ときだけ [0.5 K, 1.5 K] に落とす。
    void store_fit_curve(std::size_t step) noexcept {
        const std::size_t k  = std::min<std::size_t>(step, Snapshot::kPts - 1);
        const double      q5 = snap_.quantiles[0 * Snapshot::kPts + k];
        const double      q95 = snap_.quantiles[(Snapshot::kQuant - 1) * Snapshot::kPts + k];

        double lo = std::min(q5, 0.8 * active_.K);
        double hi = std::max(q95, 1.1 * active_.K);
        if (!(hi > lo) || !std::isfinite(lo) || !std::isfinite(hi)) {
            lo = 0.5 * active_.K;
            hi = 1.5 * active_.K;
        }
        const double dx = (hi - lo) / static_cast<double>(Snapshot::kFit - 1);
        for (std::size_t j = 0; j < Snapshot::kFit; ++j) {
            const double s  = lo + dx * static_cast<double>(j);
            snap_.fit_s[j]  = s;
            snap_.fit_v[j]  = lsm_.continuation_value(s);  // フィットが無ければ NaN
        }
    }

    double           maturity_;
    double           s0_;
    core::OptionType type_;
    std::uint64_t    seed_;
    Params           active_;

    core::Lsm                              lsm_;
    std::array<std::size_t, Snapshot::kShow> shown_index_{};  ///< 描画する 16 本のパス番号

    std::uint64_t seq_ = 0;
    Snapshot      snap_{};
};

static_assert(bridge::Model<LsmModel>, "LsmModel must satisfy the Model contract");

}  // namespace quantviz::scenes
