#pragma once
// scenes/hjb_model.hpp — シーン「Merton HJB」: CRRA 効用の最適投資問題の HJB を満期から後退に
// 1 時間反復ずつ手送りし、今日へ向かって伸びていく V(w, t) の面と、最適比率 π*(w) を吐く Model。
//
// 責務: `core::HjbMerton` を 1 台抱え、`step()` 1 回で `step_backward()` 1 回を回す。描画側には 2 系統:
//
//   1. `Snapshot`（リング）: **現在の時刻断面**だけ。富の格子 256 点に対する V(w)・π*(w)、そこへ重ねる
//      閉形式の参照（π* の解析定数と V の閉形式）、内側 90 % での π* の最大誤差、反復数・残り、
//      参照線に要るパラメータ真値。≈ 10 KB（M1 Greeks と同じ「32 KiB まで」の例外枠の内側）。
//   2. `Surface`（`bridge::TripleBuffer`）: **これまでに解けた時間レベル全部**を 200×200 の float 格子で。
//      row-major [iT][iW]、行 0 が満期 U(w)。まだ解いていない行はゼロで埋める（下記「未計算行」）。
//
// 面の出所はモデルが持つ「レベル・バッファ」= 反復が 1 つ終わるごとに V(w) を 200 列ぶん控える
// (n_t + 1) × 200 の double 配列（既定で 201×200 = 約 314 KiB）。確保はコンストラクタと
// `SetParam` による init のやり直しだけで、`step` / `snapshot` / `surface` / `apply` はどれも確保しない。
//
// 【格子と配列の関係】`core::HjbMerton` の富の節点数は `HjbSnapshot::kNodes` = 256 に固定する。
// こうすると Snapshot の 3 本（wealth / values / pi_star）は**格子の値そのもの**（補間なし）になり、
// 「素の HjbMerton を同じパラメータで回した結果と bit 一致」というテストが書ける（HJBSCENE-01）。
// 面の列は 200 しかないので、節点を `node_of_column()` で間引く（補間しない = 面も節点の値そのもの）。
//
// 【時間レベルと面の行の対応】掃引は n_t + 1 個のレベル（レベル 0 = 満期 … レベル n_t = t 0）を生むが、
// 面は kT = 200 行しか持てない。FDM シーンと同じ規則で、行 j にはレベル `round(j · n_t / (kT − 1))` を
// 載せる: 単調非減少で、行 0 は必ずレベル 0（= 終端効用 U(w)）、最終行は必ずレベル n_t（= t = 0）になる。
// 既定（n_t = 200）では 1 レベルだけが間引かれ、n_t < kT では同じレベルが複数行に写る。
//
// 【未計算行】まだ解いていない行はゼロで埋める（FDM シーンと同じ）。ゼロ平面に対して計算済みの領域が
// 楔形に伸びるほうが、後退反復が今日へ進んでいく様子が一目で分かる。V は γ > 1 で負なので「ゼロ平面の
// 下に解が伸びる」形になる（V = w^{1−γ}/(1−γ) < 0）。
//
// 【パラメータの反映】FDM シーンと同じく、`SetParam` は**その場で `init` をやり直す**: 後退掃引の途中で
// 係数（μ, r, σ, γ）を差し替えた面は何の解でもない。満期へ巻き戻すので `seq` も 0 に戻る（= Reset 扱い）。
// 値が実際に動かない SetParam（同じ値の再送・クランプで同着）では巻き戻さない。NaN / ±∞ は拒否する
// （下限に倒すと UI の事故で面が別物になる）。
//
// 【γ の 1 への吸着】CRRA 効用と閉形式は γ = 1 で対数分岐に切り替わる（U = ln w、V = ln w + κτ）。
// 分岐は `gamma == 1.0` の厳密比較で、γ = 1 ± 1e-7 のときは w^{1−γ}/(1−γ) の分子が
// (1−γ) ln w + O((1−γ)²) と激しく相殺して有効数字を 7 桁ほど失う。スライダーが 1 の近くを通るだけで
// 参照線が暴れるのは教材として害しかないので、|γ − 1| < kGammaSnap (1e-4) は厳密に 1.0 へ吸着させる。
// 吸着の幅は「数値解の空間誤差（1e-5 程度）より大きく、スライダーの見た目の刻み（1e-2）より小さい」
// ところに置いた。
//
// 【π* の誤差】`max_abs_pi_error` は**内側 90 %**（両端 5 % を除く）で測る。境界節点の π* は
// Dirichlet 境界に押し付けられた誤差の勾配を拾って片側差分ぶん（1e-4 程度）ずれる
// （`hjb_merton.hpp` の「境界」参照）ので、全節点で測ると「境界の 2 点だけが 10 倍悪い」値になり、
// 内側の収束具合が読めなくなる。

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include "quantviz/bridge/command.hpp"
#include "quantviz/bridge/model_concept.hpp"
#include "quantviz/core/exec/hjb_merton.hpp"

namespace quantviz::scenes {

/// 現在の時刻断面。256 点 × 5 本で ≈ 10 KB。
struct HjbSnapshot {
    /// 富の格子の節点数（= `core::HjbParams::n_w`）。Snapshot の配列長でもある。
    static constexpr std::size_t kNodes = 256;

    /// 現在の時刻 t（= `HjbMerton::time()`）。「掃引が今日 t = 0 に届くまでに残っている年数」。
    /// 満期（反復 0）で T、解き終わりで 0。閉形式の残存期間は τ = T − t_remaining。
    double t_remaining = 0.0;
    /// 内側 90 %（両端 5 % を除く）での max |π*_numeric − π*_analytic|（ヘッダ冒頭「π* の誤差」）。
    double max_abs_pi_error = 0.0;

    double mu    = 0.0;  ///< 以下 5 つは参照線・テレメトリ用のパラメータ真値（sanitize 済み）
    double r     = 0.0;
    double sigma = 0.0;
    double gamma = 0.0;
    double T     = 0.0;

    std::array<double, kNodes> wealth{};       ///< 富の格子 w_i（対数等間隔・昇順、両端は w_min / w_max）
    std::array<double, kNodes> values{};       ///< 現在時刻の数値解 V(w_i, t)
    std::array<double, kNodes> pi_star{};      ///< 現在時刻の数値解 π*(w_i)（[0, kPiMax]）
    /// 解析解 π* = (μ−r)/(γσ²) を制約域に丸めたもの。w に依らない定数だが、パネルが重ね描きしやすい
    /// ように配列に詰めておく（全要素が同じ値）。
    std::array<double, kNodes> analytic_pi{};
    /// 閉形式 V(w_i, t)（`core::merton_value`、現在時刻 t = t_remaining）。
    std::array<double, kNodes> analytic_v{};

    std::uint64_t seq = 0;  ///< ステップ通番（0 = 未ステップ／Reset 直後／パラメータ変更直後）

    std::uint32_t iteration = 0;  ///< 完了した後退反復数
    std::uint32_t remaining = 0;  ///< 残り反復数（0 = t = 0 まで解けた）
};

static_assert(std::is_trivially_copyable_v<HjbSnapshot>);
static_assert(std::is_standard_layout_v<HjbSnapshot>);
static_assert(sizeof(HjbSnapshot) <= 16 * 1024, "Snapshot 上限（計画書 Task 6 の型契約）");
// kNodes は `core::hjb_sanitize` が素通しする範囲（n_w ∈ [8, 4096]）に収まっていなければならない。
// 外れると sanitize が黙って n_w を書き換え、格子の節点数と Snapshot の配列長が食い違う
// （publish() は w[i] / v[i] / pi[i] を i < kNodes で読む）。
static_assert(HjbSnapshot::kNodes >= 8 && HjbSnapshot::kNodes <= 4096,
              "n_w must pass through core::hjb_sanitize unchanged");

/// 3D 面。`bridge::TripleBuffer` の使い回しスロットに書くので、`surface()` は毎回全フィールドを書く。
struct HjbSurface {
    static constexpr std::size_t kW = 200;  ///< 富の方向の列数（格子 256 節点から間引く）
    static constexpr std::size_t kT = 200;  ///< 時間方向の行数（行 0 = 満期）

    std::array<float, kW>      wealth{};  ///< w 軸（昇順。対数等間隔なので線形軸では右に寄る）
    std::array<float, kT>      times{};   ///< 各行の t（降順、行 0 = T、最終行 = 0）
    std::array<float, kW * kT> values{};  ///< V。row-major、index = iT * kW + iW。未計算行は 0
    std::uint32_t              filled_rows = 0;  ///< 先頭から何行が計算済みか（≥ 1）
};

static_assert(std::is_trivially_copyable_v<HjbSurface>);
static_assert(std::is_standard_layout_v<HjbSurface>);

class HjbModel {
public:
    using Snapshot = HjbSnapshot;
    using Surface  = HjbSurface;

    enum Param : std::uint32_t {
        kMu    = 1,  ///< μ（[kMinMu, kMaxMu]）
        kRate  = 2,  ///< r（[kMinRate, kMaxRate]）
        kSigma = 3,  ///< σ（[kMinSigma, kMaxSigma]）
        kGamma = 4,  ///< γ（[kMinGamma, kMaxGamma]、1 の近傍は 1.0 へ吸着）
    };

    // スライダーの範囲と一致させること（`viz/src/panels/hjb_panel.cpp`）。
    static constexpr double kMinMu    = -0.05;
    static constexpr double kMaxMu    = 0.30;
    static constexpr double kMinRate  = 0.0;
    static constexpr double kMaxRate  = 0.10;
    /// σ = 0 は HJB が退化する（拡散項が消えて π が決まらない）ので下限は正の値。
    static constexpr double kMinSigma = 0.05;
    static constexpr double kMaxSigma = 0.60;
    /// γ → 0 は危険中立（π* が上限に張り付く）、γ → ∞ は全額無リスク。教材として見える範囲に絞る。
    static constexpr double kMinGamma = 0.20;
    static constexpr double kMaxGamma = 10.0;
    /// |γ − 1| がこれ未満なら γ = 1.0（対数効用の分岐）へ吸着させる（ヘッダ冒頭「γ の 1 への吸着」）。
    static constexpr double kGammaSnap = 1e-4;

    /// 時間分割数の上限（レベル・バッファのメモリを縛る: (kMaxTime + 1) × kW × 8 B ≈ 8 MB）。
    static constexpr std::size_t kMinTime = 1;
    static constexpr std::size_t kMaxTime = 5000;
    // kNodes と同じ理由で、n_t も `core::hjb_sanitize` が素通しする範囲（n_t ≤ 100000）に収める。
    // 外れると sanitize が n_t を縮め、レベル・バッファの行数（n_t + 1）と掃引の長さが食い違う。
    static_assert(kMaxTime <= 100000, "n_t must pass through core::hjb_sanitize unchanged");

    struct Config {
        double      mu    = 0.08;  ///< 既定は `core::kHjbDefaults`（π* = 0.05 / 0.12 = 0.4167）
        double      r     = 0.03;
        double      sigma = 0.20;
        double      gamma = 3.0;
        /// T / w_min / w_max はスライダーに出さない（掃引の長さと格子の張り方）。不正値は
        /// `core::hjb_sanitize` が決定的に丸める（T ≥ 1e-6、w_min ≥ 1e-6、w_max > w_min、NaN は既定値）。
        double      T     = 1.0;
        double      w_min = 0.2;
        double      w_max = 5.0;
        /// 時間ステップ数（= 後退反復の回数）。面の行数 kT と同じにすると「1 反復 = 面 1 行」。
        std::size_t n_t   = HjbSurface::kT;
    };

    // GCC のバグ回避: 既定メンバ初期化子を持つ入れ子 Config を既定引数にしない（CLAUDE.md）。
    HjbModel() : HjbModel(Config{}) {}

    explicit HjbModel(Config cfg)
        : n_t_(std::clamp(cfg.n_t, kMinTime, kMaxTime)),
          grid_(sanitized(cfg, n_t_)),
          active_{grid_.mu, grid_.r, grid_.sigma, grid_.gamma},
          col_w_(Surface::kW),
          levels_((n_t_ + 1) * Surface::kW) {
        init_solver();
    }

    /// 1 ステップ = 後退反復 1 回。`dt` は使わない（時間の刻みは格子が決める）。
    /// t = 0 まで解き終わったあとは何もしないが、`seq` は進める: Runner から見れば「1 ステップ実行した」
    /// のは事実で、ここで seq を止めると描画側のレート計とテレメトリが嘘をつく（FDM シーンと同じ）。
    void step(double /*dt*/) {
        if (hjb_.step_backward()) {
            ++iteration_;
            store_level(iteration_);
        }
        ++seq_;
        publish();
    }

    Snapshot snapshot() const noexcept { return snap_; }

    /// 現在の面を s に書く。TripleBuffer の使い回しスロットなので**全フィールド**を書く。
    void surface(Surface& s) const noexcept {
        const std::size_t filled = filled_rows();
        for (std::size_t j = 0; j < Surface::kW; ++j) s.wealth[j] = static_cast<float>(col_w_[j]);

        const double inv_n = 1.0 / static_cast<double>(n_t_);  // grid_.T は sanitize 済み（有限・> 0）
        for (std::size_t row = 0; row < Surface::kT; ++row) {
            const std::size_t level = level_of_row(row);
            s.times[row] = static_cast<float>(grid_.T * static_cast<double>(n_t_ - level) * inv_n);

            float* const dst = s.values.data() + row * Surface::kW;
            if (row < filled) {
                const double* const src = levels_.data() + level * Surface::kW;
                for (std::size_t j = 0; j < Surface::kW; ++j) dst[j] = static_cast<float>(src[j]);
            } else {
                for (std::size_t j = 0; j < Surface::kW; ++j) dst[j] = 0.f;
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

    /// 終端効用へ巻き戻す（= `HjbMerton::init` のやり直し）。パラメータは保持する。
    /// このシーンに乱数は無いので seed は使わない。
    void reset(std::uint64_t /*seed*/ = 0) { init_solver(); }

    /// 面の列 j が載せる格子節点（単調増加、列 0 = 節点 0、最終列 = 節点 kNodes − 1）。
    /// テストが「面と同じ間引き」を再現できるよう公開している。
    [[nodiscard]] static constexpr std::size_t node_of_column(std::size_t col) noexcept {
        constexpr std::size_t d = Surface::kW - 1;
        return (col * (Snapshot::kNodes - 1) + d / 2) / d;
    }

    /// 面の行 row が載せる時間レベル（ヘッダ冒頭「時間レベルと面の行の対応」）。単調非減少、
    /// level_of_row(0) = 0、level_of_row(kT − 1) = n_t。
    [[nodiscard]] std::size_t level_of_row(std::size_t row) const noexcept {
        constexpr std::size_t d = Surface::kT - 1;
        return (row * n_t_ + d / 2) / d;
    }

    // ------------------------------------------------------------------ テスト用の読み取りアクセサ
    [[nodiscard]] const core::HjbMerton& solver() const noexcept { return hjb_; }
    [[nodiscard]] std::size_t            iteration() const noexcept { return iteration_; }

private:
    struct Params {
        double mu    = 0.0;
        double r     = 0.0;
        double sigma = 0.0;
        double gamma = 0.0;

        friend bool operator==(const Params&, const Params&) = default;
    };

    // ------------------------------------------------------------------ 入力の丸め（例外なし）
    /// 有限値は [lo, hi] に丸め、NaN / ∞ は**拒否**して現在値を返す（FDM シーンと同じ流儀）。
    static double clamp_param(double v, double lo, double hi, double current) noexcept {
        if (!std::isfinite(v)) return current;
        return v < lo ? lo : (v > hi ? hi : v);
    }

    /// γ = 1 の近傍を対数効用の分岐へ吸着させる（ヘッダ冒頭「γ の 1 への吸着」）。
    static double snap_gamma(double g) noexcept { return std::abs(g - 1.0) < kGammaSnap ? 1.0 : g; }

    /// Config から「格子の真値」を 1 か所で確定させる: μ/r/σ/γ はこのクラスの範囲へ、T / w_min / w_max /
    /// n_w / n_t は `core::hjb_sanitize` へ通す。以後モデルはこの sanitize 済みの値しか使わない
    /// （生の Config を抱えると、例えば T = NaN で `snapshot().T` は 1e-6 なのに面の時間軸だけ NaN、
    /// という食い違いが起きる。sanitize は冪等なので、再 init で値が動くこともない）。
    static core::HjbParams sanitized(const Config& cfg, std::size_t n_t) noexcept {
        core::HjbParams p{};
        p.mu    = clamp_param(cfg.mu, kMinMu, kMaxMu, 0.08);
        p.r     = clamp_param(cfg.r, kMinRate, kMaxRate, 0.03);
        p.sigma = clamp_param(cfg.sigma, kMinSigma, kMaxSigma, 0.20);
        p.gamma = snap_gamma(clamp_param(cfg.gamma, kMinGamma, kMaxGamma, 3.0));
        p.T     = cfg.T;
        p.w_min = cfg.w_min;
        p.w_max = cfg.w_max;
        p.n_w   = Snapshot::kNodes;
        p.n_t   = n_t;
        return core::hjb_sanitize(p);
    }

    /// noexcept は付けない: `init_solver()` が `HjbMerton::init` を呼び、その中は `std::vector::resize`。
    /// 現状は格子（kNodes / n_t_）が生涯不変なので resize は必ず「同じ大きさ」= 確保も例外も起きないが、
    /// その不変条件はこのクラスの都合であって `init` の契約ではない（FDM シーンと同じ理由）。
    void apply_param(std::uint32_t id, double v) {
        Params next = active_;
        switch (id) {
            case kMu:    next.mu = clamp_param(v, kMinMu, kMaxMu, active_.mu); break;
            case kRate:  next.r = clamp_param(v, kMinRate, kMaxRate, active_.r); break;
            case kSigma: next.sigma = clamp_param(v, kMinSigma, kMaxSigma, active_.sigma); break;
            case kGamma: next.gamma = snap_gamma(clamp_param(v, kMinGamma, kMaxGamma, active_.gamma)); break;
            default:     return;  // 未知の param_id は無視
        }
        if (next == active_) return;  // 値が動かないなら掃引を巻き戻さない
        active_ = next;
        init_solver();
    }

    // ------------------------------------------------------------------ 掃引の開始 / レベルの蓄積
    /// 格子とパラメータを組み直し、終端効用（レベル 0）を置いて反復 0 に戻す。
    /// `HjbMerton::init` の `resize` は格子サイズが変わらない限り確保しない（kNodes と n_t_ は不変）。
    void init_solver() {
        core::HjbParams p = grid_;  // T / w_min / w_max / n_w / n_t は sanitize 済みで生涯不変
        p.mu              = active_.mu;
        p.r               = active_.r;
        p.sigma           = active_.sigma;
        p.gamma           = active_.gamma;
        hjb_.init(p);

        const std::span<const double> w = hjb_.wealth();
        for (std::size_t j = 0; j < Surface::kW; ++j) col_w_[j] = w[node_of_column(j)];

        std::fill(levels_.begin(), levels_.end(), 0.0);
        iteration_ = 0;
        seq_       = 0;
        store_level(0);
        publish();
    }

    /// レベル level の V(w) を面の列 200 個ぶん控える（節点そのもの、確保なし）。
    void store_level(std::size_t level) noexcept {
        const std::span<const double> v   = hjb_.values();
        double* const                 dst = levels_.data() + level * Surface::kW;
        for (std::size_t j = 0; j < Surface::kW; ++j) dst[j] = v[node_of_column(j)];
    }

    /// 先頭から何行が計算済みか（level_of_row は単調なので最初に超えた所で止まる）。
    [[nodiscard]] std::size_t filled_rows() const noexcept {
        std::size_t n = 0;
        while (n < Surface::kT && level_of_row(n) <= iteration_) ++n;
        return n;
    }

    // ------------------------------------------------------------------ Snapshot 更新
    void publish() noexcept {
        const core::HjbParams&        p  = hjb_.params();  // sanitize 済みの真値
        const std::span<const double> w  = hjb_.wealth();
        const std::span<const double> v  = hjb_.values();
        const std::span<const double> pi = hjb_.optimal_fraction();
        const double                  t  = hjb_.time();
        const double                  pi_exact = core::merton_fraction(p);

        // 閉形式 V(w, t) = U(w) e^{(1−γ)κτ}（γ = 1 は U(w) + κτ）の w に依らない部分は節点ごとに
        // 計算し直さない: `core::merton_value` をそのまま 256 回呼ぶと hjb_sanitize と merton_kappa も
        // 256 回走る（毎ステップ、20 回/秒）。p は既に sanitize 済みで sanitize は冪等なので、
        // 括り出した式は merton_value と**同じ演算を同じ順序で**行う = bit 一致する。
        const double kappa       = core::merton_kappa(p);
        const double tau         = p.T - t;
        const bool   log_utility = (p.gamma == 1.0);
        const double kappa_tau   = kappa * tau;
        const double growth      = log_utility ? 0.0 : std::exp((1.0 - p.gamma) * kappa * tau);

        for (std::size_t i = 0; i < Snapshot::kNodes; ++i) {
            const double u       = core::crra_utility(w[i], p.gamma);
            snap_.wealth[i]      = w[i];
            snap_.values[i]      = v[i];
            snap_.pi_star[i]     = pi[i];
            snap_.analytic_pi[i] = pi_exact;
            snap_.analytic_v[i]  = log_utility ? u + kappa_tau : u * growth;
        }

        // 内側 90 %（両端 kEdge 点を除く）での最大誤差。
        constexpr std::size_t kEdge = Snapshot::kNodes * 5 / 100;
        double                worst = 0.0;
        for (std::size_t i = kEdge; i + kEdge < Snapshot::kNodes; ++i)
            worst = std::max(worst, std::abs(pi[i] - pi_exact));
        snap_.max_abs_pi_error = worst;

        snap_.t_remaining = t;
        snap_.mu          = p.mu;
        snap_.r           = p.r;
        snap_.sigma       = p.sigma;
        snap_.gamma       = p.gamma;
        snap_.T           = p.T;

        snap_.seq       = seq_;
        snap_.iteration = static_cast<std::uint32_t>(iteration_);
        snap_.remaining = static_cast<std::uint32_t>(hjb_.remaining());
    }

    std::size_t n_t_;
    /// 格子の真値（sanitize 済み）。μ/r/σ/γ だけが `active_` で差し替わり、T / w_min / w_max /
    /// n_w / n_t は生涯不変。面も Snapshot もこの値しか見ない。
    core::HjbParams grid_;

    Params          active_;
    core::HjbMerton hjb_;

    std::vector<double> col_w_;   ///< 面の w 軸（kW 点、double 精度で保持）
    std::vector<double> levels_;  ///< (n_t + 1) × kW の V。行 = 時間レベル（行 0 = 満期）

    std::size_t   iteration_ = 0;
    std::uint64_t seq_       = 0;
    Snapshot      snap_{};
};

static_assert(bridge::Model<HjbModel>, "HjbModel must satisfy the Model contract");
static_assert(bridge::SurfaceModel<HjbModel>, "HjbModel must satisfy SurfaceModel");

}  // namespace quantviz::scenes
