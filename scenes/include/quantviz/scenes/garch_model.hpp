#pragma once
// scenes/garch_model.hpp — シーン 3「GARCH」: 合成 GARCH(1,1) 過程のローリング窓オンライン最尤推定
//
// 真のパラメータ (ω, α, β) から 1 ステップ = 1 営業日のリターンを 1 本ずつ生成し（σ²_t の漸化式を
// インクリメンタルに保つので、ステップごとにパス全体を作り直さない）、直近 window 本のローリング窓に
// 対して refit_every ステップごとに `core::garch_fit_into` で再当てはめする。描画が必要とするもの
// （σ_t の 3 本、尤度面 L(α, β) の固定格子、最適化の軌跡）だけを縮約して POD の Snapshot に載せる。
//
// ステップ予算（Release, 窓 500, 10 ステップに 1 回の再当てはめ, max_iter 200 の実測）:
// 平均 0.15 ms、最悪 2〜3 ms（Nelder–Mead）/ 約 4 ms（BFGS）。最悪値が出るのは当てはめステップだけで、
// 100 steps/s の既定設定では 1 フレーム（16.6 ms）に 1 回しか起きない。
//
// 設計上の選択（根拠）:
// * **尤度格子は ω = ω̂ の断面**。各格子点で ω を再最適化した profile likelihood にすると 1024 回の
//   最適化になりステップ予算を桁違いに超える。ω を推定値に固定した断面なら格子の最大点は
//   ちょうど当てはめ結果 (α̂, β̂) に一致する（3 次元の最大点は ω = ω̂ の断面でも最大だから）ので、
//   「軌跡が谷を登って推定値に着く」という教材としての意味も保てる（GARCH-10）。
// * **非定常な格子点は NaN / −∞ ではなく格子内の有限最小値でクランプ**する。ImPlot の heatmap は
//   非有限値を色に写せず、カラースケールも壊れるため（描画側で毎フレーム直すのは責務が逆）。
// * **真値の α+β は常に kMaxPersistence 未満**に保つ。UI から α または β を上げて α+β ≥ 1 に
//   なった場合は (α, β) を比例縮小する（比を保つので「持続性だけを下げた」形になる）。
// * **ヒープ**: 再当てはめは `core::garch_fit_into` に自前の `GarchFit` を渡すので、`OptimResult::path`
//   の容量が再利用され 2 回目以降は確保が起きない（最初の 1 回だけ）。ホットパスのゼロアロケーション
//   規則は「定常状態では確保なし」として満たす。

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
#include "quantviz/core/rng.hpp"
#include "quantviz/core/stats/garch.hpp"
#include "quantviz/core/stats/optim.hpp"

namespace quantviz::scenes {

struct GarchSnapshot {
    static constexpr std::size_t kGrid = 32;  ///< 尤度面 L(α, β) の固定格子（kGrid × kGrid）
    static constexpr std::size_t kPath = 64;  ///< 最適化軌跡の最新 K 点

    /// 尤度格子の添字。row-major で 行 = β の添字、列 = α の添字（描画側は行を反転して使う）。
    static constexpr std::size_t grid_index(std::size_t ia, std::size_t ib) noexcept {
        return ib * kGrid + ia;
    }

    double t      = 0.0;  ///< sim 時刻（年）
    double r_last = 0.0;  ///< 直近ステップのリターン r_t

    double sigma2_true     = 0.0;  ///< r_last を生成した真の条件付き分散 σ²_t
    double sigma2_filtered = 0.0;  ///< 真値パラメータで窓をフィルタした σ²_t（窓が埋まるまで 0）
    double sigma2_est      = 0.0;  ///< 推定パラメータで窓をフィルタした σ²_t（同上）

    core::GarchParams true_params{};  ///< 現在の真値（参照線用）
    core::GarchParams est_params{};   ///< 直近の最尤推定値
    double            log_lik = 0.0;  ///< 推定値での窓の対数尤度

    std::array<double, kGrid>         grid_alpha{};   ///< 格子の α 軸（昇順・等間隔）
    std::array<double, kGrid>         grid_beta{};    ///< 格子の β 軸（昇順・等間隔）
    std::array<double, kGrid * kGrid> loglik_grid{};  ///< L(α, β)（ω = ω̂ の断面、全値有限）

    std::array<double, kPath> path_alpha{};  ///< 最適化軌跡（(α, β) 座標、最新 path_len 点）
    std::array<double, kPath> path_beta{};

    std::uint32_t path_len   = 0;  ///< 軌跡の有効点数（0 = まだ当てはめていない）
    std::uint32_t path_iters = 0;  ///< 直近の当てはめの反復数（間引く前の軌跡の点数）
    std::uint32_t window     = 0;  ///< ローリング窓長（本）
    std::uint8_t  optimizer  = 0;  ///< 0 = Nelder–Mead, 1 = BFGS
    std::uint64_t seq        = 0;  ///< ステップ通番（0 = 未ステップ／Reset 直後）
};
static_assert(std::is_trivially_copyable_v<GarchSnapshot>);
// M1 の例外: 128 B ではなく 32 KiB 上限（固定格子を載せるため。M2 の triple buffer までの暫定）
static_assert(sizeof(GarchSnapshot) <= 32 * 1024);

class GarchModel {
public:
    using Snapshot = GarchSnapshot;

    enum Param : std::uint32_t {
        kOmega     = 1,
        kAlpha     = 2,
        kBeta      = 3,
        kOptimizer = 4,  ///< 0 = Nelder–Mead, 1 = BFGS
        kWindow    = 5,
    };

    static constexpr std::size_t kMaxWindow = 2048;  ///< ローリング窓の容量
    static constexpr std::size_t kMinWindow = 50;    ///< これ未満では尤度面が意味を持たない

    /// 真値の α+β の上限。1 との差 5e-4 は格子の β 刻み（≈ 1.6e-2）より十分小さい。
    static constexpr double kMaxPersistence = 0.9995;
    static constexpr double kMinOmega       = 1e-12;  ///< ω > 0（定常条件）を保つ下限
    static constexpr double kMaxOmega       = 1.0;    ///< 日次分散 1（= 日次ボラ 100 %）を上限に

    // 尤度格子の範囲（日次 GARCH の実用域。真値 α = 0.08, β = 0.90 は内部に入る）
    static constexpr double kGridAlphaMin = 0.001;
    static constexpr double kGridAlphaMax = 0.4;
    static constexpr double kGridBetaMin  = 0.5;
    static constexpr double kGridBetaMax  = 0.999;

    /// 最初の当てはめの初期値。真値から離した、変換の像の内部にある固定点（決定性のため固定）。
    static constexpr core::GarchParams kInitialEstimate{1e-6, 0.05, 0.85};

    struct Config {
        core::GarchParams  truth{1e-6, 0.08, 0.90};  ///< 生成に使う真のパラメータ
        std::size_t        window      = 500;        ///< ローリング窓長（本）
        std::size_t        refit_every = 10;         ///< 何ステップごとに再当てはめするか
        bool               use_bfgs    = false;      ///< false = Nelder–Mead
        std::uint64_t      seed        = 42;
        /// ローリング再当てはめ用に反復上限を下げてある（core の推奨値: garch.hpp のコスト表）。
        /// 前回の推定値から始めるので 200 反復で十分収束し、推定値は 1000 反復と一致する
        /// （窓 500・両最適化器で確認）。最悪ステップは NM 2〜3 ms / BFGS 約 4 ms
        /// （1000 反復なら 4 / 9 ms）。
        core::OptimOptions optim{.max_iter = 200};
    };

    // GCC の既定引数バグ回避のため委譲コンストラクタにする（CLAUDE.md）
    GarchModel() : GarchModel(Config{}) {}

    explicit GarchModel(Config cfg)
        : truth_(cfg.truth),
          optim_(cfg.optim),
          rng_(cfg.seed),
          seed_(cfg.seed),
          window_(std::clamp(cfg.window, kMinWindow, kMaxWindow)),
          refit_every_(cfg.refit_every == 0 ? 1 : cfg.refit_every),
          use_bfgs_(cfg.use_bfgs) {
        sanitize_truth();
        for (std::size_t i = 0; i < GarchSnapshot::kGrid; ++i) {
            grid_alpha_[i] = axis_value(kGridAlphaMin, kGridAlphaMax, i);
            grid_beta_[i]  = axis_value(kGridBetaMin, kGridBetaMax, i);
        }
        reset(seed_);
    }

    /// 1 ステップ = リターン 1 本。σ² の漸化式は保持したまま更新する（パス全体は作り直さない）。
    void step(double dt) {
        t_ += dt;
        if (!(sigma2_next_ >= 0.0)) sigma2_next_ = 0.0;  // 不正な状態でも sqrt を NaN にしない
        sigma2_now_  = sigma2_next_;                     // r_last を生成した条件付き分散
        r_last_      = std::sqrt(sigma2_now_) * rng_.normal();
        sigma2_next_ = truth_.omega + truth_.alpha * r_last_ * r_last_ + truth_.beta * sigma2_now_;
        push(r_last_);
        ++seq_;

        if (filled_ >= window_) {  // 窓が埋まってから推定を始める
            view_n_ = build_view();
            refresh_filters();
            if (++since_refit_ >= refit_every_) {
                since_refit_ = 0;
                refit();
            }
        } else {
            invalidate_estimate();  // 窓を伸ばした直後など: 古い窓の推定を publish し続けない
        }
    }

    Snapshot snapshot() const noexcept {
        Snapshot s;
        s.t               = t_;
        s.r_last          = r_last_;
        s.sigma2_true     = sigma2_now_;
        s.sigma2_filtered = sigma2_filtered_;
        s.sigma2_est      = sigma2_est_;
        s.true_params     = truth_;
        s.est_params      = est_;
        s.log_lik         = log_lik_;
        s.grid_alpha      = grid_alpha_;
        s.grid_beta       = grid_beta_;
        s.loglik_grid     = loglik_grid_;
        s.path_alpha      = path_alpha_;
        s.path_beta       = path_beta_;
        s.path_len        = static_cast<std::uint32_t>(path_len_);
        s.path_iters      = static_cast<std::uint32_t>(path_iters_);
        s.window          = static_cast<std::uint32_t>(window_);
        s.optimizer       = static_cast<std::uint8_t>(use_bfgs_ ? 1 : 0);
        s.seq             = seq_;
        return s;
    }

    void apply(const bridge::Command& c) {
        switch (c.type) {
            case bridge::CommandType::SetParam:
                switch (c.param_id) {
                    case kOmega:
                        truth_.omega = clamp_ui(c.value, kMinOmega, kMaxOmega);
                        break;
                    case kAlpha:
                        truth_.alpha = clamp_ui(c.value, 0.0, kMaxPersistence);
                        sanitize_truth();
                        break;
                    case kBeta:
                        truth_.beta = clamp_ui(c.value, 0.0, kMaxPersistence);
                        sanitize_truth();
                        break;
                    case kOptimizer: use_bfgs_ = (c.value >= 0.5); break;
                    case kWindow:    set_window(c.value); break;
                    default:         break;  // 未知の param_id は無視
                }
                break;
            case bridge::CommandType::Reset: reset(c.seed != 0 ? c.seed : seed_); break;
            default:                         break;  // 時計系は Runner が処理済み
        }
    }

    /// 過程・窓・推定値を初期状態へ。真値 (ω, α, β) と窓長・最適化器の選択は保持する。
    void reset(std::uint64_t seed) {
        seed_ = seed;
        rng_.reseed(seed);
        t_               = 0.0;
        r_last_          = 0.0;
        sigma2_next_     = core::garch_unconditional_variance(truth_);
        sigma2_now_      = sigma2_next_;
        head_   = 0;
        filled_ = 0;
        clear_estimate();  // 窓が埋まるまでは推定を出さない（描画側は 0 / path_len == 0 を見る）
        seq_ = 0;
    }

    // テスト用の読み取りアクセサ
    core::GarchParams truth() const noexcept { return truth_; }
    core::GarchParams estimate() const noexcept { return est_; }
    std::size_t       window() const noexcept { return window_; }

private:
    /// 格子軸の i 番目（両端を含む等間隔）。
    static constexpr double axis_value(double lo, double hi, std::size_t i) noexcept {
        return lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(GarchSnapshot::kGrid - 1);
    }

    /// UI 値を [lo, hi] に入れる。NaN は lo に落ちる（比較が偽になるため）。
    static constexpr double clamp_ui(double v, double lo, double hi) noexcept {
        if (!(v > lo)) return lo;
        if (v > hi) return hi;
        return v;
    }

    /// 真値を定常領域の内部へ。α+β ≥ kMaxPersistence なら (α, β) を比例縮小する（比は保つ）。
    void sanitize_truth() noexcept {
        truth_.omega = clamp_ui(truth_.omega, kMinOmega, kMaxOmega);
        truth_.alpha = clamp_ui(truth_.alpha, 0.0, kMaxPersistence);
        truth_.beta  = clamp_ui(truth_.beta, 0.0, kMaxPersistence);
        const double s = truth_.alpha + truth_.beta;
        if (s >= kMaxPersistence) {
            const double k = kMaxPersistence / s;
            truth_.alpha *= k;
            truth_.beta *= k;
        }
    }

    void set_window(double v) noexcept {
        const double n =
            std::round(clamp_ui(v, static_cast<double>(kMinWindow), static_cast<double>(kMaxWindow)));
        window_      = static_cast<std::size_t>(n);
        since_refit_ = 0;
    }

    void push(double r) noexcept {
        buf_[head_] = r;
        head_       = (head_ + 1) % kMaxWindow;
        if (filled_ < kMaxWindow) ++filled_;
    }

    /// 循環バッファの最新 n 本を時系列順に view_ へ写す（core の span API に渡すため）。
    std::size_t build_view() noexcept {
        const std::size_t n     = std::min(window_, filled_);
        const std::size_t start = (head_ + kMaxWindow - n) % kMaxWindow;
        const std::size_t first = std::min(n, kMaxWindow - start);
        const auto        b     = buf_.begin();
        std::copy_n(b + static_cast<std::ptrdiff_t>(start), first, view_.begin());
        std::copy_n(b, n - first, view_.begin() + static_cast<std::ptrdiff_t>(first));
        return n;
    }

    std::span<const double> view() const noexcept { return {view_.data(), view_n_}; }

    /// 真値と推定値で窓をフィルタし、最新の σ²_t（= r_last の条件付き分散）を取り出す。
    void refresh_filters() noexcept {
        if (view_n_ == 0) return;
        const std::span<double> out{s2_scratch_.data(), view_n_};
        core::garch_filter(truth_, view(), out);
        sigma2_filtered_ = out[view_n_ - 1];
        core::garch_filter(est_, view(), out);
        sigma2_est_ = out[view_n_ - 1];
    }

    /// 窓に対する最尤推定 → 軌跡（(α, β) 座標）→ 尤度格子の更新。
    /// fit_ を使い回すので path / scratch の容量が再利用され、2 回目以降はヒープ確保が起きない。
    void refit() {
        core::garch_fit_into(fit_, view(), est_, optim_, use_bfgs_);
        est_     = fit_.params;
        log_lik_ = fit_.log_lik;

        // 軌跡は無制約 θ で記録されているので (α, β) に戻す。kPath に収まらない長さのときは
        // **先頭から stride 間隔で間引く**。最後の kPath 点だけ載せると、収束後の点だけが残って
        // 軌跡が推定値の上の 1 ドットに潰れる（冷たい当てはめ 114 点の末尾 64 点は α が 2e-4 幅）。
        // 終点は必ず path.back()（core の契約で optim.x = 推定値）を書くので GARCH-11 は保たれる。
        const auto&       path   = fit_.optim.path;  // core の契約により空にならない
        const std::size_t n      = path.size();
        const std::size_t stride = (n + GarchSnapshot::kPath - 1) / GarchSnapshot::kPath;
        path_iters_              = n;
        path_len_                = 0;
        for (std::size_t i = 0; i + 1 < n && path_len_ + 1 < GarchSnapshot::kPath; i += stride)
            store_path_point(core::garch_from_unconstrained(path[i]));
        store_path_point(core::garch_from_unconstrained(path[n - 1]));  // 終点 = 推定値
        build_grid();
    }

    void store_path_point(core::GarchParams p) noexcept {
        path_alpha_[path_len_] = p.alpha;
        path_beta_[path_len_]  = p.beta;
        ++path_len_;
    }

    /// 「今の窓の推定値」を捨てる。窓が埋まっていない間（起動直後・Reset 直後・窓を伸ばした直後）は
    /// 推定が定義できないので、古い値を publish し続けて描画側に平坦な線を最大 kMaxWindow ステップ
    /// 描かせない。描画側は σ² = 0 を描かず、path_len == 0 を尤度面のプレースホルダに使う。
    void clear_estimate() noexcept {
        view_n_          = 0;
        sigma2_filtered_ = 0.0;
        sigma2_est_      = 0.0;
        est_             = kInitialEstimate;  // 次の当てはめは Reset 直後と同じ固定初期値から
        log_lik_         = 0.0;
        path_len_        = 0;
        path_iters_      = 0;
        since_refit_     = 0;
        loglik_grid_.fill(0.0);
    }

    /// clear_estimate() を毎ステップ繰り返さない版（窓が埋まるまで毎ステップ呼ばれるため）。
    void invalidate_estimate() noexcept {
        if (view_n_ == 0 && path_len_ == 0) return;  // すでに無効
        clear_estimate();
    }

    /// ω = ω̂ の断面での対数尤度格子。非定常な点（α+β ≥ 1）は格子内の有限最小値にクランプする。
    void build_grid() noexcept {
        double lo = std::numeric_limits<double>::infinity();
        for (std::size_t ib = 0; ib < GarchSnapshot::kGrid; ++ib) {
            for (std::size_t ia = 0; ia < GarchSnapshot::kGrid; ++ia) {
                const core::GarchParams p{est_.omega, grid_alpha_[ia], grid_beta_[ib]};
                const double            ll = core::garch_log_likelihood(p, view());
                loglik_grid_[GarchSnapshot::grid_index(ia, ib)] = ll;
                if (std::isfinite(ll) && ll < lo) lo = ll;
            }
        }
        if (!std::isfinite(lo)) lo = 0.0;  // 全点が非定常（格子の張り方が壊れている場合の保険）
        for (double& v : loglik_grid_)
            if (!std::isfinite(v)) v = lo;
    }

    // ---- 真の過程 ----
    core::GarchParams  truth_;
    core::OptimOptions optim_;
    core::Rng          rng_;
    std::uint64_t      seed_;
    double             t_           = 0.0;
    double             r_last_      = 0.0;
    double             sigma2_now_  = 0.0;  ///< r_last を生成した σ²_t
    double             sigma2_next_ = 0.0;  ///< 次のステップで使う σ²_{t+1}

    // ---- ローリング窓（循環バッファ + core に渡す連続バッファ） ----
    std::array<double, kMaxWindow> buf_{};
    std::array<double, kMaxWindow> view_{};
    std::array<double, kMaxWindow> s2_scratch_{};
    std::size_t                    head_   = 0;
    std::size_t                    filled_ = 0;
    std::size_t                    view_n_ = 0;
    std::size_t                    window_;
    std::size_t                    refit_every_;
    std::size_t                    since_refit_ = 0;
    bool                           use_bfgs_;

    // ---- 推定 ----
    using Grid = std::array<double, GarchSnapshot::kGrid>;
    using Surf = std::array<double, GarchSnapshot::kGrid * GarchSnapshot::kGrid>;
    using Path = std::array<double, GarchSnapshot::kPath>;

    core::GarchFit    fit_{};  ///< 再当てはめの作業領域（path / scratch の容量を使い回す）
    core::GarchParams est_             = kInitialEstimate;
    double            log_lik_         = 0.0;
    double            sigma2_filtered_ = 0.0;
    double            sigma2_est_      = 0.0;
    Grid              grid_alpha_{};
    Grid              grid_beta_{};
    Surf              loglik_grid_{};
    Path              path_alpha_{};
    Path              path_beta_{};
    std::size_t       path_len_   = 0;
    std::size_t       path_iters_ = 0;

    std::uint64_t seq_ = 0;
};

static_assert(bridge::Model<GarchModel>, "GarchModel must satisfy the Model contract");

}  // namespace quantviz::scenes
