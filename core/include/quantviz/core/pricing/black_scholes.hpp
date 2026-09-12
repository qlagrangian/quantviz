#pragma once
// core/pricing/black_scholes.hpp — 無配当ヨーロピアン・オプションの Black–Scholes 価格と解析 Greeks。
//
//   d1 = (log(S/K) + (r + sigma^2/2) T) / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T)
//   C  = S Phi(d1) - K e^{-rT} Phi(d2),                      P  = K e^{-rT} Phi(-d2) - S Phi(-d1)
//
// 単位: Theta は「年あたり」（Theta = dV/dt = -dV/dT）、Rho は「金利 1.0 あたり」（dV/dr）、
// Vega は「ボラティリティ 1.0 あたり」（dV/dsigma）。1 日 / 1bp / 1% への換算は呼び出し側の責務。
// 退化ケース: T <= 0 または sigma <= 0（NaN を含む）、および S <= 0 / K <= 0 では決定的な割引
// ペイオフ max(S - K e^{-rT}, 0)（Put は max(K e^{-rT} - S, 0)）を返す。Gamma と Vega は 0、
// Delta / Theta / Rho は極限値（有限）。-0.0 は +0.0 に正規化して返す（ビューアの "-0.00" 対策）。
// S > 0, K > 0 の有限入力に対しては、中間量がオーバーフローしない限り NaN を返さない
// （sigma = 1e308, T = 4 のような極端な入力では sigma^2 T が inf になり NaN になりうる）。
//
// bs_price_strip は K だけが動く場合のベクトル化版。超越関数（log / erfc）はスカラ版と同一の
// std:: 関数をレーンごとに呼び、算術も同じ順序で行うため結果はスカラ版とビット一致する
// （BS-10 の許容は相対 1e-15、実測誤差は 0）。Abramowitz–Stegun 型の多項式近似では 1e-15 に
// 届かないので使わない。ベクトル化で効くのは exp(-rT) / sqrt(T) のループ外への追い出しと
// 除算・乗算の並列化だけである。
//
// ビット一致の作り方は 2 つ。(1) 価格の算術は detail::bs_price_block ただ 1 つで、バックエンドは
// ロード / ストアと K > 0 の判定しか持たない（3 つの Ops トレイトが演算を提供する）。
// (2) `x + y*z` / `x - y*z` の形は Ops::madd / nmsub / msub で固定する。FMA に融合されるかどうかは
// 「その中間値が何回使われるか」に依存し（strip ではループ不変量が外に出るので融合されず、
// bs_price では 1 回しか使われないので融合される）、深い OTM の打ち消しで 1 ulp の差が相対 1e-10
// まで増幅されるため。融合の可否は QUANTVIZ_BS_USE_FMA ただ 1 つのゲートで決める。
// 既定ビルド（FMA 命令なし）に加えて -mfma / -march=native / AVX2 intrinsics 経路でもビット一致を
// 確認済み。ただし -ffast-math / -Ofast / FLT_EVAL_METHOD != 0（x87 の 80bit 中間演算）は
// 非サポート: 再結合や超過精度で経路間のビット一致が崩れる（-ffast-math で実測して確認済み）。

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

// SIMD バックエンドの選択。QUANTVIZ_BS_SIMD_BACKEND を外から定義すれば上書きできる
// （0 = スカラ, 1 = AVX2 intrinsics, 2 = std::experimental::simd）。
#if !defined(QUANTVIZ_BS_SIMD_BACKEND)
#if __has_include(<experimental/simd>)
#define QUANTVIZ_BS_SIMD_BACKEND 2
#elif defined(__AVX2__)
#define QUANTVIZ_BS_SIMD_BACKEND 1
#else
#define QUANTVIZ_BS_SIMD_BACKEND 0
#endif
#endif

// FMA を使うかどうかの唯一のゲート。FP_FAST_FMA は <cmath> が定義する標準マクロで、
// 「std::fma がハードウェア 1 命令で速い」ことを表す。__FMA__ でも __AVX2__ でもなくこれを見る:
// -mfma4 -mavx2 のように FP_FAST_FMA と __AVX2__ は立つが __FMA__ が立たない構成があり、
// ゲートを分けるとスカラ経路だけ FMA を出して strip との一致が静かに壊れるため。
#if defined(FP_FAST_FMA)
#define QUANTVIZ_BS_USE_FMA 1
#else
#define QUANTVIZ_BS_USE_FMA 0
#endif

#if QUANTVIZ_BS_SIMD_BACKEND == 1 && !defined(__AVX2__)
#error "quantviz: QUANTVIZ_BS_SIMD_BACKEND=1 needs AVX2 (build with -mavx2)"
#endif
#if QUANTVIZ_BS_SIMD_BACKEND == 1 && QUANTVIZ_BS_USE_FMA && !defined(__FMA__)
#error "quantviz: build with -mfma (FP_FAST_FMA is set but the AVX2 backend has no FMA)"
#endif

#if QUANTVIZ_BS_SIMD_BACKEND == 2
#include <experimental/simd>
#elif QUANTVIZ_BS_SIMD_BACKEND == 1
#include <immintrin.h>
#endif

namespace quantviz::core {

enum class OptionType : std::uint8_t { Call, Put };

struct BsGreeks {
    double price = 0.0;
    double delta = 0.0;
    double gamma = 0.0;  ///< Call / Put で同一
    double vega  = 0.0;  ///< Call / Put で同一。ボラティリティ 1.0 あたり
    double theta = 0.0;  ///< 年あたり（dV/dt）
    double rho   = 0.0;  ///< 金利 1.0 あたり
};

inline constexpr double kInvSqrt2   = 0.70710678118654752440084436210485;
inline constexpr double kInvSqrt2Pi = 0.39894228040143267793994605993438;

namespace detail {

// ---------------------------------------------------------------------------
// Ops トレイト — バックエンドごとの「演算の語彙」。算術はここと bs_price_block にしかない。
// ---------------------------------------------------------------------------

struct ScalarOps {
    using V = double;

    static V splat(double x) noexcept { return x; }
    static V add(const V& a, const V& b) noexcept { return a + b; }
    static V mul(const V& a, const V& b) noexcept { return a * b; }
    static V div(const V& a, const V& b) noexcept { return a / b; }
    static V neg(const V& a) noexcept { return -a; }
    static V log(const V& a) noexcept { return std::log(a); }
    static V erfc(const V& a) noexcept { return std::erfc(a); }

    static V madd(const V& a, const V& b, const V& c) noexcept {  // a*b + c
#if QUANTVIZ_BS_USE_FMA
        return std::fma(a, b, c);
#else
        return a * b + c;
#endif
    }
    static V nmsub(const V& a, const V& b, const V& c) noexcept {  // a - b*c
#if QUANTVIZ_BS_USE_FMA
        return std::fma(-b, c, a);
#else
        return a - b * c;
#endif
    }
    static V msub(const V& a, const V& b, const V& c, const V& d) noexcept {  // a*b - c*d
#if QUANTVIZ_BS_USE_FMA
        return std::fma(a, b, -(c * d));
#else
        return a * b - c * d;
#endif
    }
};

#if QUANTVIZ_BS_SIMD_BACKEND == 2
namespace stdx = std::experimental;

/// 超越関数はレーンごとに std:: の同じ関数を呼ぶ。ベクトル近似ではスカラ版とビット一致せず
/// BS-10（相対 1e-15）を満たせないため。
template <class V, class F>
inline V lanewise(const V& v, F&& f) noexcept {
    V out{};
    for (std::size_t l = 0; l < V::size(); ++l) out[l] = f(static_cast<double>(v[l]));
    return out;
}

struct SimdOps {
    using V = stdx::native_simd<double>;

    static V splat(double x) noexcept { return V(x); }
    static V add(const V& a, const V& b) noexcept { return a + b; }
    static V mul(const V& a, const V& b) noexcept { return a * b; }
    static V div(const V& a, const V& b) noexcept { return a / b; }
    static V neg(const V& a) noexcept { return -a; }
    static V log(const V& a) noexcept {
        return lanewise(a, [](double x) { return std::log(x); });
    }
    static V erfc(const V& a) noexcept {
        return lanewise(a, [](double x) { return std::erfc(x); });
    }

    static V madd(const V& a, const V& b, const V& c) noexcept {
#if QUANTVIZ_BS_USE_FMA
        return stdx::fma(a, b, c);
#else
        return a * b + c;
#endif
    }
    static V nmsub(const V& a, const V& b, const V& c) noexcept {
#if QUANTVIZ_BS_USE_FMA
        return stdx::fma(-b, c, a);
#else
        return a - b * c;
#endif
    }
    static V msub(const V& a, const V& b, const V& c, const V& d) noexcept {
#if QUANTVIZ_BS_USE_FMA
        return stdx::fma(a, b, -(c * d));
#else
        return a * b - c * d;
#endif
    }
};
#endif

#if QUANTVIZ_BS_SIMD_BACKEND == 1
/// lanewise の AVX2 版。関数ポインタではなくテンプレートにして呼び出しをインライン化させる。
template <class F>
inline __m256d lanewise_avx2(__m256d v, F&& f) noexcept {
    alignas(32) double a[4];
    _mm256_store_pd(a, v);
    for (std::size_t l = 0; l < 4; ++l) a[l] = f(a[l]);
    return _mm256_load_pd(a);
}

struct Avx2Ops {
    using V = __m256d;

    static V splat(double x) noexcept { return _mm256_set1_pd(x); }
    static V add(const V& a, const V& b) noexcept { return _mm256_add_pd(a, b); }
    static V mul(const V& a, const V& b) noexcept { return _mm256_mul_pd(a, b); }
    static V div(const V& a, const V& b) noexcept { return _mm256_div_pd(a, b); }
    /// IEEE の符号反転（0 - x ではないので -0.0 も正しく扱える）。
    static V neg(const V& a) noexcept { return _mm256_xor_pd(a, _mm256_set1_pd(-0.0)); }
    static V log(const V& a) noexcept {
        return lanewise_avx2(a, [](double x) { return std::log(x); });
    }
    static V erfc(const V& a) noexcept {
        return lanewise_avx2(a, [](double x) { return std::erfc(x); });
    }

    static V madd(const V& a, const V& b, const V& c) noexcept {
#if QUANTVIZ_BS_USE_FMA
        return _mm256_fmadd_pd(a, b, c);
#else
        return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#endif
    }
    static V nmsub(const V& a, const V& b, const V& c) noexcept {
#if QUANTVIZ_BS_USE_FMA
        return _mm256_fnmadd_pd(b, c, a);
#else
        return _mm256_sub_pd(a, _mm256_mul_pd(b, c));
#endif
    }
    static V msub(const V& a, const V& b, const V& c, const V& d) noexcept {
#if QUANTVIZ_BS_USE_FMA
        return _mm256_fmsub_pd(a, b, _mm256_mul_pd(c, d));
#else
        return _mm256_sub_pd(_mm256_mul_pd(a, b), _mm256_mul_pd(c, d));
#endif
    }
};
#endif

// ---------------------------------------------------------------------------
// 共通カーネル — 全バックエンドが共有する唯一の算術。
// ---------------------------------------------------------------------------

/// ストライクに依存しないループ不変量。strip 版はこれを 1 回だけ計算する。
struct BsCtx {
    double t          = 0.0;   ///< max(T, 0)
    double disc       = 1.0;   ///< exp(-r max(T,0))
    double sqrt_t     = 0.0;   ///< sqrt(max(T,0))
    double vol_sqrt_t = 0.0;   ///< sigma sqrt(T)
    double drift      = 0.0;   ///< (r + sigma^2/2) T
    bool   degenerate = true;  ///< T <= 0 または sigma <= 0（NaN もここに落ちる）
};

inline BsCtx bs_ctx(double T, double r, double sigma) noexcept {
    const double t  = (T > 0.0) ? T : 0.0;
    const bool   ok = (T > 0.0) && (sigma > 0.0);  // NaN はどちらの比較も false になる

    BsCtx c;
    c.t          = t;
    c.disc       = std::exp(-r * t);
    c.sqrt_t     = std::sqrt(t);
    c.vol_sqrt_t = sigma * c.sqrt_t;
    // 最後の演算を mul にしないこと: `drift * t` は後続の `log(S/K) + drift` と融合されうる。
    c.drift      = ScalarOps::madd(0.5 * sigma * t, sigma, r * t);
    c.degenerate = !ok;
    return c;
}

/// 標準正規の累積分布 Phi = 0.5 erfc(-x / sqrt(2))。全バックエンドがこの 1 本を共有する。
template <class Ops>
inline typename Ops::V bs_cdf(const typename Ops::V& x) noexcept {
    return Ops::mul(Ops::splat(0.5), Ops::erfc(Ops::mul(Ops::neg(x), Ops::splat(kInvSqrt2))));
}

template <class Ops>
inline typename Ops::V bs_d1(const typename Ops::V& S, const typename Ops::V& K, const BsCtx& c) noexcept {
    return Ops::div(Ops::add(Ops::log(Ops::div(S, K)), Ops::splat(c.drift)), Ops::splat(c.vol_sqrt_t));
}

template <class Ops>
inline typename Ops::V bs_d2(const typename Ops::V& d1, double sigma, const BsCtx& c) noexcept {
    return Ops::nmsub(d1, Ops::splat(sigma), Ops::splat(c.sqrt_t));
}

template <class Ops>
inline typename Ops::V bs_price_from_d(const typename Ops::V& S, const typename Ops::V& K, const BsCtx& c,
                                       const typename Ops::V& d1, const typename Ops::V& d2,
                                       OptionType type) noexcept {
    const typename Ops::V kd = Ops::mul(K, Ops::splat(c.disc));
    if (type == OptionType::Call) return Ops::msub(S, bs_cdf<Ops>(d1), kd, bs_cdf<Ops>(d2));
    return Ops::msub(kd, bs_cdf<Ops>(Ops::neg(d2)), S, bs_cdf<Ops>(Ops::neg(d1)));
}

/// 1 ブロック分の価格。S > 0、全レーンの K > 0、非退化（T > 0 かつ sigma > 0）が前提。
template <class Ops>
inline typename Ops::V bs_price_block(const typename Ops::V& S, const typename Ops::V& K, const BsCtx& c,
                                      double sigma, OptionType type) noexcept {
    const typename Ops::V d1 = bs_d1<Ops>(S, K, c);
    const typename Ops::V d2 = bs_d2<Ops>(d1, sigma, c);
    return bs_price_from_d<Ops>(S, K, c, d1, d2, type);
}

/// -0.0 を +0.0 に正規化する（IEEE では -0.0 + 0.0 = +0.0）。ビューアの "-0.00" 表示を防ぐ。
inline double no_negative_zero(double x) noexcept { return x + 0.0; }

/// 退化ケースの価格: 決定的な割引ペイオフ。T = 0 なら本源的価値そのもの。
inline double bs_price_degenerate(double S, double K, double disc, OptionType type) noexcept {
    const double fwd = ScalarOps::madd(-K, disc, S);
    return no_negative_zero(std::max((type == OptionType::Call) ? fwd : -fwd, 0.0));
}

}  // namespace detail

/// 標準正規の密度関数 phi。
inline double norm_pdf(double x) noexcept { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }

/// 標準正規の累積分布 Phi。erfc ベース（裾でも相対精度が落ちない）。
inline double norm_cdf(double x) noexcept { return detail::bs_cdf<detail::ScalarOps>(x); }

inline double bs_price(double S, double K, double T, double r, double sigma, OptionType type) noexcept {
    const detail::BsCtx c = detail::bs_ctx(T, r, sigma);
    if (c.degenerate || !(S > 0.0) || !(K > 0.0)) return detail::bs_price_degenerate(S, K, c.disc, type);
    return detail::bs_price_block<detail::ScalarOps>(S, K, c, sigma, type);
}

inline BsGreeks bs_greeks(double S, double K, double T, double r, double sigma, OptionType type) noexcept {
    const detail::BsCtx c    = detail::bs_ctx(T, r, sigma);
    const bool          call = (type == OptionType::Call);
    BsGreeks            g;

    if (c.degenerate || !(S > 0.0) || !(K > 0.0)) {
        // sigma -> 0 / T -> 0 の極限値。Gamma と Vega は 0、その他は ITM かどうかで決まる。
        // S == K e^{-rT}（フォワードがちょうど ATM）は OTM 側に倒して Delta = 0 とする
        // （劣微分は [0,1] なので 0.5 の対称的な慣習もあるが、価格 0 と整合する側を採る）。
        const double fwd = detail::ScalarOps::madd(-K, c.disc, S);
        const double ind = (call ? (fwd > 0.0) : (fwd < 0.0)) ? 1.0 : 0.0;
        g.price          = detail::no_negative_zero(std::max(call ? fwd : -fwd, 0.0));
        g.delta          = detail::no_negative_zero(call ? ind : -ind);
        g.gamma          = 0.0;
        g.vega           = 0.0;
        g.theta          = detail::no_negative_zero((call ? -r : r) * K * c.disc * ind);
        g.rho            = detail::no_negative_zero((call ? c.t : -c.t) * K * c.disc * ind);
        return g;
    }

    const double d1  = detail::bs_d1<detail::ScalarOps>(S, K, c);
    const double d2  = detail::bs_d2<detail::ScalarOps>(d1, sigma, c);
    const double pdf = norm_pdf(d1);

    g.price = detail::bs_price_from_d<detail::ScalarOps>(S, K, c, d1, d2, type);
    g.gamma = pdf / (S * c.vol_sqrt_t);
    g.vega  = S * pdf * c.sqrt_t;

    const double time_decay = -S * pdf * sigma / (2.0 * c.sqrt_t);  // Theta の拡散項（Call/Put 共通）
    if (call) {
        g.delta = norm_cdf(d1);
        g.theta = time_decay - r * K * c.disc * norm_cdf(d2);
        g.rho   = c.t * K * c.disc * norm_cdf(d2);
    } else {
        g.delta = norm_cdf(d1) - 1.0;
        g.theta = time_decay + r * K * c.disc * norm_cdf(-d2);
        g.rho   = -c.t * K * c.disc * norm_cdf(-d2);
    }
    return g;
}

/// 参照実装: ストライクごとに bs_price を呼ぶ素直なループ（テストとベンチの比較対象）。
inline void bs_price_strip_scalar(double S, std::span<const double> strikes, double T, double r,
                                  double sigma, OptionType type, std::span<double> out) noexcept {
    const std::size_t n = std::min(strikes.size(), out.size());
    for (std::size_t i = 0; i < n; ++i) out[i] = bs_price(S, strikes[i], T, r, sigma, type);
}

namespace detail {

/// 先頭から SIMD で処理できた要素数を返す。K <= 0 / NaN を含むブロックに当たったらそこで止め、
/// 残りは呼び出し側のスカラ経路に任せる。前提は S > 0 かつ非退化。
#if QUANTVIZ_BS_SIMD_BACKEND == 2
inline std::size_t bs_strip_simd(double S, std::span<const double> strikes, const BsCtx& c, double sigma,
                                 OptionType type, std::span<double> out, std::size_t n) noexcept {
    using Ops                 = SimdOps;
    using V                   = Ops::V;
    constexpr std::size_t w   = V::size();
    const V               v_s = Ops::splat(S), v_zero = Ops::splat(0.0);

    std::size_t i = 0;
    for (; i + w <= n; i += w) {
        V k;
        k.copy_from(strikes.data() + i, stdx::element_aligned);
        if (stdx::any_of(!(k > v_zero))) break;
        bs_price_block<Ops>(v_s, k, c, sigma, type).copy_to(out.data() + i, stdx::element_aligned);
    }
    return i;
}
#elif QUANTVIZ_BS_SIMD_BACKEND == 1
inline std::size_t bs_strip_simd(double S, std::span<const double> strikes, const BsCtx& c, double sigma,
                                 OptionType type, std::span<double> out, std::size_t n) noexcept {
    using Ops                 = Avx2Ops;
    constexpr std::size_t w   = 4;
    const __m256d         v_s = Ops::splat(S), v_zero = _mm256_setzero_pd();

    std::size_t i = 0;
    for (; i + w <= n; i += w) {
        const __m256d k = _mm256_loadu_pd(strikes.data() + i);
        // _CMP_GT_OQ は NaN で false になるので、NaN も K <= 0 と同じ扱いになる。
        if (_mm256_movemask_pd(_mm256_cmp_pd(k, v_zero, _CMP_GT_OQ)) != 0xF) break;
        _mm256_storeu_pd(out.data() + i, bs_price_block<Ops>(v_s, k, c, sigma, type));
    }
    return i;
}
#else
inline std::size_t bs_strip_simd(double, std::span<const double>, const BsCtx&, double, OptionType,
                                 std::span<double>, std::size_t) noexcept {
    return 0;
}
#endif

}  // namespace detail

/// ストライク配列版。out[i] = bs_price(S, strikes[i], T, r, sigma, type)。
/// out.size() < strikes.size() の場合は out の長さだけ書く（範囲外には書かない）。
inline void bs_price_strip(double S, std::span<const double> strikes, double T, double r, double sigma,
                           OptionType type, std::span<double> out) noexcept {
    const std::size_t   n = std::min(strikes.size(), out.size());
    const detail::BsCtx c = detail::bs_ctx(T, r, sigma);

    // 退化ケース（T <= 0 / sigma <= 0 / S <= 0）はホットパスではないので丸ごとスカラで。
    std::size_t i =
        (c.degenerate || !(S > 0.0)) ? 0 : detail::bs_strip_simd(S, strikes, c, sigma, type, out, n);

    // 端数（および K <= 0 / NaN を含むブロック以降）はスカラ経路で処理する。
    for (; i < n; ++i) out[i] = bs_price(S, strikes[i], T, r, sigma, type);
}

}  // namespace quantviz::core
