# M3 — マイクロストラクチャ（板・Hawkes）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 板（指値板）とマッチングエンジン、Hawkes 過程による注文フロー生成と推定、板深度ラダー・直近約定・λ(t)・価格×時間ヒートマップを描く LOB シーン。「大口注入」ボタンで板が崩れて回復する様子を見る。

**Architecture:** 5 層と Snapshot/Command の双対を維持。板は core（外部依存ゼロ、ヒープはコンストラクタでのみ確保する固定容量プール + 価格ティック配列 + intrusive 双方向リスト）。Hawkes は core/models。LOB シーンの Snapshot は上位 N レベル・直近 K 約定・λ を縮約した ~1.5 KB の POD（通常のリング経由、面チャネルは使わない）。ヒートマップは vizcore の固定サイズ 2D 循環履歴（HEAT）に描画側で積む。時間は **秒**（dt = 1 ms、`steps_per_second = 1000` で実時間）。

**Tech Stack:** C++20 header-only core、Catch2 v3、ImPlot（`PlotBars`（水平）で深度ラダー、`PlotHeatmap` でヒートマップ）。

**規約:** `CLAUDE.md`。M1/M2 で確立した約束（`ClockControlState` + `clock_panel.hpp`、`std::in_place` 構築、`panel_common.hpp`、`RateMeter`、`prev_seq_` ガード、NaN は `!(v > lo)` 形で拒否、`ImGuiSliderFlags_AlwaysClamp` + ウィジェット後のクランプで配列添字を守る）。

---

## Task 0（先行・小）: Runner — 一時停止中の Reset / SetParam を即座に反映する（RUNNER-10）

M2 の FDM シーンで露呈した bridge の穴: `Runner::tick` はステップ後にしか publish しないため、一時停止中に送った `Reset` / `SetParam` は次の Step まで画面に現れない（各パネルは「waiting…」で誤魔化している）。`drain_commands()` がモデル系コマンドを 1 つ以上適用し、その tick でステップが 0 なら、Snapshot（と SurfaceModel なら面）を 1 回 publish する。R3（コマンドはステップ前）と R4（seq 単調増加。同じ seq の再 publish は許す — 描画側の `prev_seq_` ガードは `seq <= prev_seq_` で clear するので、Reset で seq が 0 に戻る場合も同じ seq の再送も正しく扱える）を維持。仕様行 RUNNER-10 を `03_tdd_spec.md` に追加し、`tests/test_runner.cpp` に追記。統合担当が M3 の最初に単独タスクとして実施し、FDM / GARCH / Kalman / Greeks パネルの「waiting…」文言を確認する。

## ファイル構成（所有権）

共有ファイル（`tests/CMakeLists.txt`, `viz/CMakeLists.txt`, `viz/src/main.cpp`, `.github/*`, `docs/*`, `README.md`）は統合担当だけが触る。テストファイルは足場として登録済み（空）。

| Task | 作成 | テスト |
|---|---|---|
| 1 板 + マッチング | `core/include/quantviz/core/micro/order_book.hpp`（データ構造 + プール + 不変条件）, `core/include/quantviz/core/micro/matching_engine.hpp`（submit/cancel の約定ロジック。`OrderBook` を包む） | `tests/test_order_book.cpp`（LOB-01..14）, `tests/bench_order_book.cpp`（BENCH-05） |
| 2 Hawkes | `core/include/quantviz/core/models/hawkes.hpp` | `tests/test_hawkes.cpp`（HAWKES-01..09） |
| 3 2D 履歴（vizcore） | `viz/include/quantviz/viz/history2d.hpp` | `tests/test_history2d.cpp`（HEAT-01..03） |
| 4 LOB シーン | `scenes/include/quantviz/scenes/lob_model.hpp`, `viz/src/panels/lob_panel.{hpp,cpp}` | `tests/test_lob_scene.cpp`（LOBSCENE-01..04） |
| 5 統合 | `viz/src/main.cpp`, `viz/CMakeLists.txt`, `docs/*`, `README.md`, CI（Debug で不変条件検査が有効なことの確認） | 全テスト・サニタイザ・viewer チェック・PR |

依存: 1, 2, 3 は独立。4 は 1+2+3。5 は全部。

---

## 型契約

### Task 1 — 板とマッチング（`namespace quantviz::core::micro`）

```cpp
using Price = std::int64_t;   // ティック整数（外部の double 価格は tick_size で変換）
using Qty   = std::uint64_t;
using OrderId = std::uint64_t;  // 0 は無効
enum class Side : std::uint8_t { Bid = 0, Ask = 1 };
enum class OrderType : std::uint8_t { Limit, Market };

struct Fill { OrderId maker; OrderId taker; Side taker_side; Price price; Qty qty; std::uint64_t seq; };
struct LevelView { Price price; Qty qty; std::uint32_t orders; };

/// 固定容量の板。ヒープはコンストラクタでのみ（プール MaxOrders、価格レベル MaxLevels）。
/// 価格は [base_price, base_price + MaxLevels) のティックのみ受け付ける（範囲外は拒否 → false）。
template <std::size_t MaxOrders, std::size_t MaxLevels>
class OrderBook {
public:
    explicit OrderBook(Price base_price);
    std::optional<Price> best_bid() const noexcept;  std::optional<Price> best_ask() const noexcept;
    Qty  level_qty(Side, Price) const noexcept;  std::uint32_t level_orders(Side, Price) const noexcept;
    std::size_t depth(Side, std::span<LevelView> out) const noexcept;   // best から out.size() レベルまで（空レベルは飛ばす）
    Qty  total_qty(Side) const noexcept;  std::size_t order_count() const noexcept;  std::size_t pool_free() const noexcept;
    /// 検査（Debug では 1 ステップごと、Release ではテストから）: bid<ask、各レベルの qty/count とリストの整合、プールの整合
    bool check_invariants() const noexcept;
    // matching_engine が使う低レベル操作（レベル末尾に追加 / 先頭から削る / 取消）
    ...
};

/// 価格時間優先のマッチング。fills は呼び手のバッファに書く（アロケーションなし）。
template <std::size_t MaxOrders, std::size_t MaxLevels>
class MatchingEngine {
public:
    explicit MatchingEngine(Price base_price);
    /// 指値: 反対側と交差する分を約定（複数レベルを掃く）、残りを板に載せる。返り値は付与した id（拒否なら 0）。
    OrderId submit_limit(Side, Price, Qty, std::span<Fill> fills, std::size_t& n_fills) noexcept;
    /// 成行: 反対側を数量分だけ掃く。板が尽きたら残りは捨てる（返り値 = 約定数量）。
    Qty     submit_market(Side, Qty, std::span<Fill> fills, std::size_t& n_fills) noexcept;
    bool    cancel(OrderId) noexcept;      // 存在しなければ false
    const OrderBook<MaxOrders, MaxLevels>& book() const noexcept;
    std::uint64_t next_seq() const noexcept;   // 約定通番
    // 統計（LOB-08 の数量保存に使う）: 投入数量、取消数量、約定数量（買い/売り）
};
```
規則: 自己約定は考えない（単一プレイヤーではない合成フロー）が、**板に bid ≥ ask が存在しない**ことは常に保つ（LOB-11）。同価格は先着順（LOB-05）。部分約定は残数量を板に残す（LOB-06）。取消で注文が消え、最後ならレベルも空になる（LOB-07）。数量保存（LOB-08）。プール枯渇は拒否（LOB-14、`submit_limit` が 0 を返す）。seed 固定の合成フローで約定列が bit 一致（LOB-12）。

BENCH-05（`bench_order_book.cpp`, `[!benchmark][lob]`）: 1e6 注文（指値 70 % / 成行 10 % / 取消 20 %、mid ± 20 ティック、`core::Rng` seed 固定）を処理する時間 → ≥ 1e6 注文/秒。

### Task 2 — Hawkes（`core/models/hawkes.hpp`, `namespace quantviz::core`）

```cpp
struct HawkesParams { double mu; double alpha; double beta; };   // λ(t) = μ + Σ_{t_i<t} α e^{−β (t−t_i)}
bool   hawkes_stable(HawkesParams p) noexcept;                   // μ>0, α≥0, β>0, α/β < 1
double hawkes_intensity(HawkesParams p, std::span<const double> times, double t) noexcept;   // 直接和（テストの参照）
/// 逐次強度: 事象を 1 つ追加するたび O(1) で更新（A_{i+1} = e^{−β Δt}(A_i + α)）
class HawkesIntensity { public: explicit HawkesIntensity(HawkesParams); void add_event(double t) noexcept; double at(double t) const noexcept; void reset() noexcept; };
/// Ogata の thinning で [0, T) の事象列を生成。out に書き、件数を返す（容量超過なら打ち切って容量を返す）。
std::size_t hawkes_simulate(HawkesParams p, double T, core::Rng& rng, std::span<double> out) noexcept;
double hawkes_log_likelihood(HawkesParams p, std::span<const double> times, double T) noexcept;   // O(n) 再帰、補償子は閉形式
double hawkes_compensator(HawkesParams p, std::span<const double> times, double T) noexcept;      // ∫_0^T λ
struct HawkesFit { HawkesParams params; double log_lik; OptimResult<3> optim; };
HawkesFit hawkes_fit(std::span<const double> times, double T, HawkesParams init, OptimOptions = {});   // 無制約変換（softplus, 分岐比は sigmoid）
/// 時間再スケール残差 τ_i = Λ(t_i) − Λ(t_{i−1})（単位指数のはず）
std::size_t hawkes_rescaled_residuals(HawkesParams p, std::span<const double> times, std::span<double> out) noexcept;
```

### Task 3 — 2D 履歴（`viz/include/quantviz/viz/history2d.hpp`, vizcore）

```cpp
/// 固定寸法（Rows 価格ビン × Cols 時間列）の循環バッファ。列方向に循環し、ImPlot の PlotHeatmap に
/// そのまま渡せる「最古→最新」順の連続配列を `ordered()` で返す（1 回のコピー）。
template <std::size_t Rows, std::size_t Cols>
class History2D {
public:
    void push_column(std::span<const float> column) noexcept;   // Rows 要素。列を 1 つ進める
    std::size_t count() const noexcept;                          // 埋まった列数（≤ Cols）
    std::span<const float> ordered() const noexcept;             // row-major [row][col]、col は最古→最新（count 列分）
    void clear() noexcept;                                       // 全ゼロ・count 0
};
```

### Task 4 — LOB シーン

```cpp
struct LobSnapshot {
    static constexpr std::size_t kLevels = 16, kTrades = 32;
    double t;                                     // 秒
    double tick_size;  Price mid_ticks;  double mid;  double spread;
    std::array<LevelView, kLevels> bids, asks;    // best から。空は qty 0
    std::uint32_t n_bids, n_asks;
    std::array<Fill, kTrades> trades;  std::uint32_t n_trades;   // 直近（新しい順）
    double lambda_buy, lambda_sell;               // Hawkes 強度（買い・売り側それぞれ）
    std::uint64_t orders_submitted, orders_cancelled, fills;  std::uint32_t pool_free;
    std::uint64_t seq;
};   // ≈ 1.6 KB
class LobModel {   // dt は秒。1 step = dt 内の全事象を処理
    enum Param : uint32_t { kMu = 1, kAlpha = 2, kBeta = 3, kMarketFrac = 4, kCancelFrac = 5, kInjectBuy = 6, kInjectSell = 7 };
    // 二つの独立な Hawkes（買い・売り）で到着時刻を生成。各到着で: 成行（kMarketFrac）/ 取消（kCancelFrac、ランダムな既存注文）/ 指値（mid ± 幾何分布の距離、数量は 1〜100 の対数一様）。
    // kInjectBuy/Sell: value = 数量。次ステップの先頭で成行として投入（「大口注入」）。
    // Reset: 板を空にし、λ を μ に戻し、初期板（mid の周りに両側 10 レベル）を敷く。
};
```
パネル: 「Depth ladder」（`PlotBars` 水平: bid を負、ask を正、価格を y 軸）、「Price × time heatmap」（`History2D<64 ビン, 256 列>`、mid を中心とした ±32 ティックの数量を各列に積む。mid が動いたらビンを追従させる規則を文書化）、「Trades / intensity」（直近約定の散布、λ_buy / λ_sell の History）、「Control」（μ α β、成行/取消比率、「Inject buy 500」「Inject sell 500」ボタン、時計、テレメトリ: 板の注文数、プール残、約定数、dropped）。

---

## 各タスクの手順（共通）

1. `docs/03_tdd_spec.md` の該当 ID を確認。統計テストは seed 固定 + 4 SE + 根拠。
2. Red → Green → Refactor。`-Werror`（Release / Debug）。Debug では `check_invariants()` を毎ステップ呼ぶ（`assert`）。
3. 報告: 状態行、ファイル、ctest 要約行、RED の証拠、計測値（BENCH-05、Hawkes MLE の復元幅）、逸脱と根拠。

## Task 5: 統合（DoD）

- [ ] `main.cpp` に "Order book" を登録、`viz/CMakeLists.txt` に `lob_panel.cpp`。
- [ ] 全テスト（Release/Debug/FMA）、ASan/UBSan、TSan Green。BENCH-05 実測を仕様書へ。
- [ ] viewer: 大口注入で片側が掃かれ、指値の流入で回復する。1000 steps/s で dropped 0。60 fps。
- [ ] docs 更新（状態列、設計書 §6 の 🔜M3 → ✅、README）、PR → CI → squash merge。
