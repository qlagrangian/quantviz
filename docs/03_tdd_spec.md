# quantviz TDD 仕様網羅

| 項目 | 内容 |
|---|---|
| 版 | 1.0（M0 実装済 59 ケース + BENCH 2 本 / M1〜M5 は仕様のみ） |
| 原則 | **仕様 = テスト ID**。コードより先に本書へ ID を追加し、テスト名は `"<ID>: <仕様文>"` とする |
| 関連文書 | `01_design.md`（設計）, `02_implementation_plan.md`（タスクは本書の ID を参照） |

状態欄: ✅ 実装済・Green ／ ⬜ 未実装（Red 状態の仕様） ／ 🔧 実装中

---

## 1. テスト戦略

### 1.1 レイヤ別の狙い

| 層 | 何を検証するか | 主な種別 |
|---|---|---|
| core | 数式が正しいか（解析解・閉形式・性質）、決定性、数値安定性 | 単体 U / 数値 N / 性質 P / 統計 S |
| bridge | 並行契約（無ロス・無重複・順序）、状態機械（時計・コマンド）、ノンブロッキング | 単体 U / 並行 C |
| scenes | Model 契約の充足、Snapshot が POD、コアと同一結果（dual-run） | 契約 K / 決定性 D |
| vizcore | 描画補助の純ロジック（History, 2D history, 行列, カメラ） | 単体 U |
| viz（GUI） | 起動・操作・fps — **自動テストしない**。手動チェックリスト（`02_implementation_plan.md` §5.3） | 手動 |

### 1.2 種別タグ

| タグ | 種別 | 規則 |
|---|---|---|
| `[unit]` | U 単体 | 決定的、1 ms 未満 |
| `[numeric]` | N 数値 | 解析解・閉形式・別実装との一致。許容誤差に根拠を書く |
| `[property]` | P 性質 | 不等式・単調性・不変条件。ランダム入力は seed 固定 |
| `[statistical]` | S 統計 | seed 固定、許容 = 4 SE。N を大きくして誤差を下げる（許容を緩めない） |
| `[concurrency]` | C 並行 | スレッド起動。TSan で回す。タイムアウト付き |
| `[contract]` | K 契約 | `STATIC_REQUIRE` による concept / POD / サイズ検査 |
| `[determinism]` | D 決定性 | 同 seed → bit 一致 |
| `[!benchmark]` | B ベンチ | 既定で実行されない。退行検知（2 倍以上で要調査） |

### 1.3 TDD の運用

* Red → Green → Refactor。1 ID = 1 `TEST_CASE`（複数の観点は `SECTION`）
* 契約テスト（K）は新シーンごとに必ず 1 本（`STREAM-01` をテンプレートにする）
* 統計テスト（S）は必ず「期待値・SE・何倍」をテスト内コメントに残す
* 並行テスト（C）は `tick()` のような同期 API で先に振る舞いを固定し、スレッド版は 1〜2 本に絞る

---

## 2. 規約

### 2.1 ID 体系

`<MODULE>-<2 桁連番>`。MODULE はテストファイルと 1:1（`tests/test_<module>.cpp`）。連番は欠番を作らず、廃止は「(廃止)」と注記して残す。

### 2.2 命名

```cpp
TEST_CASE("RING-07: producer/consumer threads transfer 1M items with no loss, dup or reorder",
          "[ring][concurrency]") { ... }
```

第 2 引数のタグは `[module]` + 種別。ベンチは `[!benchmark][module]`。

### 2.3 seed

統計・決定性テストの seed はテスト内で明示（`Gbm g(params, 31415)`）。同じ seed を複数テストで使ってよい。

### 2.4 許容誤差の指針

| 比較対象 | 許容 | 例 |
|---|---|---|
| 厳密解（指数・閉形式） | 相対 1e-12 | GBM-04, EWMA-03 |
| 別実装との一致（二パス・密行列） | 相対 1e-10 | WEL-03, TRIDIAG-01 |
| 数値微分（中心差分） | 相対 1e-6（h = 1e-4 × スケール） | BS-07 |
| 有名な数表値 | 絶対 1e-4 | BS-03 |
| FDM vs 解析解 | 相対 1e-3（N=M=200）+ 収束次数 | FDM-02/03 |
| モンテカルロ・推定量 | 4 SE（seed 固定） | GBM-07, STREAM-07 |
| MLE パラメータ復元 | 推定 SE の 4 倍 or 絶対幅（根拠を注記） | GARCH-06, HAWKES-08 |

---

## 3. M0 仕様（実装済）

### 3.1 RING — `bridge/spsc_ring.hpp` → `tests/test_spsc_ring.cpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| RING-01 | 新規リングは空。`try_pop` は false を返し出力引数を変更しない | U | ✅ |
| RING-02 | push した順に pop される（FIFO）。`size_approx` は要素数 | U | ✅ |
| RING-03 | ちょうど Capacity 個入る。Capacity+1 個目の push は**ブロックせず** false。1 個 pop すれば再び入る | U | ✅ |
| RING-04 | インデックスが何周しても順序が保たれる（Capacity 4 で 3000 要素） | U | ✅ |
| RING-05 | `capacity()` はコンパイル時定数。Command は trivially copyable。非 2 冪・非 POD は `static_assert` でコンパイルエラー | K | ✅ |
| RING-06 | POD 構造体（Command）が bit 一致で往復する | U | ✅ |
| RING-07 | 生産者スレッド 1 本 × 消費者 1 本で 1,000,000 要素を転送し、欠落・重複・順序逆転が無い | C | ✅ |
| RING-08 | head / tail / 各キャッシュ / バッファが別キャッシュラインに置かれる（`sizeof ≥ 4×64`, `alignof == 64`） | K | ✅ |

### 3.2 CLOCK — `bridge/sim_clock.hpp` → `tests/test_sim_clock.cpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| CLOCK-01 | 壁 1 秒 × 1000 steps/s → ちょうど 1000 ステップ | U | ✅ |
| CLOCK-02 | 端数は捨てずに蓄積する（0.5 step + 0.5 step = 1 step） | U | ✅ |
| CLOCK-03 | 一時停止中はステップ 0 かつ壁時間を蓄積しない → 再開直後にバーストしない | U | ✅ |
| CLOCK-04 | speed は倍率（2.0 → 2 倍、0.5 → 半分）。speed 0 は停止 | U | ✅ |
| CLOCK-05 | `request_step` は一時停止中でも**ちょうど 1** ステップ。複数リクエストは加算 | U | ✅ |
| CLOCK-06 | 走行中の `request_step` は壁時計分に +1 される | U | ✅ |
| CLOCK-07 | `max_steps_per_tick` で 1 回の返値を上限に抑え、**超過分は捨てる**（追いつかない） | U | ✅ |
| CLOCK-08 | 負・NaN の elapsed は 0 扱い。負・NaN の speed は 0 にクランプ | U | ✅ |
| CLOCK-09 | `start_paused` で一時停止状態から開始する | U | ✅ |

### 3.3 RUNNER — `bridge/runner.hpp` → `tests/test_runner.cpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| RUNNER-01 | `tick(elapsed)` は期限ステップ数だけ `step(dt)` を呼び、1 ステップごとに Snapshot を 1 枚発行（seq 1..n 順） | U | ✅ |
| RUNNER-02 | 時計コマンドはステップ**前**に適用（Pause → 0 ステップ、StepOnce → 1、Resume → 再開） | U | ✅ |
| RUNNER-03 | SetSpeed は時計へ、SetParam / Reset は `Model::apply` へ届く。Reset は同一 tick 内のステップより先 | U | ✅ |
| RUNNER-04 | `publish_every = k` で seq が k の倍数の Snapshot だけ出る | U | ✅ |
| RUNNER-05 | Snapshot リング満杯でもコアは止まらず、`dropped_snapshots` に加算。古い Snapshot が残る | U | ✅ |
| RUNNER-06 | Command リング満杯なら `send` は false（ブロックしない） | U | ✅ |
| RUNNER-07 | スレッド実行で Snapshot の seq が単調増加。`stop()` 前に `send` が true を返した Command は `stop()` 後に必ず適用済み。`running()` が正しい | C | ✅ |
| RUNNER-08 | `start()` は冪等。デストラクタは走行中スレッドを join する（ハングしない） | C | ✅ |
| RUNNER-09 | `SurfaceModel` を満たす Model の Runner は `surface_every` ステップごとに面を `TripleBuffer` へ publish し、`poll_surface` は最新 1 枚だけを返す（古い面は捨てる）。非 SurfaceModel の Runner にはチャネルが生えない | U | ✅ |
| RUNNER-10 | 一時停止中（その tick のステップ数が 0）に `Reset` / `SetParam` を適用したら、Snapshot（SurfaceModel なら面も）を 1 回 publish する。seq は再送でも減らない（Reset は 0 に戻す）。ステップがあった tick では追加の publish はしない | U | ✅ |
| RUNNER-11 | Runner は各 step の所要時間を対数ビンのヒストグラム（atomic, relaxed）に記録し、描画側が読める。`measure_every` で k ステップに 1 サンプル。非計測設定では挙動不変 | U | ⬜ |
| RUNNER-12 | 一時停止中の `StepOnce` で実行したステップは `publish_every` の位相に関わらず publish する（SurfaceModel なら面も）。走行中の間引きは変えない | U | ✅ |

### 3.4 GBM — `core/models/gbm.hpp` → `tests/test_gbm.cpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| GBM-01 | 初期状態は (s0, t=0)、与えたパラメータを保持 | U | ✅ |
| GBM-02 | 同 seed → 1000 ステップのパスが bit 一致 | D | ✅ |
| GBM-03 | 異なる seed → パスが異なる | D | ✅ |
| GBM-04 | σ = 0 なら S_t = s0·exp(μt)（相対 1e-12） | N | ✅ |
| GBM-05 | σ = 3.0、20000 ステップでも S > 0 かつ有限 | P | ✅ |
| GBM-06 | `step()` の返値 = log(S_new / S_old)（絶対 1e-12） | U | ✅ |
| GBM-07 | 対数リターンの標本平均 ≈ (μ − σ²/2)dt、標本分散 ≈ σ²dt（N=200000, 4 SE） | S | ✅ |
| GBM-08 | `reset(seed)` でパスが再現される。`set_mu` した値は reset を跨いで保持される | U | ✅ |
| GBM-09 | `set_sigma(負)` は 0 にクランプ | U | ✅ |
| GBM-10 | 時刻は n·dt を蓄積（絶対 1e-9） | U | ✅ |

### 3.5 WEL — `core/stats/welford.hpp` → `tests/test_welford.cpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| WEL-01 | 空の状態: count 0、mean 0、variance 0、population_variance 0 | U | ✅ |
| WEL-02 | 1 値: mean = x、variance 0 | U | ✅ |
| WEL-03 | 5000 個の正規乱数で二パス法の平均・標本分散・母分散・標準偏差と一致（相対 1e-10） | N | ✅ |
| WEL-04 | オフセット 1e9 の値でも分散が正しい（E[x²]−E[x]² の破綻を回避） | N | ✅ |
| WEL-05 | `reset` で空に戻る | U | ✅ |
| WEL-06 | 教科書例 (2,4,4,4,5,5,7,9): mean 5、母分散 4、標本分散 32/7 | U | ✅ |

### 3.6 EWMA — `core/stats/ewma.hpp` → `tests/test_ewma.cpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| EWMA-01 | 初回観測で var = r² に初期化。それ以前は 0 | U | ✅ |
| EWMA-02 | 定数入力 c で var = c² を維持 | P | ✅ |
| EWMA-03 | 再帰式 = 閉形式 λ^{n−1}r₁² + (1−λ)Σλ^{n−i}rᵢ²（相対 1e-10） | N | ✅ |
| EWMA-04 | ショック後ゼロ入力で var は λ 倍ずつ幾何減衰 | N | ✅ |
| EWMA-05 | λ が小さいほどレジーム変化に速く追従 | P | ✅ |
| EWMA-06 | `set_lambda` は (0,1) にクランプ、NaN は既定 0.94 | U | ✅ |
| EWMA-07 | `reset` で空、次の push で再初期化 | U | ✅ |

### 3.7 STREAM — `scenes/streaming_model.hpp` → `tests/test_streaming_model.cpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| STREAM-01 | `bridge::Model<StreamingModel>` を満たす。Snapshot は trivially copyable かつ ≤ 128 B | K | ✅ |
| STREAM-02 | 新規: seq 0、spot = s0、t = 0、真値 μ/σ、分散 0 | U | ✅ |
| STREAM-03 | 同 seed の素の `Gbm` と spot・log_return・seq が毎ステップ一致（dual-run）。Welford / EWMA の count = n | D | ✅ |
| STREAM-04 | SetParam(μ/σ/λ) は次ステップから反映され seq は進まない。未知 id は無視 | U | ✅ |
| STREAM-05 | Reset でパスと統計は初期化、パラメータは保持 | U | ✅ |
| STREAM-06 | `Reset(seed)` で別インスタンスと同一パスになる | D | ✅ |
| STREAM-07 | 年率換算の Welford ボラ ≈ σ（40000 本, 2 %）、EWMA(λ=0.999) ボラ ≈ σ（10 %） | S | ✅ |

### 3.8 HIST — `viz/history.hpp` → `tests/test_history.cpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| HIST-01 | 新規は空、offset 0、latest は 0 | U | ✅ |
| HIST-02 | 満杯前は offset 0、挿入順に格納、latest / oldest が正しい | U | ✅ |
| HIST-03 | 満杯後は offset が**最古**の index（ImPlot 循環バッファ規約） | U | ✅ |
| HIST-04 | `clear` で空。再アロケーションなし。次の push は index 0 | U | ✅ |

### 3.9 BENCH（M0）

| ID | 内容 | 目標 | 実測（GCC 13 -O3） | 状態 |
|---|---|---|---|---|
| BENCH-01 | `SpscRing<StreamingSnapshot,1024>` push+pop | < 20 ns | 4.4 ns | ✅ |
| BENCH-02 | `StreamingModel::step + snapshot` | < 100 ns | 39 ns | ✅ |

**M0 合計: 59 テストケース（B を除く）+ BENCH 2 本 — 全 Green。**

---

## 4. M1 仕様 — 2D 理論パネル

### 4.1 VIZ — `viz/scene_registry.*`, `panels/common_controls.*`（vizcore 部分のみ）

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| VIZ-01 | レジストリは登録された全シーン名を登録順で列挙する | U | ✅ |
| VIZ-02 | シーン選択で Runner と Panel の対が生成され、前のシーンの Runner は `stop()` される（`running()` false） | U | ✅ |
| VIZ-03 | 共通 Control の状態（speed / paused）は Command 生成関数の純関数として検査できる（ImGui 非依存部） | U | ✅ |
| VIZ-04 | 実 Runner を持つ `RunnerScene` を `select` で切り替えると、前シーンの計算スレッドが join され `running()` が false になる | C | ✅ |
| VIZ-05 | `RateMeter`（受信レートの 0.25 s 窓 + EMA）は定常入力で真値に収束し、最初の窓が閉じるまでは 0、`reset` で 0 に戻る | U | ✅ |

### 4.2 BS — `core/pricing/black_scholes.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| BS-01 | 無裁定境界: max(S−Ke^{−rT},0) ≤ C ≤ S、max(Ke^{−rT}−S,0) ≤ P ≤ Ke^{−rT} | P | ✅ |
| BS-02 | プット・コール・パリティ C − P = S − Ke^{−rT}（相対 1e-12） | N | ✅ |
| BS-03 | 数表値: S=K=100, T=1, r=0.05, σ=0.2 → C=10.4506, P=5.5735（絶対 1e-4） | N | ✅ |
| BS-04 | σ→0 極限で C → max(S−Ke^{−rT},0)。退化ブランチ（T≤0 / σ≤0 / K≤0 / S≤0 / σ=NaN）も有限値を返し、0 は +0.0 で返す。S = Ke^{−rT} ちょうどでは OTM 側（Δ=0）に倒す | N | ✅ |
| BS-05 | T→0 極限で C → max(S−K,0)、Γ は ATM で発散方向（有限 T で単調増大） | N | ✅ |
| BS-06 | Δ_call ∈ (0,1)、Δ_put = Δ_call − 1、Γ と ν は call/put で同一かつ > 0 | P | ✅ |
| BS-07 | 解析 Greeks（Δ Γ ν Θ ρ）が中心差分と一致（相対 1e-6） | N | ✅ |
| BS-08 | 単調性: C は S・σ・T で増加、K で減少 | P | ✅ |
| BS-09 | 一次同次性: C(λS, λK) = λ·C(S, K) | P | ✅ |
| BS-10 | `price_strip(span<K>)` の各要素がスカラ版と一致（相対 1e-15） | N | ✅ |
| BS-11 | ストリップ長が SIMD 幅の倍数でない場合（N=1, 7, 65）も端が正しい | U | ✅ |
| BS-12 | (= BENCH-03) N=1024 ストリップがスカラループより高速（目標 ≥ 1.2×、§9 の判断記録を参照。当初の ≥ 2× は厳密一致 BS-10 と両立しないため改定） | B | ✅ |

### 4.3 GREEKS — `scenes/greeks_model.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| GREEKS-01 | Model 契約充足、Snapshot は POD（固定 N ストライク配列 + 固定格子） | K | ✅ |
| GREEKS-02 | ストライク配列は昇順・等間隔、Snapshot 全要素が有限 | P | ✅ |
| GREEKS-03 | SetParam(σ / r / T) は apply の場で Snapshot に反映され（配列と真値フィールドは同じパラメータで再計算、seq は進まない）、以後のステップでも維持される。未知 id は無視（M3 の R10 に合わせて「次ステップから」を改定） | U | ✅ |
| GREEKS-04 | Γ の最大ストライクは S に最も近いストライク（±1 グリッド） | P | ✅ |
| GREEKS-05 | 同 seed の素の Gbm と S が一致（dual-run） | D | ✅ |

### 4.4 OPTIM — `core/stats/optim.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| OPTIM-01 | Nelder–Mead が 2 次関数の最小点を 1e-8 で求める | N | ✅ |
| OPTIM-02 | Nelder–Mead が Rosenbrock で (1,1) に 1e-4 で到達 | N | ✅ |
| OPTIM-03 | 返す反復履歴の best f は単調非増加 | P | ✅ |
| OPTIM-04 | BFGS（数値勾配）が 2 次関数で 20 反復以内に 1e-8 収束 | N | ✅ |
| OPTIM-05 | `max_iter` を超えない。到達時は `converged = false` | U | ✅ |
| OPTIM-06 | 制約変換（logit / softplus）の往復が恒等（相対 1e-12） | N | ✅ |

### 4.5 GARCH — `core/stats/garch.hpp`, `scenes/garch_model.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| GARCH-01 | σ²_t = ω + α r²_{t−1} + β σ²_{t−1} の 3 ステップ手計算と一致 | N | ✅ |
| GARCH-02 | 無条件分散 ω/(1−α−β) にフィルタが収束（定数入力） | N | ✅ |
| GARCH-03 | 対数尤度 −½Σ(log 2π + log σ²_t + r²_t/σ²_t) が直接計算と一致 | N | ✅ |
| GARCH-04 | α+β ≥ 1 は定常制約違反として拒否（ペナルティ or 変換で到達不能） | P | ✅ |
| GARCH-05 | パラメータ変換（無制約 ↔ 制約）の往復が恒等 | N | ✅ |
| GARCH-06 | 合成 GARCH（ω=1e-6, α=0.08, β=0.90, N=20000）から MLE が α̂, β̂ を ±0.03 で復元（seed 固定・根拠: 漸近 SE ≈ 0.007） | S | ✅ |
| GARCH-07 | 大標本で真値の尤度 ≥ 摂動値の尤度 | P | ✅ |
| GARCH-08 | 半減期 ln(0.5)/ln(α+β) が α+β↑ で単調増大 | P | ✅ |
| GARCH-09 | Model 契約充足、Snapshot（σ_t, 真値, 尤度格子 固定 G×G, 軌跡 最新 K 点）は POD | K | ✅ |
| GARCH-10 | 尤度格子の全値が有限、最大値の格子点が MLE 推定値の隣接格子内 | P | ✅ |
| GARCH-11 | 最適化軌跡の終点 = 推定値 | U | ✅ |

### 4.6 KALMAN — `core/math/mat.hpp`, `core/stats/kalman.hpp`, `scenes/kalman_pair_model.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| KALMAN-01 | `Mat<N,M>` の積・転置・2×2/3×3 逆行列が手計算と一致。特異・非有限な行列で `inverse` は `nullopt`、`is_symmetric` / `is_psd` は負例（`diag(0,−1)` 等）と NaN を拒否する | N | ✅ |
| KALMAN-02 | Q=0・スカラ状態のカルマン推定 = 累積平均（RLS と一致） | N | ✅ |
| KALMAN-03 | predict で共分散が F P Fᵀ + Q に増える。`reset` と縮退した観測（NaN / 特異 S）は状態を壊さずスキップ数を数える | U | ✅ |
| KALMAN-04 | update で共分散が減少（P_post ≼ P_prior） | P | ✅ |
| KALMAN-05 | 1000 ステップ後も共分散が対称・半正定値 | P | ✅ |
| KALMAN-06 | イノベーションの 1 次自己相関が 4 SE 以内で 0（ホワイト） | S | ✅ |
| KALMAN-07 | ランダムウォーク β を追跡し、±3σ 帯に 95 % 以上の時点で入る | S | ✅ |
| KALMAN-08 | Q=0 で定常回帰の β が OLS 推定値に収束（相対 1e-6） | N | ✅ |
| KALMAN-09 | Model 契約充足、Snapshot POD | K | ✅ |
| KALMAN-10 | 共和分ペア生成: スプレッドが定常（分散が N で発散しない） | S | ✅ |
| KALMAN-11 | SetParam(観測ノイズ) は次ステップから反映 | U | ✅ |

---

## 5. M2 仕様 — 3D サーフェス・FDM

### 5.1 GL — `viz/gl/camera.hpp`, `viz/gl/math.hpp`（vizcore, GUI 非依存）

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| GL-01 | look-at / perspective 行列が参照値と一致（相対 1e-6） | N | ✅ |
| GL-02 | 軌道カメラは回転で目標点との距離を保つ | P | ✅ |
| GL-03 | unproject(project(p)) = p（絶対 1e-5） | N | ✅ |
| GL-04 | アスペクト比変更で射影行列の [0][0] が 1/aspect に比例 | U | ✅ |

### 5.2 SURF — `viz/gl/surface_mesh.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| SURF-01 | N×M グリッドで頂点 N·M、三角形 2(N−1)(M−1) | U | ✅ |
| SURF-02 | 全インデックスが頂点数未満 | P | ✅ |
| SURF-03 | 全法線が単位長（1e-6） | P | ✅ |
| SURF-04 | 平面 z=const の法線は全て (0,0,1) | N | ✅ |
| SURF-05 | z 更新でバッファのポインタ・サイズが変わらない（再アロケーションなし） | U | ✅ |
| SURF-06 | z=f(x,y) の既知関数で法線が解析勾配と一致（1e-3） | N | ✅ |

### 5.3 TRIPLE — `bridge/triple_buffer.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| TRIPLE-01 | 読み手は常に最後に publish された値を見る | U | ✅ |
| TRIPLE-02 | 書き手は決してブロックしない（読み手が遅くても） | C | ✅ |
| TRIPLE-03 | 読み手が裂けた値（torn read）を観測しない（各スロットにチェックサム、1e6 回） | C | ✅ |
| TRIPLE-04 | 新データが無ければ `read` は false | U | ✅ |
| TRIPLE-05 | T は trivially copyable（`static_assert`） | K | ✅ |

### 5.4 VOLSURF — `core/pricing/vol_surface.hpp`, `scenes/vol_surface_model.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| VOLSURF-01 | 全 (K,T) で IV > 0 | P | ✅ |
| VOLSURF-02 | 既定パラメータでカレンダー裁定なし（総分散 σ²T が T で単調増大）。`ssvi_calendar_arbitrage_free` は Gatheral–Jacquier Thm 4.1 の条件と η(1+\|ρ\|) ≤ 2 の翼バンドの連言（後者は十分条件） | P | ✅ |
| VOLSURF-03 | スキュー 0 でスマイルが ATM 対称 | N | ✅ |
| VOLSURF-04 | 面（`Surface`, 64×32, 8.6 KB）は固定サイズ・POD で `surface()` が全フィールドを書く。SetParam で次の `surface()` から形が変わり、apply は seq を動かさない。Snapshot は 80 B。Runner の `poll_surface` 往復 | K | ✅ |

### 5.5 TRIDIAG — `core/math/tridiag.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| TRIDIAG-01 | n=5 で密行列解と一致（相対 1e-12） | N | ✅ |
| TRIDIAG-02 | 単位行列で x = d | U | ✅ |
| TRIDIAG-03 | n=1 が動く | U | ✅ |
| TRIDIAG-04 | 対角優位ランダム n=1000 の残差 ‖Ax−d‖ ≤ 1e-10 | N | ✅ |

### 5.6 FDM — `core/pricing/fdm_cn.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| FDM-01 | 格子 (S_max, N, M) と境界条件（S=0, S=S_max）が設定通り | U | ✅ |
| FDM-02 | CN European call が BS と一致（N=M=200、相対 1e-3。K が格子点上なら 5e-5、格子点間なら 9e-4 — 線形補間の h²Γ/8） | N | ✅ |
| FDM-03 | N を 2 倍にすると誤差が約 1/4（収束次数 ≈ 2。実測比 4.14 / 4.03。2 次はキンクのセル平均化で得る。Rannacher 起動は小さい M での Γ の振動抑制を担い、FDM-05 の節で守る） | N | ✅ |
| FDM-04 | 格子上でプット・コール・パリティ（1e-3） | N | ✅ |
| FDM-05 | 格子から求めた Δ が BS Δ と一致（1e-2） | N | ✅ |
| FDM-06 | American put ≥ European put（全格子点） | P | ✅ |
| FDM-07 | American ≥ 本源的価値（全格子点） | P | ✅ |
| FDM-08 | q=0 の American call = European call（早期行使なし） | N | ✅ |
| FDM-09 | プットの行使境界 S*(t) は満期に向かって単調非減少 | P | ✅ |
| FDM-10 | PSOR が ω∈(1,2) で最大反復内に収束 | U | ✅ |
| FDM-11 | `step_backward` を M 回で t が T→0、以後は no-op | U | ✅ |

### 5.7 FDMSCENE — `scenes/fdm_american_model.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| FDMSCENE-01 | Model 契約充足、Snapshot（V(S) 固定 N, 行使境界, 残反復数, status）は POD | K | ✅ |
| FDMSCENE-02 | 1 `step` = 1 後ろ向き反復（残反復数が 1 減る） | U | ✅ |
| FDMSCENE-03 | M ステップ後の S0 における値 = `fdm_cn` 単体の価格 | D | ✅ |
| FDMSCENE-04 | Reset で満期ペイオフに戻る | U | ✅ |

---

## 6. M3 仕様 — マイクロストラクチャ

### 6.1 LOB — `core/micro/order_book.hpp`, `core/micro/matching_engine.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| LOB-01 | 空の板は best bid / ask を持たない | U | ✅ |
| LOB-02 | 指値買いを入れると best bid になる | U | ✅ |
| LOB-03 | bid は価格降順、ask は昇順で列挙される | P | ✅ |
| LOB-04 | ランダム操作 1e5 回の後も best bid < best ask | P | ✅ |
| LOB-05 | 同価格では先に入った注文が先に約定（価格時間優先） | U | ✅ |
| LOB-06 | 部分約定で残数量が板に残る | U | ✅ |
| LOB-07 | 取消で注文が消え、最後の注文ならレベルも消える | U | ✅ |
| LOB-08 | 数量保存: 約定数量の買い合計 = 売り合計、板の数量 + 約定 = 投入 − 取消 | P | ✅ |
| LOB-09 | 成行は複数レベルを掃く | U | ✅ |
| LOB-10 | スプレッドを跨ぐ指値は攻撃的に約定し、残りが板に載る | U | ✅ |
| LOB-11 | 自己交差なし（板に bid ≥ ask が存在しない） | P | ✅ |
| LOB-12 | seed 固定フローで約定列が bit 一致 | D | ✅ |
| LOB-13 | `depth(N)` は上位 N レベルの (価格, 数量, 件数) | U | ✅ |
| LOB-14 | 注文プールはウォームアップ後にアロケーションしない。枯渇時は拒否を返す | U | ✅ |

### 6.2 HAWKES — `core/models/hawkes.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| HAWKES-01 | λ(t) = μ + Σ α e^{−β(t−tᵢ)} が手計算と一致 | N | ✅ |
| HAWKES-02 | λ(t) ≥ μ | P | ✅ |
| HAWKES-03 | thinning で生成した件数の平均 ≈ μT/(1−α/β)（4 SE。定常近似。空履歴からの厳密な期待値 μ/(1−η)·[T − η(1−e^{−β(1−η)T})/(β(1−η))] とも 4 SE で照合。T=200, R=200） | S | ✅ |
| HAWKES-04 | 分岐比 α/β ≥ 1 は拒否 | U | ✅ |
| HAWKES-05 | seed 固定で事象列が一致 | D | ✅ |
| HAWKES-06 | 再帰 O(n) 対数尤度 = 直接 O(n²) 計算（相対 1e-10） | N | ✅ |
| HAWKES-07 | 補償子 ∫λ の閉形式が数値積分と一致（1e-8） | N | ✅ |
| HAWKES-08 | 5000 事象から MLE が (μ, α, β) を推定 SE の 4 倍以内で復元 | S | ✅ |
| HAWKES-09 | 時間再スケール残差の平均 ≈ 1（単位指数、4 SE） | S | ✅ |

### 6.3 LOBSCENE / HEAT

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| LOBSCENE-01 | Model 契約充足、Snapshot（上位 N レベル, 直近約定, λ）は POD | K | ✅ |
| LOBSCENE-02 | 「大口注入」Command で best ask が上がる（買い）／best bid が下がる（売り） | U | ✅ |
| LOBSCENE-03 | Snapshot の深度が板の `depth(N)` と一致（dual-run） | D | ✅ |
| LOBSCENE-04 | Reset で板が空・λ が μ に戻る | U | ✅ |
| LOBSCENE-05 | シーンの逐次 thinning（1 ms 窓ごとに再開）で生成した到着数の平均が閉形式の期待値と一致（成行・取消 0 %、seed 固定、4 SE。α=0 の Poisson と η=0.5 の両方） | S | ✅ |
| HEAT-01 | 2D History（価格 × 時間）の寸法が固定、時間方向に循環。`push_column` は Rows 未満の列を無視し、長い列は先頭 Rows 要素だけ使う | U | ✅ |
| HEAT-02 | 循環後の列順が最古→最新 | U | ✅ |
| HEAT-03 | `clear` で全ゼロ | U | ✅ |

---

## 7. M4 仕様 — 動的処理

### 7.1 LSM — `core/pricing/lsm.hpp`, `scenes/lsm_model.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| LSM-01 | アンチセティック対の Z の和は厳密に 0 | U | ⬜ |
| LSM-02 | American put 価格が CN（FDM-06 の設定）の 95 % CI 内 | S | ⬜ |
| LSM-03 | 同パスの European MC 価格以上 | P | ⬜ |
| LSM-04 | 基底数 2→5 で価格差が CI 内 | S | ⬜ |
| LSM-05 | seed 固定で価格 bit 一致 | D | ⬜ |
| LSM-06 | σ=0 で決定的ペイオフ | N | ⬜ |
| LSM-07 | SE が 1/√N で減少（N を 4 倍で SE 半分、±20 %） | S | ⬜ |
| LSMSCENE-01 | Model 契約充足、Snapshot（縮約パス K 本 + 分位帯, 継続価値フィット）は POD | K | ⬜ |
| LSMSCENE-02 | 1 step = 1 時点の後ろ向き回帰 | U | ⬜ |
| LSMSCENE-03 | 全時点処理後の価格 = `lsm` 単体 | D | ⬜ |

### 7.2 AC — `core/exec/almgren_chriss.hpp`, `scenes/exec_model.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| AC-01 | 閉形式軌道 x_j = X·sinh(κ(T−t_j))/sinh(κT) と一致（相対 1e-10） | N | ⬜ |
| AC-02 | Σ 取引量 = X（絶対 1e-9） | P | ⬜ |
| AC-03 | λ=0 で線形（TWAP） | N | ⬜ |
| AC-04 | λ 増大で前倒し（全内点で残量が小さくなる） | P | ⬜ |
| AC-05 | 期待コスト・分散の閉形式が Monte Carlo と一致（4 SE） | S | ⬜ |
| AC-06 | フロンティア: λ↑ でコスト↑・分散↓ | P | ⬜ |
| AC-07 | κ = √(λσ²/η) の実装が定義と一致 | U | ⬜ |
| ACSCENE-01 | Model 契約充足、Snapshot（λ 別軌道 固定 L 本, フロンティア格子）は POD | K | ⬜ |
| ACSCENE-02 | SetParam(λ) で軌道が変わる | U | ⬜ |
| ACSCENE-03 | （オプション）M3 の板に流したときの実現コストが期待コスト ± 4 SE | S | ⬜ |

### 7.3 HJB — `core/exec/hjb_merton.hpp`, `scenes/hjb_model.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| HJB-01 | CRRA で最適比率 π* = (μ−r)/(γσ²) が全 w で定数（1e-3） | N | ⬜ |
| HJB-02 | 価値関数が w で凹 | P | ⬜ |
| HJB-03 | 価値関数が w で単調増加 | P | ⬜ |
| HJB-04 | 終端条件 V(w,T) = U(w) | U | ⬜ |
| HJB-05 | 格子細分で解析解への誤差が減少 | N | ⬜ |
| HJB-06 | 時間整合: T/2 から V(·,T/2) を終端として解いた結果 = 全区間解 | N | ⬜ |
| HJBSCENE-01 | Model 契約充足、Snapshot（V(w) 固定格子, π*(w)）は POD | K | ⬜ |
| HJBSCENE-02 | 1 step = 1 時間反復、SetParam(γ) で π* が変わる | U | ⬜ |

---

## 8. M5 仕様 — AAD・性能

### 8.1 AAD — `core/aad/tape.hpp`

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| AAD-01 | `Var` の + − × ÷ の偏微分が解析値と一致 | N | ⬜ |
| AAD-02 | exp / log / sqrt / 正規 CDF の微分が一致 | N | ⬜ |
| AAD-03 | 合成関数で連鎖律が成立 | N | ⬜ |
| AAD-04 | 複数入力に対する全勾配を 1 回の逆伝播で得る | U | ⬜ |
| AAD-05 | `rewind` でテープ長 0 | U | ⬜ |
| AAD-06 | テープ長 = 演算回数 | U | ⬜ |
| AAD-07 | ランダム式で数値微分と一致（相対 1e-8） | N | ⬜ |
| AAD-08 | `reserve` 後の記録・逆伝播でアロケーションなし | U | ⬜ |
| AAD-09 | BS on AAD の Δ = 解析 Δ（1e-12） | N | ⬜ |
| AAD-10 | ν・ρ・Θ（T 微分）も解析値と一致 | N | ⬜ |
| AAD-11 | AAD Greeks = バンプ Greeks（相対 1e-6） | N | ⬜ |
| AAD-12 | (= BENCH-06) 全 Greeks の時間 ≤ 価格計算の 5 倍 | B | ⬜ |
| AADSCENE-01 | Model 契約充足、Snapshot POD | K | ⬜ |
| AADSCENE-02 | 3 手法（AAD / バンプ / 解析）の Greeks が Snapshot 上で一致 | N | ⬜ |
| AADSCENE-03 | 計算時間・テープ長が Snapshot に載る | U | ⬜ |

### 8.2 PERF — `viz/perf.hpp`（vizcore）

| ID | 仕様 | 種別 | 状態 |
|---|---|---|---|
| PERF-01 | 固定ビンのヒストグラムに値が正しく分類される（境界含む） | U | ⬜ |
| PERF-02 | p50 / p95 / p99 が整列済み配列の定義と一致 | U | ⬜ |
| PERF-03 | EMA レートが定常入力で真値に収束 | N | ⬜ |
| PERF-04 | dropped 比率 = dropped / (dropped + received) | U | ⬜ |

---

## 9. ベンチマーク一覧（`[!benchmark]`）

| ID | 対象 | 目標 | M | 状態 |
|---|---|---|---|---|
| BENCH-01 | SpscRing push+pop（Snapshot 72 B） | < 20 ns | M0 | ✅ 4.4 ns |
| BENCH-02 | StreamingModel step+snapshot | < 100 ns | M0 | ✅ 39 ns |
| BENCH-03 | BS ストリップ SIMD vs スカラ（N=1024、厳密版: BS-10 で bit 一致） | ≥ 1.2×（改定） | M1 | ✅ 1.19〜1.28×（GCC 15, SSE2〜AVX-512） |
| BENCH-04 | サーフェス 200×200 の z・法線更新 | < 2 ms | M2 | ✅ 0.22 ms（加重中心差分、GCC 15 -O3） |
| BENCH-05 | マッチングエンジン 1e6 注文/秒（dropped 0） | ≥ 1e6/s | M3 | ✅ 19〜22 M 注文/秒（`MatchingEngine<65536,4096>`、70/10/20 混合、拒否 0） |
| BENCH-06 | AAD 全 Greeks / 価格 1 回 | ≤ 5× | M5 | ⬜ |

---

### 9.1 判断記録 — BENCH-03 の目標改定（M1）

| 項目 | 内容 |
|---|---|
| 状況 | BS-10 は「ストリップ版の各要素がスカラ版と相対 1e-15 で一致」を要求する。実装はスカラ版と bit 一致を達成したが、そのためには `erfc` と `log` をレーンごとに libm のスカラ関数で呼ぶしかなく、ベクトル化されるのは四則演算だけになる。 |
| 推論 | (1) ストライク 1 本あたりのコストは `log` 1 回 + `erfc` 2 回（≈ 45 ns）が支配的で、四則演算は数 ns。(2) 多項式近似の Φ（A&S 7.1.26 等）をベクトル化すれば 2× 以上は出るが精度は 1e-7 で、深い OTM では `S·Φ(d1) − Ke^{−rT}Φ(d2)` の桁落ちで相対誤差がさらに増幅され、1e-13 でも通らない。(3) 「スカラ版 == 配列版」は設計書 §6.1 の不変条件であり、ベンチ目標より優先する。 |
| 結論 | BS-10 の厳密一致を維持し、BENCH-03 の目標を「≥ 1.2×」に改定する（実測 1.19〜1.21×、AVX-512 で 1.24〜1.28×）。高精度ベクトル `erfc`（Cody 型有理近似）による高速版は M5 の性能項目に繰り延べ、別 ID（許容 1e-13）で扱う。 |
| 結果 | （M5 で追記） |

## 10. 依存規則の機械検査

| ID | 仕様 | 方法 | 状態 |
|---|---|---|---|
| DEP-01 | `core/`, `bridge/`, `scenes/`, `viz/include/` に `imgui`, `implot`, `GLFW`, `GL/` の include が無い | CI で `grep -R` → 検出したら失敗 | ⬜（M1 で CI に追加） |
| DEP-02 | `core/` が `bridge/` `scenes/` `viz/` を include しない | 同上 | ⬜ |
| DEP-03 | 各層は `target_link_libraries` 経由でのみ include path を得る | CMake 構成で保証（レビュー） | ✅ |

---

## 11. カバレッジと CI ゲート

| ゲート | 条件 |
|---|---|
| 必須 | 全 U/N/P/S/K/D/C が Green（3 OS × Debug/Release） |
| 必須 | ASan+UBSan Green、TSan（RING / RUNNER / VIZ / TRIPLE）Green |
| 必須 | `-Werror` |
| 必須（M1〜） | DEP-01/02 |
| 情報 | BENCH の前回比。2 倍以上の退行はレビューで理由を記録 |
| 手動 | viewer チェックリスト（`02_implementation_plan.md` §5.3） |

---

## 12. 索引 — ID → ファイル

| プレフィックス | テストファイル | 対象 |
|---|---|---|
| RING | `tests/test_spsc_ring.cpp` | `bridge/spsc_ring.hpp` |
| CLOCK | `tests/test_sim_clock.cpp` | `bridge/sim_clock.hpp` |
| RUNNER | `tests/test_runner.cpp` | `bridge/runner.hpp` |
| GBM | `tests/test_gbm.cpp` | `core/models/gbm.hpp` |
| WEL | `tests/test_welford.cpp` | `core/stats/welford.hpp` |
| EWMA | `tests/test_ewma.cpp` | `core/stats/ewma.hpp` |
| STREAM | `tests/test_streaming_model.cpp` | `scenes/streaming_model.hpp` |
| HIST | `tests/test_history.cpp` | `viz/include/quantviz/viz/history.hpp` |
| BENCH | `tests/bench_*.cpp` | 各対象 |
| VIZ, BS, GREEKS, OPTIM, GARCH, KALMAN | `tests/test_<module>.cpp`（M1） | 〃 |
| GL, SURF, TRIPLE, VOLSURF, TRIDIAG, FDM, FDMSCENE | 〃（M2） | 〃 |
| LOB, HAWKES, LOBSCENE, HEAT | 〃（M3） | 〃 |
| LSM, LSMSCENE, AC, ACSCENE, HJB, HJBSCENE | 〃（M4） | 〃 |
| AAD, AADSCENE, PERF | 〃（M5） | 〃 |
| DEP | `.github/workflows/ci.yml` | 依存規則 |
