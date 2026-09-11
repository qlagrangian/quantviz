# quantviz 設計書

| 項目 | 内容 |
|---|---|
| 版 | 1.0（M0 実装時点） |
| 対象 | C++20 / CMake ≥ 3.25 / ImGui + ImPlot（M2 以降 OpenGL 3.3） |
| 関連文書 | `02_implementation_plan.md`（実装計画）, `03_tdd_spec.md`（TDD 仕様） |

---

## 1. 目的

金融工学の理論（確率過程・オプション価格・時系列推定・マイクロストラクチャ・動的最適化）を C++ で実装し、その内部状態を**リアルタイムに 2D / 3D で描画**することで、次の 3 つを同時に深める。

1. **Coding** — GC なし・メモリレイアウト自由・型による最適化という C++ の強みを、①リアルタイム ②時系列 ③動的処理で体感する
2. **描画** — 理論の状態量（価格・分散・尤度面・価値関数・板）を「見える形」に写す設計
3. **金融理論** — スライダーを動かした瞬間に理論がどう反応するかを観察して理解する

実株価データは不要。全シーンは合成データ（GBM / Heston / Hawkes 等）で動く。

---

## 2. 設計原則 — コアと描画の双対

> **コアは描画を知らず、描画はコアを変更しない。両者を繋ぐのは Snapshot の一方通行と Command の一方通行の 2 本だけ。**

| # | 原則 | 具体的な規則 |
|---|---|---|
| P1 | **コアは描画を知らない** | `core/` と `bridge/` と `scenes/` は ImGui / ImPlot / GLFW / OpenGL を include しない（CMake の依存で強制） |
| P2 | **描画は Snapshot の純関数** | パネルは `Snapshot` と自分の UI 状態だけから描く。`Model` の型を参照するのは `Runner<M>` の型引数としてのみ |
| P3 | **境界は POD** | `Snapshot` と `Command` は `trivially copyable`・固定長・ヒープ参照なし。リングバッファがゼロアロケーションになる |
| P4 | **時間は明示** | シミュレーション時刻（年）と壁時計（秒）を `SimClock` が変換する。速度・一時停止・単ステップはコマンドで制御 |
| P5 | **決定性** | 同じ seed・同じコマンド列なら同じ結果。RNG は `core::Rng` に集約し、統計テストは seed 固定で行う |

双対の対応表：

| コア側 | 描画側 |
|---|---|
| `Model::step(dt)` — 時間発展 | フレームループ — 60 fps |
| `Model::snapshot()` — 状態を POD に射影 | `Panel::draw(Snapshot)` — POD を図に射影 |
| `Model::apply(Command)` — 入力を受ける | スライダー / ボタン — Command を生成 |
| 計算スレッド | 描画スレッド |

---

## 3. アーキテクチャ

### 3.1 レイヤと依存規則

```
 Layer 4b  viz       ImGui / ImPlot / OpenGL を触る唯一の層。パネル = Snapshot → 描画
 Layer 4a  vizcore   GUI 非依存の描画補助（History 等）。単体テスト可能
 Layer 3   scenes    core の部品を束ねて Model 契約を満たす。Snapshot / Command の語彙を定義
 Layer 2   bridge    SpscRing, Command, Model concept, SimClock, Runner      （std のみ）
 Layer 1   core      数値コード：models, stats, pricing, micro, exec, aad     （外部依存ゼロ）
```

許される依存は**下向きのみ**：

| 層 | 依存してよい |
|---|---|
| core | なし（標準ライブラリのみ） |
| bridge | 標準ライブラリ（`<atomic>`, `<thread>`, `<stop_token>`） |
| scenes | core, bridge |
| vizcore | scenes（型のみ） |
| viz | すべて + ImGui / ImPlot / GLFW / OpenGL |

CMake では各層を INTERFACE ライブラリ（`quantviz::core` 等）にし、`target_link_libraries` でしか include path を得られないようにして規則を機械的に守らせる。

### 3.2 スレッドモデル

```
 描画スレッド (main)                          計算スレッド (Runner::loop)
 ─────────────────────                        ──────────────────────────
 glfwPollEvents                               elapsed = now - last
 panel.ingest():  while(runner.poll(s))  ◀──  snapshots_.try_push(model.snapshot())
 panel.draw():    ImPlot::PlotLine(...)       n = clock.due_steps(elapsed)
 panel.controls(): runner.send(cmd)     ──▶   commands_.try_pop(c) → dispatch(c)
 glfwSwapBuffers (vsync)                      for n: model.step(dt)
```

* スレッドはちょうど 2 本。`Runner` が `std::jthread` で計算スレッドを所有する
* 共有状態は `SpscRing` 2 本と `std::atomic<uint64_t>` の計数（dropped / steps）だけ
* `Model` と `SimClock` は計算スレッドが**所有**する。`start()` 後に外から触るのはデータ競合（テストでは `tick()` を使う）
* リングが満杯なら Snapshot は**捨てる**（`dropped_` に加算）。コアは描画に引きずられて止まらない
* `stop()` は「停止要求 → 未消化 Command の適用 → join」。`send()` が true を返した Command は必ず適用される

### 3.3 データフロー（1 フレーム）

1. 計算スレッドが `steps_per_second × speed` の速さで `step(dt)` を回し、`publish_every` ステップごとに Snapshot を push
2. 描画スレッドはフレーム先頭で ring を**空になるまで** poll し、`History`（描画側の循環バッファ）に時系列として蓄える
3. パネルは History と最新 Snapshot から描く
4. UI の変化は即座に `Command` として送る。適用は「次のステップから」

コアは「今の状態」しか吐かない。**時系列にするのは描画側の責務**。これによりコアの Snapshot は 1 枚分の固定長で済む。

---

## 4. 型契約

### 4.1 `Model` concept（`bridge/model_concept.hpp`）

```cpp
template <class S>
concept SnapshotType = std::is_trivially_copyable_v<S> && std::is_default_constructible_v<S>;

template <class M>
concept Model = requires(M& m, const M& cm, double dt, const Command& c) {
    typename M::Snapshot;
    requires SnapshotType<typename M::Snapshot>;
    { m.step(dt) };
    { cm.snapshot() } -> std::same_as<typename M::Snapshot>;
    { m.apply(c) };
};
```

| 契約 | 意味 |
|---|---|
| `step(dt)` | sim 時間を `dt`（年）だけ進める。例外を投げない設計にする |
| `snapshot() const` | 現在状態の POD 射影。O(フィールド数)、アロケーションなし |
| `apply(Command)` | `SetParam` / `Reset` を受ける。未知の `param_id` は**無視**。時計系コマンドは Runner が処理済みなので来ない前提だが、来ても無視する |

### 4.2 `Snapshot` の設計規則

* 固定長 POD。`std::vector` / ポインタ / `std::string` 禁止。配列は `std::array`
* 目安 ≤ 128 バイト（キャッシュライン 2 本）。大きな状態（板全体・グリッド）は**縮約**して載せる（M3 の板は上位 N レベル、M2 のサーフェスは描画グリッドの固定サイズ）
* `seq`（ステップ通番）を必ず含める。`seq == 0` は「未ステップ／Reset 直後」の意味で、描画側はスキップできる
* 描画に必要な**真値**（現在の σ 等）も載せる。参照線を描くために描画側が Model を覗く必要をなくす

### 4.3 `Command`（`bridge/command.hpp`）

```cpp
enum class CommandType : uint8_t { SetParam, Reset, Pause, Resume, StepOnce, SetSpeed };
struct Command { CommandType type; uint32_t param_id; double value; uint64_t seed; };
```

* 時計系（Pause / Resume / StepOnce / SetSpeed）→ `Runner` が `SimClock` に適用
* モデル系（SetParam / Reset）→ `Model::apply`
* `param_id` の名前空間はシーンごと（`StreamingModel::Param` 列挙）。0 は予約
* `Reset.seed == 0` は「現在の seed で再現」を意味する

### 4.4 `Runner<M, SnapCap, CmdCap>` の保証

| 保証 | 内容 |
|---|---|
| R1 | `send()` は決してブロックしない。満杯なら false |
| R2 | `poll()` は決してブロックしない。空なら false |
| R3 | Command は**ステップの前**に消化される（同一 tick 内で適用→反映） |
| R4 | Snapshot の `seq` は単調増加（`publish_every` の倍数） |
| R5 | リング満杯時はコアを止めず、`dropped_snapshots()` に加算 |
| R6 | `stop()` 後、`send()` が true を返していた Command は全て適用済み。`model()` を読んでも競合しない |
| R7 | `start()` は冪等。デストラクタは join する |
| R8 | `tick(elapsed)` は 1 ループ分の同期実行。`start()` 中に呼んではならない |

---

## 5. 時間モデル

### 5.1 二つの時間

| 時間 | 単位 | 持ち主 |
|---|---|---|
| sim 時刻 `t` | 年（`dt = 1/(252·390)` = 1 分足 など） | `Model` |
| 壁時計 | 秒 | `Runner::loop` が `steady_clock` で測る |

変換は `SimClock`：

```
due_steps(elapsed) = min( pending + floor( acc += elapsed · speed · steps_per_second ), max_steps_per_tick )
```

### 5.2 規則

* 一時停止中は壁時間を**蓄積しない** → 再開時にバーストしない
* `request_step()` は一時停止中でも 1 ステップだけ許す（デバッガの step と同じ）
* `max_steps_per_tick` を超えた分は**捨てる**（追いつこうとしない → spiral of death 防止）
* `speed ≤ 0`・NaN は 0 にクランプ
* `SimClock` は時間源を持たない純ロジック。壁時計は Runner が外から渡す → 単体テスト可能

### 5.3 「動的処理」の教材としての時間制御

M2 の CN-FDM では「後ろ向き反復」の 1 ステップを `StepOnce` で手動送りできる。動的計画法が**何を 1 回の反復で更新しているか**を目で追うのが狙い。時間制御は全シーン共通の UI にする。

---

## 6. モジュール設計

凡例: ✅ M0 実装済 / 🔜 マイルストーン番号

### 6.1 `core/`（外部依存ゼロ）

| モジュール | 責務 | 公開 API（要点） | 不変条件 |
|---|---|---|---|
| `rng.hpp` ✅ | 決定性 RNG | `normal()`, `uniform()`, `reseed(seed)` | 同 seed → 同系列。`reseed` は分布のキャッシュも捨てる |
| `models/gbm.hpp` ✅ | GBM 厳密離散化 | `step(dt)→log return`, `spot()`, `set_mu/sigma`, `reset(seed)` | `spot > 0`。σ<0 は 0 にクランプ。`reset` は mu/sigma を保持 |
| `stats/welford.hpp` ✅ | 逐次平均・分散 | `push`, `mean`, `variance`(n-1), `population_variance` | n<2 で分散 0。O(1) 更新 |
| `stats/ewma.hpp` ✅ | EWMA 分散（平均ゼロ仮定） | `push`, `variance`, `volatility`, `set_lambda` | λ∈(0,1) にクランプ。初回は r² で初期化 |
| `pricing/black_scholes.hpp` ✅M1 | BS 価格・Greeks、ストライク配列版（SIMD） | `bs_price(S,K,T,r,σ,type)`, `bs_greeks(...)`, `bs_price_strip(S, span<K>, T, r, σ, type, span<out>)`, `norm_cdf` | put-call parity、境界条件、スカラ版 == 配列版（bit 一致。算術は `bs_price_block<Ops>` 1 本に集約し、FMA 縮約は `QUANTVIZ_BS_USE_FMA` で両経路同一） |
| `stats/garch.hpp` 🔜M1 | GARCH(1,1) 尤度・フィルタ | `log_likelihood(ω,α,β, span<r>)`, `filter(...)` | α+β<1 の定常制約。σ²>0 |
| `stats/optim.hpp` 🔜M1 | Nelder-Mead / BFGS | `minimize(f, x0, opts)→{x, f, iters, path}` | 反復履歴を返す（尤度面上の軌跡描画用） |
| `math/mat.hpp` ✅M1 | 固定サイズ行列（`std::array`、ヒープなし） | `Mat<R,C>`, `transpose`, `inverse()`→`optional`（≤3×3 閉形式、`tol` は行列式スケールに対する相対値）, `is_symmetric`, `is_psd` | POD。NaN は全述語で拒否 |
| `stats/kalman.hpp` ✅M1 | 線形カルマン（固定サイズ） | `predict(F,Q)`, `update(H,z,R)`→イノベーション（事前残差）, `state`, `cov`, `reset`, `skipped_updates` | Joseph 形 + 明示的対称化で共分散は対称・半正定値。NaN 観測・特異 S は状態を壊さずスキップして数える（例外なし） |
| `pricing/fdm_cn.hpp` 🔜M2 | Crank–Nicolson + PSOR（American） | `init(grid)`, `step_backward()`, `values()`, `exercise_boundary()` | American ≥ intrinsic、≥ European |
| `math/tridiag.hpp` 🔜M2 | Thomas 法 | `solve(a,b,c,d)` | 密行列解と一致 |
| `micro/order_book.hpp` 🔜M3 | 板・マッチング | `submit(limit/market)`, `cancel`, `best_bid/ask`, `depth(N)` | bid<ask、価格時間優先、数量保存 |
| `models/hawkes.hpp` 🔜M3 | 自己励起過程 | `simulate(thinning)`, `intensity(t)`, `log_likelihood` | 分岐比 α/β<1 |
| `pricing/lsm.hpp` 🔜M4 | Longstaff–Schwartz | `price(paths, basis)` | ≥ European（MC 誤差内） |
| `exec/almgren_chriss.hpp` 🔜M4 | 最適執行 | `trajectory(X,T,λ,η,γ,σ)`, `frontier()` | Σ trades = X。λ=0 で TWAP |
| `exec/hjb_merton.hpp` 🔜M4 | Merton HJB 数値解 | `solve(grid)`, `optimal_fraction(w)` | CRRA で定数比率 |
| `aad/tape.hpp` 🔜M5 | 随伴自動微分 | `Var`, `Tape::rewind`, `Tape::propagate` | 解析微分・バンプと一致 |

### 6.2 `bridge/`（std のみ）

| モジュール | 責務 | 重要な設計判断 |
|---|---|---|
| `spsc_ring.hpp` ✅ | SPSC ロックフリー・リング | 2 の冪容量（マスク 1 命令）、head/tail と各キャッシュを別キャッシュラインに `alignas(64)`、生産者は head だけ書き消費者は tail だけ書く、満杯で false |
| `command.hpp` ✅ | 描画→コアの語彙 | POD、`static constexpr` ファクトリ、`is_clock_command` |
| `model_concept.hpp` ✅ | 契約 | `Model`, `SnapshotType` |
| `sim_clock.hpp` ✅ | 時間変換 | 時間源なし・純ロジック。蓄積は一時停止中に止まる |
| `runner.hpp` ✅ | 計算スレッド | `RunnerConfig` は非依存型（テンプレート引数違いの Runner 間で共有可）。`tick()` で同期テスト |
| `triple_buffer.hpp` 🔜M2 | 「最新 1 枚」だけ欲しい大きな状態（グリッド）向け | サーフェスは履歴不要 → ring より triple buffer が適切 |

### 6.3 `scenes/`（core + bridge）

1 シーン = 1 `Model` 実装 + Snapshot 定義 + Param 列挙。描画パネルは `viz/` に置く（同名で対応）。

| シーン | Model | Snapshot の主内容 | Param |
|---|---|---|---|
| Streaming ✅ | `StreamingModel` | t, spot, log_return, Welford 平均/分散, EWMA 分散, μ/σ 真値, seq | mu, sigma, ewma_lambda |
| Greeks 🔜M1 | `GreeksModel` | S, グリークス配列（ストライク軸 固定 N）, サーフェス格子（S×T 固定） | S, r, σ, T |
| Garch 🔜M1 | `GarchModel` | σ_t 推定, 真値, 尤度面（α×β 固定格子）, 最適化軌跡（最新 K 点） | ω, α, β（真値）, optimizer |
| Kalman 🔜M1 | `KalmanPairModel` | 2 価格, β 推定, β 分散, スプレッド | 観測ノイズ, 状態ノイズ, 真の β |
| Fdm 🔜M2 | `FdmAmericanModel` | V(S) の現在ステップ, 行使境界, 残り反復数 | K, r, σ, q, グリッド |
| Lob 🔜M3 | `LobModel` | 上位 N レベル bid/ask, 直近約定, λ(t) | 到着率, Hawkes α/β, 大口注入 |
| Lsm / Exec / Hjb 🔜M4 | 各 Model | パス束の縮約, 執行軌道, 価値関数格子 | シーン固有 |
| Aad 🔜M5 | `AadModel` | Greeks（AAD / バンプ）, 計算時間, テープ長 | 入力 |

### 6.4 `viz/`

| モジュール | 責務 |
|---|---|
| `history.hpp`（vizcore） ✅ | 描画側の固定長循環履歴。ImPlot の `offset` 規約（満杯時 offset = 最古の index） |
| `panels/streaming_panel.*` ✅ | Spot / Volatility / Control の 3 ウィンドウ |
| `main.cpp` ✅ | GLFW + ImGui + ImPlot の起動・フレームループ・終了 |
| `panels/<scene>_panel.*` 🔜 | シーンごとに 1 パネル。`draw(Runner<M>&)` の形を揃える |
| `gl/surface_renderer.*` 🔜M2 | グリッド → 三角形メッシュ → 法線 → 単純ライティング → カメラ。ImGui ウィンドウ内にテクスチャとして描く |
| `clock_controls.hpp`（vizcore）✅M1 | 時計 UI の状態 `ClockControlState` と Command 生成の純関数（`toggle_pause`, `set_speed`, `step_once`, `reset`, `step_allowed`）。ImGui 非依存でテスト可能（VIZ-03） |
| `panels/clock_panel.hpp` ✅M1 | 上記に ImGui を被せた共通ウィジェット `draw_clock_controls` と共通テレメトリ行 `draw_runner_telemetry`。全シーンの Control ウィンドウが使う |
| `scene_registry.hpp`（vizcore）✅M1 | `Scene`（Runner + Panel の型消去）, `RunnerScene<M, Panel, SnapCap>`（唯一の具象、デストラクタで join）, `SceneRegistry`（名前→生成関数。`select` は前シーンを `stop()` してから破棄し、新シーンを `start()`）。`main.cpp` はメニューバーで切り替えるだけ。生きているシーンは常に高々 1 つ |

---

## 7. 性能設計

| 観点 | 方針 | 測定 |
|---|---|---|
| アロケーション | コアのホットパス（`step`, `snapshot`, ring）はゼロアロケーション。Snapshot は固定長 | `BENCH-01/02` |
| false sharing | ring の head / tail / 各キャッシュを別キャッシュラインに配置 | `RING-08`（レイアウト検査） |
| 同期コスト | SPSC + acquire/release のみ。mutex・condvar なし | `BENCH-01`（目標 < 20 ns / push+pop） |
| 描画とコアの分離 | vsync（60 fps）とコアの `steps_per_second` は独立。描画の遅延はコアに伝播しない（drop で吸収） | テレメトリ `dropped`, `queued` |
| SIMD ✅M1 | Black–Scholes のストライク配列版で `std::experimental::simd`（無ければ AVX2 intrinsics、無ければスカラ）。スカラ版と **bit 一致**（`FP_FAST_FMA` に応じて両経路で同じ縮約を行う）。超越関数はレーンごとに libm を呼ぶため速度は ≈1.2×。高速近似版は M5 | `BS-10`, `BENCH-03` |
| メモリ配置 🔜M2/M3 | FDM グリッドは連続配列（SoA）、板は価格レベル配列 + intrusive list | `BENCH-xx` |
| 計測 🔜M5 | シーンごとの `step` 時間・フレーム時間・dropped をパフォーマンスパネルで常時表示。`perf` でキャッシュミス |  |

M0 実測（GCC 13, -O3, Xeon 想定）：`push+pop ≈ 4.4 ns`、`StreamingModel::step + snapshot ≈ 39 ns`。

---

## 8. 数値設計と決定性

* **RNG**: `std::mt19937_64` を `core::Rng` に閉じ込める。分布の `reset()` を `reseed` で必ず呼ぶ（Box–Muller のキャッシュ対策）
* **離散化**: GBM は厳密解を使う（離散化誤差ゼロ）。Heston 🔜 は QE スキーム（Andersen）を採用予定
* **年率換算**: ステップ分散 → 年率 σ は `sqrt(var_step / dt)`。描画側で行う
* **許容誤差の方針**（`03_tdd_spec.md` §2.4 に詳細）: 解析解比較は相対 1e-10〜1e-6、モンテカルロは seed 固定 + 4 SE、FDM は格子収束次数を測る
* **浮動小数の UI**: ImGui スライダーは `float`。Command には `double` で載せる（精度はコア側で決める）

---

## 9. 描画設計

### 9.1 2D パネル規約（ImPlot）

* 1 ウィンドウ = 1 主題（Spot / Volatility / 尤度面 …）。Control は別ウィンドウ
* 時間軸は取引日（`t · 252`）。`follow latest` と `window (days)` は全シーン共通
* 参照線（真値）は History として持ち、パラメータ変更が「段」として見えるようにする
* 縮約された固定長配列（Greeks ストリップ等）は `PlotLine(xs, ys, N)` で直接描く

### 9.2 描画側 History

`History<N>`（固定長）を系列ごとに持つ。`push(x, y)`、ImPlot に `xs(), ys(), count(), offset()` を渡す。Reset 時は `clear()`。N は「1 画面で見たい期間 × steps_per_second」から決める（M0: 8192 ≈ 16 秒 @ 500 steps/s）。

### 9.3 3D サーフェス（M2）

```
 Snapshot(grid N×M, z 値)  →  頂点バッファ更新（位置 + 法線）  →  VBO/EBO  →  シェーダ（Lambert + 等高線）
                                                                        → FBO → ImGui::Image
```

* 頂点の x,y は固定（グリッド）。毎フレーム更新するのは z と法線のみ → `glBufferSubData`
* カメラは軌道カメラ（マウスドラッグ回転、ホイールでズーム）。ImPlot3D を採用する場合はこの層を差し替えるだけ
* 大きなグリッドは ring ではなく **triple buffer** で「最新 1 枚」を渡す

### 9.4 UI → Command 規約

* スライダーは変更されたフレームだけ `send`。失敗（ring 満杯）は次のフレームで値が変わればまた送られるので握り潰してよい
* Pause/Resume はトグルボタン 1 つ。`Step` は一時停止中のみ有効
* Reset は `Command::reset()` を送り、同時に描画側の History を `clear()`

### 9.5 フレーム予算

60 fps = 16.6 ms。目標: `ingest`（poll + History 更新）≤ 0.5 ms、`draw` ≤ 3 ms。Snapshot 到着が多い（高 `steps_per_second`）ときは `publish_every` で間引く。

---

## 10. エラー処理・堅牢性

| 状況 | 方針 |
|---|---|
| UI からの不正値（負の σ、λ∉(0,1)、NaN） | コア側でクランプ／デフォルトへ。例外は投げない（`noexcept`） |
| ring 満杯 | Snapshot は捨てて計数、Command は `send` が false。両方ノンブロッキング |
| 未知の `param_id` | 無視 |
| GLFW / GL 初期化失敗 | `main` が非 0 で終了。ログは stderr |
| 数値の発散（FDM 不安定等） 🔜 | Snapshot に `status` フラグを載せ、描画が警告表示。コアは停止しない |
| スレッド終了 | `Runner::stop()` が join まで責任を持つ。`jthread` で例外経路でも join |

---

## 11. 拡張手順 — 新しいシーンを追加する

1. **core** に純粋な数値部品を書く（テスト先行: `03_tdd_spec.md` に ID を追加）
2. **scenes** に `XxxSnapshot`（POD, ≤128B 目安, `seq` 含む）と `XxxModel`（`step` / `snapshot` / `apply` / `Param`）を書き、`static_assert(bridge::Model<XxxModel>)`
3. **契約テスト**を書く（`STREAM-01` と同型: concept 充足・POD・サイズ・決定性の dual-run）
4. **viz/panels** に `XxxPanel { void draw(Runner<XxxModel>&); }` を書く。History と共通 Control を再利用
5. `main.cpp`（M1 以降は `scene_registry`）に登録
6. `02_implementation_plan.md` の DoD を満たすことを確認（テスト全緑、ベンチ、`-Werror`、サニタイザ）

---

## 12. ビルド・規約

### 12.1 ツールチェーン

* C++20（`concepts`, `std::jthread`, `std::stop_token`, `SeparatorText` 等）。GCC ≥ 12 / Clang ≥ 15 / MSVC ≥ 19.34
* CMake ≥ 3.25、Ninja、vcpkg（manifest モード）。vcpkg なしの Linux ではシステム GLFW + FetchContent（imgui / implot）
* 警告: `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wold-style-cast`（`quantviz::warnings`）。CI は `-Werror`
* 依存ライブラリは `SYSTEM` include で警告対象外

### 12.2 命名・コーディング規約

| 対象 | 規約 | 例 |
|---|---|---|
| 名前空間 | `quantviz::<layer>` | `quantviz::core`, `quantviz::bridge` |
| 型 | PascalCase | `SpscRing`, `StreamingSnapshot` |
| 関数・変数 | snake_case | `due_steps`, `steps_per_second` |
| メンバ変数 | 末尾 `_` | `head_`, `cfg_` |
| 定数 | `k` 接頭 | `kCacheLineSize`, `kDt` |
| Param 列挙 | `k` 接頭 + シーン内で一意 | `kMu = 1` |
| ヘッダ | `#pragma once`、ファイル先頭に 1 段落の責務コメント |  |
| 例外 | コアのホットパスでは投げない。設定時のクランプで吸収 |  |

### 12.3 ディレクトリ

```
quantviz/
  CMakeLists.txt  CMakePresets.json  vcpkg.json  README.md
  core/include/quantviz/core/{rng.hpp, models/, stats/, pricing/, micro/, exec/, aad/}
  bridge/include/quantviz/bridge/{spsc_ring, command, model_concept, sim_clock, runner}.hpp
  scenes/include/quantviz/scenes/<scene>_model.hpp
  viz/include/quantviz/viz/history.hpp        viz/src/{main.cpp, panels/, gl/}
  tests/test_<module>.cpp  tests/bench_<module>.cpp
  docs/01_design.md  02_implementation_plan.md  03_tdd_spec.md
  .github/workflows/ci.yml
```

---

## 13. 非目標

* 実データの取込み・ブローカー接続・発注（研究用途に限定）
* 分散実行・GPU 計算（M5 までは単一プロセス 2 スレッド）
* 汎用プロットライブラリ化（描画はシーン専用パネルで良い）
* ImGui docking ブランチ依存（標準ブランチで動く UI にする）

---

## 付録 A. M0 の Snapshot / Command 定義

```cpp
struct StreamingSnapshot {          // 72 bytes
    double   t, spot, log_return;
    double   mean_return, var_return;   // Welford
    double   ewma_var;                  // EWMA
    double   mu_true, sigma_true;       // 参照線
    uint64_t seq;                       // 0 = 未ステップ
};

enum StreamingModel::Param : uint32_t { kMu = 1, kSigma = 2, kEwmaLambda = 3 };

// 描画 → コア
Command::set_param(kSigma, 0.35);  Command::set_speed(4.0);  Command::pause();
Command::step_once();              Command::resume();        Command::reset(/*seed=*/0);
```

## 付録 B. 用語

| 用語 | 意味 |
|---|---|
| Snapshot | Model の状態の POD 射影。コア→描画の唯一の出力 |
| Command | 描画→コアの唯一の入力 |
| Scene | 1 テーマ = 1 Model + 1 Panel の対 |
| sim 時刻 | モデル内部の時間（年） |
| tick | Runner の 1 ループ（コマンド消化 → ステップ → 発行） |
| publish_every | k ステップに 1 回 Snapshot を出す間引き |
