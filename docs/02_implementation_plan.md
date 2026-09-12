# quantviz 実装計画書

| 項目 | 内容 |
|---|---|
| 版 | 1.0（M0 完了時点） |
| 期間の目安 | 個人開発・週末中心で約 5〜6 か月（M0: 完了 / M1〜M5: 各 3〜5 週末） |
| 関連文書 | `01_design.md`（設計）, `03_tdd_spec.md`（テスト仕様。本書のタスクはテスト ID を参照する） |

---

## 1. 前提と環境

| 項目 | 内容 |
|---|---|
| 言語 / ビルド | C++20, CMake ≥ 3.25, Ninja, vcpkg（manifest）。Linux では vcpkg 無しでもビルド可（`README.md` B） |
| 描画 | GLFW 3, Dear ImGui 1.91.x, ImPlot 0.16。M2 で OpenGL 3.3 core（`glad` or ImGui 同梱ローダ） |
| テスト | Catch2 v3（vcpkg or FetchContent）。ベンチは `[!benchmark]` タグ |
| 検証 | GitHub Actions: 3 OS × Debug/Release、ASan+UBSan、TSan（ring / runner）、viewer ビルド |
| エディタ | `compile_commands.json` を出力（clangd）。`.clang-format` 同梱 |

---

## 2. 開発プロセス

### 2.1 TDD サイクル（1 タスクの単位）

1. `03_tdd_spec.md` の該当 ID を確認（無ければ ID を追加してから始める）
2. **Red**: `tests/test_<module>.cpp` に `TEST_CASE("<ID>: <仕様文>", "[module][kind]")` を書き、失敗を見る
3. **Green**: 最小の実装で通す
4. **Refactor**: 警告ゼロ（`-Werror`）、命名規約、責務コメント
5. 数値・統計テストは seed を固定し、許容誤差の根拠（SE 何倍か）をテスト内コメントに書く
6. コミット（1 ID = 1 コミットが理想。メッセージ先頭に ID）

### 2.2 完了条件（全マイルストーン共通の Definition of Done）

- [ ] 該当 ID のテストが全て Green、既存テストが全て Green（`ctest`）
- [ ] `-DQUANTVIZ_WARNINGS_AS_ERRORS=ON` でビルド成功（GCC / Clang）
- [ ] ASan+UBSan で Green。並行部品を触った場合は TSan で Green
- [ ] 依存規則（`01_design.md` §3.1）違反なし。core に GUI include なし
- [ ] 新シーンは契約テスト（concept 充足・POD・サイズ・dual-run 決定性）を持つ
- [ ] `BENCH-xx` を回し、前マイルストーン比で 2 倍以上の退行が無い
- [ ] viewer を起動し、スライダー操作・Pause/Step/Reset・60 fps 維持を**手動確認**（チェックリスト §5.3）
- [ ] `03_tdd_spec.md` の状態欄を ✅ に更新、`01_design.md` に設計変更を反映

### 2.3 ブランチとコミット

* `main` は常に Green。マイルストーンごとに `m1-greeks` 等のブランチ → PR → squash
* コミットメッセージ: `RING-07: producer/consumer 1M items stress test` のように ID 先頭
* 設計変更は同じ PR で `docs/` を更新（ドキュメントとコードの乖離を作らない）

### 2.4 レビュー観点（セルフレビュー用）

1. Snapshot にヒープ参照が入っていないか（`static_assert(is_trivially_copyable)`）
2. Command の適用が「次のステップから」になっているか（RUNNER-03）
3. 数値テストの許容誤差に根拠があるか（4 SE、解析解、収束次数）
4. ホットパスにアロケーション・mutex・例外が無いか
5. 描画が Model を直接参照していないか

---

## 3. マイルストーン一覧

| M | テーマ | 主な成果物 | 核となる C++ の学び | 核となる理論の学び | 状態 |
|---|---|---|---|---|---|
| **M0** | 骨組み・Streaming | SpscRing, SimClock, Runner, Gbm, Welford, EWMA, StreamingModel, viewer 3 パネル, CI | ロックフリー SPSC、concept、jthread、POD 境界 | GBM 厳密解、逐次統計、EWMA の窓 | ✅ 完了 |
| **M1** | 2D 理論パネル | Black–Scholes + Greeks（SIMD）, GARCH(1,1) MLE, Kalman ヘッジ比率, 共通 Control, シーン切替 | SIMD、固定サイズ行列テンプレート、最適化器の自作 | Γ の尖り、尤度面の谷、フィルタの追従 | ✅ 完了 |
| **M2** | 3D サーフェス・FDM | OpenGL サーフェス描画（自作）, ボラ面, Thomas 法, Crank–Nicolson + PSOR, 行使境界, triple buffer | GL パイプライン、連続メモリ、三重対角 | American の早期行使境界、後ろ向き反復 | ✅ 完了 |
| **M3** | マイクロストラクチャ | 板 + マッチングエンジン, Hawkes 生成・推定, 板深度ラダー, 価格×時間ヒートマップ | intrusive list、カスタムアロケータ、O(n) 再帰 | 価格時間優先、自己励起、板の崩れ | ✅ 完了 |
| **M4** | 動的処理 | LSM, Almgren–Chriss, Merton HJB, パス束・執行軌道・価値関数面 | パス並列、DP のメモリ設計 | 最適停止、執行のリスク回避、HJB | ⬜ |
| **M5** | AAD・性能 | テープ式 AAD, Greeks 比較, パフォーマンスパネル, perf 連携 | 演算子オーバーロード、テープ設計、計測 | 逆伝播 1 回で全 Greeks | ⬜ |

依存関係：

```
 M0 ──▶ M1 ──▶ M2 ──▶ M4
          └──▶ M3 ──┘
                     └──▶ M5
```

M2 と M3 は独立に進められる。M4 は M2（FDM が LSM の参照解）と M3（板が執行の環境）に依存。

---

## 4. マイルストーン詳細

### M0 — 骨組み・Streaming ✅

**目的** 双対構造を最小構成で動かし、以後の全シーンが乗る土台を固める。

**スコープ** `core/{rng, gbm, welford, ewma}`, `bridge/*`, `scenes/streaming_model`, `viz/{history, streaming_panel, main}`, `tests/*`, CI。

**成果物**

| ファイル | 内容 |
|---|---|
| `bridge/spsc_ring.hpp` | SPSC ロックフリー・リング（キャッシュライン分離） |
| `bridge/sim_clock.hpp` | sim/壁時計変換（純ロジック） |
| `bridge/runner.hpp` | 計算スレッド・コマンド消化・Snapshot 発行・`tick()` |
| `core/models/gbm.hpp` 他 | GBM 厳密離散化、Welford、EWMA、Rng |
| `scenes/streaming_model.hpp` | Model 契約の最初の実装 |
| `viz/*` | Spot / Volatility / Control |
| `tests/*` | 59 ケース（RING/CLOCK/RUNNER/GBM/WEL/EWMA/STREAM/HIST）+ BENCH 2 本 |

**完了確認** 全テスト Green（GCC 13, -O3, -Werror）、viewer 起動（Xvfb 下でも描画確認）、`snapshots/s ≈ steps_per_second`、`dropped = 0`。

**M0 で見送ったもの → 次へ** triple buffer（M2）、共通 Control の分離（M1）、シーン切替（M1）、docking（非目標）。

---

### M1 — 2D 理論パネル（Greeks / GARCH / Kalman）✅

**実績（完了時点）** 4 シーン（Streaming / Greeks / GARCH / Kalman pair）を `SceneRegistry` + メニューバーで切替。テスト 108 ケース（M0 の 59 + 49）、ASan/UBSan/TSan Green、`-Werror` を GCC 15 で確認。計画からの主な逸脱: BENCH-03 の目標を ≥ 1.2× に改定（`03_tdd_spec.md` §9.1、厳密一致を優先）、Snapshot サイズ上限の例外（Greeks 16.5 KB、GARCH 9.8 KB、`SnapCap` 256 — M2 の triple buffer で解消予定）、Kalman の β_t は平均回帰付きランダムウォーク（フィルタは F=1 のまま、意図的な軽い誤特定）。並列実装のために各タスクを隔離 worktree で進め、仕様レビューと品質レビューを 2 段で通した（`docs/superpowers/plans/2026-09-12-m1-2d-panels.md`）。


**目的** ImPlot だけで完結する 3 シーンを追加し、「シーンを増やす手順」を確立する。

**スコープ外** 3D、OpenGL 直叩き。

**タスク（順序 = 推奨着手順、括弧内 = テスト ID）**

1. **共通化** — `viz/panels/common_controls.*`（時計 UI + テレメトリ）を Streaming から抽出。`scene_registry` で起動時にシーン選択（各シーン独立 Runner）。（`VIZ-01..03`）
2. **Black–Scholes コア** — 価格、Δ Γ ν Θ ρ、`price_strip(span<K>)`。まずスカラ版。（`BS-01..09`）
3. **SIMD 版** — ストライク軸をベクトル化（`std::experimental::simd` があれば優先、無ければ AVX2 intrinsics を `#ifdef`）。スカラ版と一致。（`BS-10..12`）
4. **Greeks シーン** — `GreeksModel`：S を GBM で動かし、固定 N ストライクの Greeks 配列と Γ(S,T) 格子（M2 で 3D 化する前提で 2D は等高線表示）。（`GREEKS-01..05`）
5. **最適化器** — Nelder–Mead → BFGS（数値勾配）。反復履歴を返す。（`OPTIM-01..06`）
6. **GARCH コア** — 尤度・フィルタ、定常制約のパラメータ変換（logit / softplus）。（`GARCH-01..08`）
7. **GARCH シーン** — 合成 GARCH 過程を流し、ローリング窓で MLE を回し、σ_t 推定 vs 真値、尤度面 L(α,β) 固定格子 + 最適化軌跡。（`GARCH-09..11`）
8. **Kalman コア** — 固定サイズ `Mat<N,M>` テンプレートと線形カルマン。（`KALMAN-01..08`）
9. **Kalman シーン** — 共和分ペアを生成、β を追跡、信頼帯とスプレッド。（`KALMAN-09..11`）

**DoD 追加項目**
- 3 シーンが `scene_registry` から選べ、Streaming を含む 4 シーンで Pause/Step/Reset が同じ UI で動く
- BS の SIMD 版がスカラ版より高速（`BENCH-03`、目標 ≥ 1.2×。≥ 2× は BS-10 の厳密一致と両立しないため改定、`03_tdd_spec.md` §9.1）

**リスク** `std::experimental::simd` のコンパイラ差 → intrinsics へのフォールバックを最初から用意。GARCH の MLE が局所解 → 複数初期値・パラメータ変換で対処、テストは合成データの復元幅を SE ベースで緩める。

---

### M2 — 3D サーフェス・FDM ✅

**実績（完了時点）** 6 シーン（+ Vol surface, FDM American、Greeks の Γ 3D 化）。テスト 147 ケース（M1 の 108 + 39）、ASan/UBSan/TSan（TRIPLE 追加）Green、`-Werror`。BENCH-04 は 200×200 の z・法線更新 0.22 ms（目標 2 ms）。FDM シーンは 200×200 の面を毎ステップ TripleBuffer 経由で渡し、llvmpipe（ソフトウェア GL）でも 10〜12 ms/フレーム。計画からの主な逸脱: 行使境界は 3D 面上ではなく専用の 2D ウィンドウ（レンダラは三角形のみ）、GL ローダは外部依存なしで `glfwGetProcAddress` から必要な 53 関数だけを解決、`SurfaceModel` 概念 + `Runner` の面チャネルで Snapshot（リング）と Surface（最新 1 枚）を分離、M1 で例外扱いだった大きな Snapshot は Greeks / GARCH に残る（M3 以降の整理候補）。Runner が一時停止中のコマンドを publish しない穴（RUNNER-10）は M3 冒頭で対応。


**目的** 自作 OpenGL サーフェス描画を作り、Crank–Nicolson の後ろ向き反復を「手で 1 ステップずつ送れる 3D アニメ」にする。本 PJ の山場。

**タスク**

1. **GL 基盤** — FBO にレンダして `ImGui::Image` で表示。軌道カメラ。（`GL-01..04`：ユニットは行列演算・カメラ数学のみ、描画は手動）
2. **サーフェスメッシュ** — N×M グリッド → 頂点/インデックス生成、法線計算、`glBufferSubData` で z のみ更新。（`SURF-01..06`）
3. **triple buffer** — 大きな Snapshot（格子）向け。（`TRIPLE-01..05`）
4. **ボラ面シーン** — SVI or SABR 風のパラメトリック IV サーフェス（合成）。スライダーで形が変わる。（`VOLSURF-01..04`）
5. **Thomas 法** — 三重対角ソルバ、密行列解と一致。（`TRIDIAG-01..04`）
6. **CN European** — 解析解と一致、格子収束次数 ≈ 2。（`FDM-01..05`）
7. **CN American + PSOR** — American ≥ European、≥ intrinsic、行使境界抽出。（`FDM-06..11`）
8. **FDM シーン** — 後ろ向き反復を 1 ステップ = 1 sim ステップとして `StepOnce` で送れる。V(S,t) 面が満期から現在へ「育つ」。行使境界を面上に線で重ねる。（`FDMSCENE-01..04`）
9. Greeks シーンの Γ(S,T) を 3D 化。

**DoD 追加項目**
- FDM シーンで `Pause → Step` を連打すると 1 反復ずつ面が伸びる
- 200×200 グリッドで 60 fps 維持（`BENCH-04`）

**リスク** GL の環境差（macOS のコアプロファイル）→ ImGui 例示コードの GLSL バージョン切替を踏襲。PSOR の収束が遅い → ω の調整と最大反復、`status` フラグを Snapshot に載せる。

---

### M3 — マイクロストラクチャ（板・Hawkes）✅

**実績（完了時点）** 7 シーン（+ Order book）。テスト 179 ケース（M2 の 147 + 32）、ASan/UBSan/TSan Green、`-Werror`。BENCH-05 は 1e6 混合注文で ≈ 20 M 注文/秒（目標 1 M）。板は独立参照実装との差分テスト 72 万操作で不一致ゼロ、Hawkes は分岐過程シミュレータと KS 検定で不偏性を確認。先行タスクとして bridge の R10（一時停止中の Reset / SetParam を即時 publish）を入れ、Greeks の SetParam を「apply の場で反映」に改定。計画からの逸脱: 取消確率を板の注文数に比例させる（固定比率だとプールが 30 秒で飽和）、板の価格範囲を ±2048 ティックに拡張し壁への到達を `orders_clamped` で可視化、Hawkes の推定 SE は Hessian ではなく実測に合わせて調整。M4 の先行タスクに RUNNER-12（StepOnce は publish_every の位相に関わらず publish）を積んだ。


**タスク**

1. **板データ構造** — 価格レベルは固定ティック配列、レベル内は intrusive doubly-linked list、注文はプールアロケータ。（`LOB-01..08`）
2. **マッチングエンジン** — 指値/成行/取消、部分約定、価格時間優先、自己約定なし。（`LOB-09..14`）
3. **注文フロー生成** — ポアソン → Hawkes。Ogata thinning でシミュレート。（`HAWKES-01..05`）
4. **Hawkes 推定** — 再帰式 O(n) の対数尤度、MLE。（`HAWKES-06..09`）
5. **LOB シーン** — 上位 N レベルを Snapshot に縮約、板深度ラダー、直近約定、λ(t)。「大口注入」ボタン。（`LOBSCENE-01..04`）
6. **ヒートマップ** — 価格 × 時間 × 数量（Bookmap 風）。描画側 History を 2D 化。（`HEAT-01..03`）

**DoD 追加項目** 1e6 注文/秒の合成フローでマッチングが `dropped = 0`（`BENCH-05`）。板の不変条件を毎ステップ検査する Debug ビルドが存在する。

**リスク** intrusive list のバグ → 不変条件チェックを Debug で常時実行。Hawkes の MLE が不安定 → 分岐比の制約付きパラメータ化。

---

### M4 — 動的処理（LSM / 最適執行 / HJB）

**タスク**

1. **LSM** — パス生成（アンチセティック）、後ろ向き回帰（多項式基底 / Laguerre）、価格 + SE。参照解は M2 の CN。（`LSM-01..07`）
2. **LSM シーン** — パス束（縮約: 上位 K 本 + 分位帯）、各時点の継続価値フィット。（`LSMSCENE-01..03`）
3. **Almgren–Chriss** — 線形インパクトの閉形式軌道、リスク回避 λ でのフロンティア。（`AC-01..07`）
4. **執行シーン** — λ 別の執行軌道を重ね描き、フロンティア面（コスト × 分散 × λ）を 3D。M3 の板に実際に流す拡張（オプション）。（`ACSCENE-01..03`）
5. **Merton HJB** — CRRA 効用の HJB を格子で解く。閉形式（定数比率）と一致。（`HJB-01..06`）
6. **HJB シーン** — 価値関数面 V(w,t)、最適比率 vs 富。（`HJBSCENE-01..02`）

**DoD 追加項目** LSM 価格が CN 価格の 95% 信頼区間内（`LSM-02`）。AC の λ=0 が TWAP、λ→大 で前倒し（`AC-03/04`）。

---

### M5 — AAD・性能

**タスク**

1. **テープ** — `Var` 型（演算子オーバーロード）、テープ記録、逆伝播、rewind。（`AAD-01..08`）
2. **BS on AAD** — 価格を `Var` で書き、Δ ν ρ を 1 回の逆伝播で得る。解析解・バンプと一致。（`AAD-09..12`）
3. **AAD シーン** — Greeks（AAD / バンプ / 解析）、計算時間比、テープ長。（`AADSCENE-01..03`）
4. **パフォーマンスパネル** — 全シーンの `step` 時間ヒストグラム、フレーム時間、dropped、ring 占有率。（`PERF-01..04`）
5. **計測ワークフロー** — `perf stat` / `perf record` の手順を `docs/` に追加。SoA 化・アロケータ差し替えのビフォーアフターを記録。

**DoD 追加項目** AAD の全 Greeks が価格計算の ≤ 5 倍の時間（`AAD-12`）。パフォーマンスパネルが全シーンで有効。

---

## 5. 横断タスク

### 5.1 CI（M0 で導入済）

| ジョブ | 内容 | ゲート |
|---|---|---|
| core-tests | 3 OS × Debug/Release、`-Werror`、ctest | 必須 |
| sanitizers | ASan+UBSan 全テスト、TSan（RING / RUNNER） | 必須 |
| viz-build | Ubuntu、FetchContent imgui/implot、viewer ビルドのみ | 必須（実行は手動） |
| bench（M5 で追加） | `[!benchmark]` を実行し結果を artifact に保存 | 情報のみ |

### 5.2 ドキュメント更新規則

* 設計判断を変えたら `01_design.md` の該当節を同じ PR で更新
* 新しいテストは必ず `03_tdd_spec.md` に ID を追加し、状態欄を更新
* 各マイルストーン完了時に `README.md` の「動くもの」を更新

### 5.3 viewer 手動確認チェックリスト（各マイルストーン）

- [ ] 起動直後に全パネルが描画され、`snapshots/s ≈ steps_per_second × speed`
- [ ] σ スライダーを動かすと参照線が段になり、EWMA が先に、Welford が遅れて追従する（Streaming）
- [ ] Pause → Step ×5 で seq が 5 増える。Resume でバーストしない
- [ ] speed 0.01x〜100x で dropped が増えない（増える場合は `publish_every` を上げる）
- [ ] Reset で履歴が消え、パラメータは保持される
- [ ] 60 fps（`frame` ≈ 16.7 ms）を維持

---

## 6. リスク登録簿

| # | リスク | 影響 | 対策 |
|---|---|---|---|
| R1 | 描画ライブラリのバージョン差（ImGui 1.91 と ImPlot 0.16 の互換） | ビルド不可 | vcpkg でピン留め。FetchContent もタグ固定。CI の viz-build で検知 |
| R2 | GCC のネスト `Config` 既定引数バグ（M0 で遭遇） | コンパイルエラー | `Xxx() : Xxx(Config{})` の委譲コンストラクタで回避（規約化） |
| R3 | 統計テストのフレーク | CI 赤 | seed 固定 + 4 SE。フレークしたら許容誤差の根拠を見直す（緩めるのではなく N を増やす） |
| R4 | 3D 描画の環境差（macOS / Wayland） | 起動不能 | ImGui 例示の GLSL 切替を踏襲。ImPlot3D への差し替え口を残す |
| R5 | 高頻度 Snapshot で描画が追いつかない | dropped 増・UI 遅延 | `publish_every`、History サイズ、triple buffer（格子系） |
| R6 | スコープ膨張（シーンを増やしすぎる） | 完成しない | 各 M の DoD を守る。M2 の FDM アニメを「最初の到達点」と定める |
| R7 | 数値の発散（PSOR、GARCH MLE） | 画面が壊れる | Snapshot に `status`。コアは停止せず、描画が警告 |
| R8 | 板の intrusive list のバグ | 不変条件破れ | Debug ビルドで毎ステップ不変条件検査（`LOB-xx` を実行時アサートに転用） |

---

## 7. 見積り（週末 1 日 ≈ 6 時間換算）

| M | 見積り | 主な工数要因 |
|---|---|---|
| M0 | 完了 | — |
| M1 | 4〜5 週末 | 最適化器と GARCH の推定安定化 |
| M2 | 5〜6 週末 | GL 基盤の初期セットアップ、PSOR |
| M3 | 4〜5 週末 | 板データ構造のデバッグ |
| M4 | 4 週末 | LSM の検証、HJB 格子 |
| M5 | 3 週末 | AAD テープ設計 |
