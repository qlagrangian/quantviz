# quantviz — C++ 金融工学ラボ（コア ⇄ リアルタイム描画）

金融工学の理論を **C++20 のコア**で実装し、その状態を **ImGui / ImPlot（M2 以降 OpenGL 3D）でリアルタイム描画**する個人研究プロジェクト。
コアと描画は「Snapshot（コア→描画）」「Command（描画→コア）」の 2 本のロックフリー・リングだけで結ばれる（**双対構造**）。

```
   描画スレッド                             計算スレッド
 ┌──────────────┐   Snapshot ring (SPSC)   ┌──────────────┐
 │ ImPlot 2D    │ ◀──────────────────────  │ Model::step  │
 │ OpenGL 3D    │                          │ Model::snap  │
 │ Control      │ ──────────────────────▶  │ Model::apply │
 └──────────────┘   Command ring  (SPSC)   └──────────────┘
```

## ドキュメント

| 文書 | 内容 |
|---|---|
| [docs/01_design.md](docs/01_design.md) | 設計書 — 原則・アーキテクチャ・型契約・時間モデル・モジュール設計・拡張手順 |
| [docs/02_implementation_plan.md](docs/02_implementation_plan.md) | 実装計画書 — M0〜M5 のスコープ・タスク・完了条件・リスク |
| [docs/03_tdd_spec.md](docs/03_tdd_spec.md) | TDD 仕様網羅 — 全モジュールの仕様を ID 付きテストケースとして列挙（M0 は実装済） |

## M4（この時点）で動くもの

![M0 viewer](docs/m0_viewer.png)

メニューバーの **Scene** で 10 シーンを切り替える（同時に走るシーンは 1 つ）。3D 面は自前の OpenGL 3.0 レンダラ（FBO → `ImGui::Image`）。一時停止中でもスライダーと Reset は即座に画面に反映される（Runner が適用直後に Snapshot を再送）。

| シーン | コア | 描画 |
|---|---|---|
| Streaming（M0） | `Gbm` + `Welford` + `EwmaVariance` | Spot / Volatility / Control |
| Greeks（M1, 3D は M2） | `black_scholes.hpp`（SIMD ストリップはスカラ版と bit 一致） | Greeks vs K / Γ(S,T) の 3D 面とヒートマップ |
| GARCH（M1） | `garch.hpp` + `optim.hpp` | σ_t 真値 vs 推定 / 尤度面 + 最適化軌跡 |
| Kalman pair（M1） | `mat.hpp` + `kalman.hpp` | 2 価格 / β̂ ± 2σ 帯 / スプレッド |
| Vol surface（M2） | `vol_surface.hpp`（SSVI） | IV 面の 3D / スマイル断面 |
| FDM American（M2） | `tridiag.hpp` + `fdm_cn.hpp`（CN + PSOR、Rannacher） | V(S,t) が満期から今へ育つ 3D 面（Pause → Step で 1 反復）/ V(S) / 行使境界 |
| Order book（M3） | `micro/order_book.hpp` + `matching_engine.hpp`（固定容量、intrusive list、≈20 M 注文/秒）+ `hawkes.hpp` | 板深度ラダー / 価格×時間ヒートマップ / 約定と λ(t) / 「Inject buy/sell」で板が崩れて回復する |
| LSM American（M4） | `pricing/lsm.hpp` + `math/linsolve.hpp`（Longstaff–Schwartz、アンチセティック、ランク打ち切り付き正規方程式） | 16 本のパス + 分位帯と現在時点の縦線 / 継続価値フィット vs 本源的価値と行使境界（Pause → Step で 1 行使時点ずつ回帰が進む）/ 価格 ± SE が European から American へせり上がる |
| Optimal execution（M4） | `exec/almgren_chriss.hpp`（閉形式軌道・コスト・フロンティア、溢れなし sinh 比） | λ 別の執行軌道 + 再生ヘッド / 効率フロンティア上を動くマーカー / 残量 x(t, λ) の 3D 面 |
| Merton HJB（M4） | `exec/hjb_merton.hpp`（対数富裕度の陰的 Euler、M 行列、制約付き閉形式境界） | V(w, t) が満期から育つ 3D 面 / V(w) 数値 vs 閉形式 / π*(w) が定数に乗る（誤差 1e-5） |

- `core/`   : `Gbm`, `Welford`, `EwmaVariance`, `Rng`, `black_scholes`, `optim`, `garch`, `Mat`, `Kalman`, `tridiag`, `FdmCn`, `vol_surface`, `OrderBook`, `MatchingEngine`, `Hawkes`, `linsolve`, `Lsm`, `almgren_chriss`, `HjbMerton`
- `bridge/` : `SpscRing`, `TripleBuffer`, `Command`, `Model` / `SurfaceModel` concept, `SimClock`, `Runner`（面チャネル、R10 再送）
- `scenes/` : 10 の `Model`（すべて POD Snapshot、4 つは SurfaceModel）
- `viz/`    : `SceneRegistry` / `RunnerScene`, 共通の時計 UI・`RateMeter`・`History` / `History2D`、`gl/`（行列・カメラ・メッシュ）、`src/gl/`（GL ローダ・レンダラ・`SurfaceView`）、10 パネル
- `tests/`  : Catch2 v3, 207 テストケース + ベンチマーク 5 本（BENCH-01..05）

## ビルド

### A. vcpkg（推奨・全プラットフォーム）
```bash
export VCPKG_ROOT=/path/to/vcpkg
cmake --preset dev          # 依存は vcpkg.json から自動解決
cmake --build --preset dev
ctest --preset dev
./build/dev/viz/quantviz_viz
```

### B. vcpkg 無し（Linux）
GLFW をシステムから、ImGui / ImPlot を FetchContent で取得する。
```bash
sudo apt install libglfw3-dev libgl-dev ninja-build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/tests/quantviz_tests
./build/viz/quantviz_viz
```

### C. コア + テストのみ（GUI 依存ゼロ、CI 用）
```bash
cmake --preset core-only && cmake --build --preset core-only && ctest --preset core-only
```

ベンチマーク: `./build/.../tests/quantviz_tests "[!benchmark]"`

## ディレクトリ

```
core/    include/quantviz/core/{rng, models/{gbm,hawkes}, stats/{welford,ewma,optim,garch,kalman}, math/{mat,tridiag}, pricing/{black_scholes,fdm_cn,vol_surface}, micro/{order_book,matching_engine}}.hpp   外部依存ゼロ
bridge/  include/quantviz/bridge/{spsc_ring,triple_buffer,command,model_concept,sim_clock,runner}.hpp                              std のみ
scenes/  include/quantviz/scenes/{streaming,greeks,garch,kalman_pair,vol_surface,fdm_american,lob}_model.hpp                      core + bridge
viz/     include/quantviz/viz/{history,history2d,rate_meter,clock_controls,scene_registry,gl/{math,camera,surface_mesh}}.hpp（GUI 非依存）
         src/{main.cpp, gl/{gl_loader,surface_renderer,surface_view}, panels/{clock_panel,panel_common}.hpp, panels/*_panel.*}  ImGui/ImPlot/GLFW/GL
tests/   Catch2 v3（test_*.cpp = 仕様 ID と 1:1）
docs/    設計書・実装計画書・TDD 仕様（docs/superpowers/plans/ にマイルストーンごとの実行計画）
```

## 依存規則（違反はレビューで差し戻し）

`core` → なし　|　`bridge` → std　|　`scenes` → core, bridge　|　`vizcore` → scenes（GUI なし）　|　`viz` → 全部 + ImGui/ImPlot/GLFW

描画層が `Model` を直接参照すること、コア層が描画ライブラリを include することは禁止。
