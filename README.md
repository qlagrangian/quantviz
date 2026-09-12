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

## M2（この時点）で動くもの

![M0 viewer](docs/m0_viewer.png)

メニューバーの **Scene** で 6 シーンを切り替える（同時に走るシーンは 1 つ。各シーンは独立した Runner + Panel の対）。3D 面は自前の OpenGL 3.0 レンダラ（FBO → `ImGui::Image`、ドラッグ回転・ホイールズーム・等高線）で描く。

| シーン | コア | 描画 |
|---|---|---|
| Streaming（M0） | `Gbm` + `Welford` + `EwmaVariance` | Spot / Volatility / Control |
| Greeks（M1, 3D は M2） | `black_scholes.hpp`（SIMD ストリップはスカラ版と bit 一致） | Greeks vs K / Γ(S,T) の 3D 面とヒートマップ（タブ） |
| GARCH（M1） | `garch.hpp` + `optim.hpp` | σ_t 真値 vs 推定 / 尤度面 + 最適化軌跡 |
| Kalman pair（M1） | `mat.hpp` + `kalman.hpp` | 2 価格 / β̂ ± 2σ 帯 / スプレッド |
| Vol surface（M2） | `vol_surface.hpp`（SSVI） | IV 面の 3D / 選んだ満期のスマイル / σ_atm ρ η γ |
| FDM American（M2） | `tridiag.hpp` + `fdm_cn.hpp`（Crank–Nicolson + PSOR、Rannacher 起動） | V(S,t) が満期から今へ育つ 3D 面（Pause → Step で 1 反復ずつ）/ V(S) と本源的価値・European 参照 / 行使境界 S*(t) |

- `core/`   : `Gbm`, `Welford`, `EwmaVariance`, `Rng`, `black_scholes`, `optim`, `garch`, `Mat`, `Kalman`, `tridiag`, `FdmCn`, `vol_surface`
- `bridge/` : `SpscRing`, `TripleBuffer`（最新 1 枚）, `Command`, `Model` / `SurfaceModel` concept, `SimClock`, `Runner`（面チャネル付き）
- `scenes/` : `StreamingModel`, `GreeksModel`, `GarchModel`, `KalmanPairModel`, `VolSurfaceModel`, `FdmAmericanModel`
- `viz/`    : `SceneRegistry` / `RunnerScene`, 共通の時計 UI・`RateMeter`、`gl/`（行列・軌道カメラ・サーフェスメッシュ）、`src/gl/`（GL ローダ・レンダラ・`SurfaceView`）、6 パネル
- `tests/`  : Catch2 v3, 147 テストケース + ベンチマーク 4 本（BENCH-01..04）

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
core/    include/quantviz/core/{rng, models/gbm, stats/{welford,ewma,optim,garch,kalman}, math/{mat,tridiag}, pricing/{black_scholes,fdm_cn,vol_surface}}.hpp   外部依存ゼロ
bridge/  include/quantviz/bridge/{spsc_ring,triple_buffer,command,model_concept,sim_clock,runner}.hpp                              std のみ
scenes/  include/quantviz/scenes/{streaming,greeks,garch,kalman_pair,vol_surface,fdm_american}_model.hpp                          core + bridge
viz/     include/quantviz/viz/{history,rate_meter,clock_controls,scene_registry,gl/{math,camera,surface_mesh}}.hpp（GUI 非依存）
         src/{main.cpp, gl/{gl_loader,surface_renderer,surface_view}, panels/{clock_panel,panel_common}.hpp, panels/*_panel.*}  ImGui/ImPlot/GLFW/GL
tests/   Catch2 v3（test_*.cpp = 仕様 ID と 1:1）
docs/    設計書・実装計画書・TDD 仕様（docs/superpowers/plans/ にマイルストーンごとの実行計画）
```

## 依存規則（違反はレビューで差し戻し）

`core` → なし　|　`bridge` → std　|　`scenes` → core, bridge　|　`vizcore` → scenes（GUI なし）　|　`viz` → 全部 + ImGui/ImPlot/GLFW

描画層が `Model` を直接参照すること、コア層が描画ライブラリを include することは禁止。
