# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

quantviz is a C++20 financial-engineering lab: numerical models run on a compute thread and are drawn in real time by an ImGui/ImPlot viewer. Docs and code comments are written in Japanese; identifiers and test names are English. The three documents in `docs/` are authoritative and must be updated in the same PR as any code change they describe:

- `docs/01_design.md` — principles, layer rules, type contracts, thread/time model, how to add a scene
- `docs/02_implementation_plan.md` — milestones M0–M5, Definition of Done, branch/commit rules, manual viewer checklist
- `docs/03_tdd_spec.md` — every test case as a spec ID; **add the ID row there before writing the test**

M0–M4 are complete (10 scenes: Streaming, Greeks, GARCH, Kalman pair, Vol surface, FDM American, Order book, LSM American, Optimal execution, Merton HJB). M5 (tape AAD, performance panel) is planned in `02_implementation_plan.md`; per-milestone execution plans with type contracts live in `docs/superpowers/plans/`. The module tables in `01_design.md` §6 list files and APIs.

## Build and test

CMake ≥ 3.25, C++20 (GCC ≥ 12 / Clang ≥ 15 / MSVC ≥ 19.34). The presets use the Ninja generator; `dev`/`release` additionally need vcpkg via `$VCPKG_ROOT`.

```bash
# Full (viewer + tests) with vcpkg
export VCPKG_ROOT=/path/to/vcpkg
cmake --preset dev && cmake --build --preset dev && ctest --preset dev
./build/dev/viz/quantviz_viz

# Core + tests only: no GUI deps, no vcpkg (this is what CI runs). Catch2 comes via FetchContent.
cmake --preset core-only && cmake --build --preset core-only && ctest --preset core-only

# Same thing without Ninja (only cmake + a compiler needed)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQUANTVIZ_BUILD_VIZ=OFF -DQUANTVIZ_WARNINGS_AS_ERRORS=ON
cmake --build build -j && ctest --test-dir build --output-on-failure

# Viewer without vcpkg (Linux): system GLFW, imgui/implot via FetchContent
sudo apt install libglfw3-dev libgl-dev ninja-build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ./build/viz/quantviz_viz
```

Build with `-DQUANTVIZ_WARNINGS_AS_ERRORS=ON` before finishing any change; CI uses `-Werror` on all three compilers with `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wold-style-cast`. Third-party code is excluded via `SYSTEM` includes.

### Running tests

One test binary, `quantviz_tests` (in `build/<preset>/tests/` or `build/tests/`). Every `TEST_CASE` is also registered with ctest under its full name.

```bash
./build/tests/quantviz_tests "RING-07*"          # one spec ID (Catch2 wildcard on the name)
./build/tests/quantviz_tests "[clock]"           # one module by tag
./build/tests/quantviz_tests "[concurrency]"     # one kind by tag
ctest --test-dir build -R "RUNNER" --output-on-failure
./build/tests/quantviz_tests "[!benchmark]"      # benchmarks; skipped by default
```

Sanitizers, as CI runs them (ASan+UBSan on everything, TSan on the RING/RUNNER tests):

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DQUANTVIZ_BUILD_VIZ=OFF \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DQUANTVIZ_BUILD_VIZ=OFF \
      -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer"
cmake --build build-tsan -j && ctest --test-dir build-tsan -R "RING|RUNNER" --output-on-failure
```

Formatting is `.clang-format` (Google base, 4-space indent, 110 columns, `T* p`, aligned consecutive assignments). The viewer has no automated tests; use the manual checklist in `02_implementation_plan.md` §5.3.

## Architecture

### Layers — dependencies point down only

| CMake target | Directory | May depend on |
|---|---|---|
| `quantviz::core` | `core/` | nothing but std. Numerics: `Rng`, `Gbm`, `Welford`, `EwmaVariance` (later pricing/micro/exec/aad) |
| `quantviz::bridge` | `bridge/` | std (`<atomic>`, `<thread>`, `<stop_token>`). `SpscRing`, `Command`, `Model` concept, `SimClock`, `Runner` |
| `quantviz::scenes` | `scenes/` | core, bridge. One `XxxModel` + `XxxSnapshot` + `Param` enum per scene |
| `quantviz::vizcore` | `viz/include/` | scenes. GUI-free viewer helpers (`History<N>`), unit-testable |
| `quantviz_viz` | `viz/src/` | everything + ImGui/ImPlot/GLFW/OpenGL. **The only target allowed to include GUI headers** |

Everything except `quantviz_viz` is header-only and exposed as an INTERFACE library, so include paths only arrive through `target_link_libraries`; an illegal include simply fails to compile. Never let a panel reference a `Model` directly, and never let core/bridge/scenes include a GUI header.

### The core ⇄ viewer duality

Core and viewer talk only through two SPSC lock-free rings owned by `bridge::Runner<M>`:

- **Snapshot ring (core → viz).** `Model::snapshot()` yields a fixed-size POD for the *current* state only. The viewer polls the ring empty at the start of each frame and appends to `viz::History<N>` circular buffers; building time series is the viewer's job, so a snapshot stays one cache line or two.
- **Command ring (viz → core).** Widgets produce `bridge::Command`. Clock commands (Pause/Resume/StepOnce/SetSpeed) are applied by `Runner` to `SimClock`; model commands (SetParam/Reset) are forwarded to `Model::apply`. Commands are drained **before** stepping in each tick, so a change takes effect from the next step; if a tick applies a model command but runs no steps (paused), the Runner re-publishes the snapshot once so the change is visible immediately (R10).

Panels are pure functions of the latest snapshot plus their own UI state. Thread count is exactly two: the compute thread (a `std::jthread` inside `Runner`) owns the `Model` and the `SimClock`.

### Contracts to preserve

- **`Model` concept** (`bridge/model_concept.hpp`): `step(dt)`, `snapshot() const` returning a trivially-copyable, default-constructible `Snapshot`, and `apply(const Command&)`. Every model ends with `static_assert(bridge::Model<XxxModel>)`.
- **Snapshot rules**: fixed-size POD, no pointers/`std::vector`/`std::string` (use `std::array`), target ≤ 128 bytes, must carry `seq` (0 = never stepped or just reset) and any true parameter values the panel needs for reference lines. Large state is reduced before it goes in. Grid-sized state (surfaces) goes through the optional **`SurfaceModel`** extension instead: `Surface` POD + `surface(Surface&) const noexcept` writing every field; `Runner` then owns a `bridge::TripleBuffer<Surface>` (latest one wins, writer never blocks) read with `poll_surface()`. Existing exceptions above 128 B: Greeks (16.5 KB), GARCH (9.8 KB), LSM (12.2 KB), HJB (10.3 KB), FDM (6.3 KB), Exec (5.7 KB), LOB (2.5 KB) — with `SnapCap` reduced accordingly (64 for the M2–M4 scenes).
- **3D**: `viz/include/quantviz/viz/gl/` (vizcore: matrices, orbit camera, `SurfaceMesh`) is GUI-free; only `viz/src/gl/` and `main.cpp` touch OpenGL (functions hand-loaded via `glfwGetProcAddress`, no glad). Panels own a `SurfaceMesh` + `SurfaceRenderer` + `SurfaceView`, upload only when `poll_surface` returns a new frame, and keep the dirty flag sticky until `SurfaceView::draw()` returns true. Degrade to text when `glapi::gl_available()` is false.
- **`SimClock`** is pure logic with no time source; `Runner` passes wall-clock seconds in. Paused time is not banked (no burst on resume), `request_step()` allows one step even while paused, `max_steps_per_tick` discards excess instead of catching up, and `speed ≤ 0` or NaN clamps to 0.
- **`Runner` guarantees** (`01_design.md` §4.4): `send`/`poll` never block; a full snapshot ring drops and counts (`dropped_snapshots()`) rather than stalling the core; every `send` that returned true is applied before `stop()` returns; `start()` is idempotent; the destructor joins. `tick(elapsed)` runs one loop synchronously for tests and must not be called while the thread is running; `model()`/`clock()` are only safe to read then too.
- **`SpscRing<T, Capacity>`**: `T` trivially copyable, `Capacity` a power of two, head/tail/caches on separate cache lines (layout is asserted by RING-08).
- **Determinism**: all randomness goes through `core::Rng`; same seed + same command sequence gives bit-identical results. `Reset` rewinds path and statistics but keeps mu/sigma/lambda; `Command::reset(0)` means replay with the current seed.
- **Hot paths** (`step`, `snapshot`, rings) are zero-allocation with no mutex and no exceptions. Bad UI input is clamped in `noexcept` core setters; unknown `param_id` is ignored.

### Adding a scene (`01_design.md` §11)

1. Pure numerics in `core/`, tests first (IDs added to `03_tdd_spec.md`).
2. `scenes/<scene>_model.hpp`: `XxxSnapshot` (POD with `seq`), `XxxModel` (`step`/`snapshot`/`apply`, a `Param` enum with `k`-prefixed values, 0 reserved), `static_assert(bridge::Model<...>)`.
3. A contract test modeled on `STREAM-01`: concept satisfied, POD, size bound, dual-run determinism against the bare core component.
4. `viz/src/panels/<scene>_panel.{hpp,cpp}` exposing `void draw(Runner<XxxModel>&)`; reuse `History` and the clock controls.
5. Register in `viz/src/main.cpp` (a `scene_registry` and shared `common_controls` are M1 work).

## Development conventions

- **Spec ID = test.** IDs are `<MODULE>-<NN>` (`RING-07`, `STREAM-03`), one per `TEST_CASE`, with `tests/test_<module>.cpp` matching the module 1:1. Names are `"<ID>: <spec sentence>"` and tags are `[module][kind]` where kind is one of `unit`, `numeric`, `property`, `statistical`, `concurrency`, `contract`, `determinism`. Benchmarks live in `tests/bench_<module>.cpp` tagged `[!benchmark][module]`. Flow: add the row in `03_tdd_spec.md` → Red → Green → Refactor → flip the status column to ✅. New test files must be added to `tests/CMakeLists.txt`.
- **Statistical tests** use a fixed seed, tolerance = 4 SE, and state the expected value, SE, and multiplier in a comment. Tighten by raising N, never by loosening the tolerance. The tolerance guide is `03_tdd_spec.md` §2.4.
- **Concurrency tests**: pin behaviour with the synchronous `tick()` API first; keep threaded tests to one or two per module and run them under TSan.
- **Commits** start with the spec ID (`RING-07: producer/consumer 1M items stress test`). Milestone work goes on a branch such as `m1-greeks` → PR → squash; `main` stays green.
- **Naming**: `quantviz::<layer>` namespaces, PascalCase types, snake_case functions, trailing `_` members, `k` prefix for constants and `Param` enumerators. Every header starts with `#pragma once` and a one-paragraph responsibility comment.
- **GCC workaround**: a nested `Config` struct with default member initializers used as a default argument trips a GCC bug. Write a delegating constructor `Xxx() : Xxx(Config{})` instead (see `SimClock`, `StreamingModel`).
- **UI → Command**: ImGui sliders are `float`; widen to `double` when building the `Command`. Send only on the frame the widget changed; a failed `send` (full ring) can be ignored. Reset sends `Command::reset()` and clears the viewer's `History` in the same frame. Panels drop stale snapshots with a strict rewind guard (`seq < prev_seq_` → clear) and never push a same-seq re-publish into a History. Any slider value that indexes an array needs `ImGuiSliderFlags_AlwaysClamp` and a `std::clamp` after the widget (Ctrl+click text entry bypasses slider bounds). ImPlot 0.16's `PlotShaded` crashes on an empty History — guard with `count() > 1`.
- **Running the viewer**: use the project skill `run-viewer` (`.claude/skills/run-viewer/drive.py`: launch, screenshot by window id, click/drag/wheel/key). Pin your window with `QV_WID` / `QV_PID_FILE` when several viewers share the display, and set `QV_SEND=1` so clicks go to that window by id (XTest clicks land on whichever viewer is topmost). Every scene must pass the manual checklist (snapshots/s ≈ steps_per_second, dropped 0, Pause → Step ×5 = seq +5, Reset clears plots, paused sliders update on the next frame).
- **Parallel work**: tasks run in isolated `git worktree`s with per-task build directories; shared files (`main.cpp`, CMake lists, `docs/*`, `README.md`, CI) are touched only by the integrator. Test names must be ASCII and must not contain `[` or `]` (ctest passes them on the Windows command line, and CatchAddTests treats an unmatched `[` as CMake list nesting so every case after it is swallowed into one bogus entry; CI forbids brackets outright).
- **Definition of Done** for a milestone task: all tests green, `-Werror` build, ASan/UBSan green (TSan too if `bridge/` was touched), no layer-rule violations, benchmarks not regressed more than 2×, manual viewer checklist passed, docs updated.
