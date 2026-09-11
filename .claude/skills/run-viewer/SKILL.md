---
name: run-viewer
description: Build, launch, drive, and screenshot the quantviz ImGui/ImPlot viewer on this WSL2/WSLg machine. Use when asked to run the viewer, check a scene visually, or walk the manual viewer checklist in docs/02_implementation_plan.md §5.3.
---

The viewer (`quantviz_viz`) is a GLFW + OpenGL window. On WSL2 with WSLg a real display exists
(`DISPLAY=:0`), so the window appears on the Windows desktop; this skill drives it with XTest and
captures it with ffmpeg so an agent can see it too.

## Prerequisites (one-time, needs sudo)

```bash
sudo apt install libglfw3-dev libgl-dev ninja-build   # ffmpeg, python3, libX11/libXtst runtime are already present
```

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DQUANTVIZ_WARNINGS_AS_ERRORS=ON
cmake --build build -j          # imgui/implot come from FetchContent on first configure
```

## Run + drive (agent path)

```bash
D=.claude/skills/run-viewer/drive.py
python3 $D launch ./build/viz/quantviz_viz     # forces GLFW onto X11, waits for the window, prints wid
python3 $D shot /tmp/shots/01.png              # capture the viewer window; then Read the PNG
python3 $D click 838 201                       # window-relative coordinates (read them off a screenshot)
python3 $D click 946 201 5                     # click 5 times (e.g. Step)
python3 $D drag 800 92 900 92                  # drag a slider
python3 $D quit
```

Telemetry to read off the Control window for the §5.3 checklist: `seq`, `snapshots/s`, `dropped`, `frame`.
Pause → `Step` ×5 must raise `seq` by exactly 5; Resume must not burst.

## Gotchas

- **Root capture is black.** WSLg's XWayland is rootless; always capture by window id (the script does).
- **First click only focuses.** Click a neutral spot on the black background outside every ImGui window before
  the click that matters (e.g. `click 1210 710` when the layout leaves that corner empty).
- **ImGui windows drag from empty areas.** A "neutral" click inside an ImGui window moves it and shifts all
  button coordinates. Re-screenshot after any accidental drag and re-read coordinates.
- **GLFW 3.4 prefers Wayland.** The launcher unsets `WAYLAND_DISPLAY`/`XDG_RUNTIME_DIR` so the window is an X11
  window that XTest and x11grab can reach. A plain `./build/viz/quantviz_viz` from a shell uses Wayland instead.
- **imgui.ini.** The viewer writes its layout to `imgui.ini` in the cwd; the launcher runs from `/tmp`.
