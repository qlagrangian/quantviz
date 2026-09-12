#!/usr/bin/env python3
"""XTest / ffmpeg driver for the quantviz viewer on WSLg (or any X11 display).

Usage (all window coordinates are client-area pixels of the viewer window):
  drive.py launch [path-to-quantviz_viz]   start the viewer with X11 forced, wait for its window
  drive.py find                            print the viewer window id and geometry
  drive.py shot <name.png>                 capture the viewer window (ffmpeg x11grab -window_id)
  drive.py click <x> <y> [n] [delay]       left-click n times (default 1) at window coords
  drive.py move <x> <y>                    move the pointer (hover)
  drive.py drag <x0> <y0> <x1> <y1>        press, move, release (e.g. to drag a slider)
  drive.py key <keysym> [n]                press a key (X keysym name, e.g. Return, Escape, a)
  drive.py wheel <x> <y> [clicks]          mouse wheel at window coords (+up / -down)
  drive.py sleep <seconds>
  drive.py quit                            kill the viewer started by `launch` (pid file), else every viewer

Environment: QV_WID=0x<id> pins the window all commands act on (several agents, several viewers);
QV_PID_FILE overrides the pid file used by launch/quit (default /tmp/quantviz_viz.pid).

Gotchas learned on WSLg: the X root window is black (rootless XWayland) so capture the
window id, never :0; the first click on an unfocused window only focuses it — click a
neutral spot inside the window first; ImGui windows drag when you press on their empty
area, so keep "neutral" clicks on the black background outside any ImGui window.
"""
import ctypes, ctypes.util, os, subprocess, sys, time

_X = ctypes.CDLL(ctypes.util.find_library("X11"))
_T = ctypes.CDLL(ctypes.util.find_library("Xtst"))
_X.XOpenDisplay.restype = ctypes.c_void_p; _X.XOpenDisplay.argtypes = [ctypes.c_char_p]
_X.XDefaultRootWindow.restype = ctypes.c_ulong; _X.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
_X.XQueryTree.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
                          ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)), ctypes.POINTER(ctypes.c_uint)]
_X.XFree.argtypes = [ctypes.c_void_p]
_X.XFlush.argtypes = [ctypes.c_void_p]
_X.XInternAtom.restype = ctypes.c_ulong; _X.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
_X.XGetWindowProperty.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_long, ctypes.c_long, ctypes.c_int,
                                  ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_char_p)]
_X.XTranslateCoordinates.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong)]
_X.XStringToKeysym.restype = ctypes.c_ulong; _X.XStringToKeysym.argtypes = [ctypes.c_char_p]
_X.XKeysymToKeycode.restype = ctypes.c_ubyte; _X.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
_T.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
_T.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
_T.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]

class _Attrs(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int), ("width", ctypes.c_int), ("height", ctypes.c_int),
                ("border_width", ctypes.c_int), ("depth", ctypes.c_int), ("visual", ctypes.c_void_p),
                ("root", ctypes.c_ulong), ("cls", ctypes.c_int), ("bit_gravity", ctypes.c_int),
                ("win_gravity", ctypes.c_int), ("backing_store", ctypes.c_int), ("backing_planes", ctypes.c_ulong),
                ("backing_pixel", ctypes.c_ulong), ("save_under", ctypes.c_int), ("colormap", ctypes.c_ulong),
                ("map_installed", ctypes.c_int), ("map_state", ctypes.c_int), ("all_event_masks", ctypes.c_long),
                ("your_event_mask", ctypes.c_long), ("do_not_propagate_mask", ctypes.c_long),
                ("override_redirect", ctypes.c_int), ("screen", ctypes.c_void_p)]
_X.XGetWindowAttributes.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(_Attrs)]

os.environ.pop("WAYLAND_DISPLAY", None)
os.environ.setdefault("DISPLAY", ":0")
_d = _X.XOpenDisplay(None)
if not _d:
    sys.exit("XOpenDisplay failed (DISPLAY=%s)" % os.environ.get("DISPLAY"))
_root = _X.XDefaultRootWindow(_d)

def _prop_string(w, name):
    atom = _X.XInternAtom(_d, name.encode(), 1)
    if not atom: return ""
    t = ctypes.c_ulong(); f = ctypes.c_int(); n = ctypes.c_ulong(); b = ctypes.c_ulong(); data = ctypes.c_char_p()
    if _X.XGetWindowProperty(_d, w, atom, 0, 1024, 0, 0, ctypes.byref(t), ctypes.byref(f), ctypes.byref(n), ctypes.byref(b), ctypes.byref(data)) != 0:
        return ""
    s = ctypes.string_at(data, n.value).decode(errors="replace") if data and n.value else ""
    if data: _X.XFree(data)
    return s

def _walk(w, out):
    r = ctypes.c_ulong(); p = ctypes.c_ulong(); kids = ctypes.POINTER(ctypes.c_ulong)(); n = ctypes.c_uint()
    if not _X.XQueryTree(_d, w, ctypes.byref(r), ctypes.byref(p), ctypes.byref(kids), ctypes.byref(n)): return
    for i in range(n.value):
        c = kids[i]; a = _Attrs(); _X.XGetWindowAttributes(_d, c, ctypes.byref(a))
        if a.map_state == 2:
            out.append((c, a.width, a.height, _prop_string(c, "_NET_WM_NAME") or _prop_string(c, "WM_NAME")))
        _walk(c, out)
    if n.value: _X.XFree(kids)

PID_FILE = os.environ.get("QV_PID_FILE", "/tmp/quantviz_viz.pid")

def _all_named(needle="quantviz"):
    wins = []; _walk(_root, wins)
    return [(w, width, height, name) for w, width, height, name in wins if needle in name]

def find(needle="quantviz"):
    wins = _all_named(needle)
    return wins[0] if wins else None

def origin(w):
    rx, ry, ch = ctypes.c_int(), ctypes.c_int(), ctypes.c_ulong()
    _X.XTranslateCoordinates(_d, w, _root, 0, 0, ctypes.byref(rx), ctypes.byref(ry), ctypes.byref(ch))
    return rx.value, ry.value

def _need():
    wid = os.environ.get("QV_WID")           # pin a specific window when several viewers are running
    if wid:
        w = int(wid, 16); a = _Attrs(); _X.XGetWindowAttributes(_d, w, ctypes.byref(a))
        return (w, a.width, a.height, "<QV_WID>")
    f = find()
    if not f: sys.exit("viewer window not found (is it running? try: drive.py launch)")
    return f

def move(wx, wy):
    w = _need()[0]; ox, oy = origin(w)
    _T.XTestFakeMotionEvent(_d, -1, ox + wx, oy + wy, 0); _X.XFlush(_d); time.sleep(0.15)

def click(wx, wy, n=1, delay=0.3):
    for _ in range(n):
        move(wx, wy)
        _T.XTestFakeButtonEvent(_d, 1, 1, 0); _X.XFlush(_d); time.sleep(0.1)
        _T.XTestFakeButtonEvent(_d, 1, 0, 0); _X.XFlush(_d); time.sleep(delay)

def drag(x0, y0, x1, y1, steps=10):
    move(x0, y0)
    _T.XTestFakeButtonEvent(_d, 1, 1, 0); _X.XFlush(_d); time.sleep(0.1)
    w = _need()[0]; ox, oy = origin(w)
    for i in range(1, steps + 1):
        x = x0 + (x1 - x0) * i // steps; y = y0 + (y1 - y0) * i // steps
        _T.XTestFakeMotionEvent(_d, -1, ox + x, oy + y, 0); _X.XFlush(_d); time.sleep(0.03)
    _T.XTestFakeButtonEvent(_d, 1, 0, 0); _X.XFlush(_d); time.sleep(0.3)

def wheel(wx, wy, clicks=1):
    """Scroll at window coords: positive clicks = up (X button 4), negative = down (button 5)."""
    move(wx, wy)
    button = 4 if clicks > 0 else 5
    for _ in range(abs(int(clicks))):
        _T.XTestFakeButtonEvent(_d, button, 1, 0); _X.XFlush(_d); time.sleep(0.03)
        _T.XTestFakeButtonEvent(_d, button, 0, 0); _X.XFlush(_d); time.sleep(0.08)

def key(name, n=1):
    kc = _X.XKeysymToKeycode(_d, _X.XStringToKeysym(name.encode()))
    for _ in range(n):
        _T.XTestFakeKeyEvent(_d, kc, 1, 0); _X.XFlush(_d); time.sleep(0.05)
        _T.XTestFakeKeyEvent(_d, kc, 0, 0); _X.XFlush(_d); time.sleep(0.15)

def shot(path):
    w = _need()[0]
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "x11grab", "-window_id", hex(w),
                    "-i", os.environ["DISPLAY"], "-frames:v", "1", "-y", path], check=True)
    print(path)

def launch(binary):
    env = dict(os.environ); env.pop("WAYLAND_DISPLAY", None); env.pop("XDG_RUNTIME_DIR", None)  # force GLFW onto X11
    log = open("/tmp/quantviz_viz.log", "w")
    before = {w for w, *_ in _all_named()}
    p = subprocess.Popen([os.path.abspath(binary)], env=env, stdout=log, stderr=log, cwd="/tmp")  # cwd=/tmp: imgui.ini goes there
    for _ in range(100):
        time.sleep(0.1)
        new = [f for f in _all_named() if f[0] not in before]   # only windows that appeared after our launch
        if new:
            f = new[0]; time.sleep(0.5)
            open(PID_FILE, "w").write("%d 0x%x\n" % (p.pid, f[0]))
            print("pid=%d wid=0x%x %dx%d %r   (export QV_WID=0x%x to pin this window)" % (p.pid, f[0], f[1], f[2], f[3], f[0]))
            return
    sys.exit("viewer window did not appear; see /tmp/quantviz_viz.log")

def main(argv):
    if not argv: sys.exit(__doc__)
    cmd, a = argv[0], argv[1:]
    if cmd == "launch": launch(a[0] if a else "./build/viz/quantviz_viz")
    elif cmd == "find":
        f = find(); print("wid=0x%x %dx%d origin=%s name=%r" % (f[0], f[1], f[2], origin(f[0]), f[3]) if f else "not found")
    elif cmd == "shot": shot(a[0])
    elif cmd == "click": click(int(a[0]), int(a[1]), int(a[2]) if len(a) > 2 else 1, float(a[3]) if len(a) > 3 else 0.3)
    elif cmd == "move": move(int(a[0]), int(a[1]))
    elif cmd == "drag": drag(int(a[0]), int(a[1]), int(a[2]), int(a[3]))
    elif cmd == "key": key(a[0], int(a[1]) if len(a) > 1 else 1)
    elif cmd == "wheel": wheel(int(a[0]), int(a[1]), int(a[2]) if len(a) > 2 else 1)
    elif cmd == "sleep": time.sleep(float(a[0]))
    elif cmd == "quit":
        try:
            pid = int(open(PID_FILE).read().split()[0]); os.kill(pid, 15); os.remove(PID_FILE); print("killed pid", pid)
        except (OSError, ValueError, IndexError):
            subprocess.run(["pkill", "-x", "quantviz_viz"])   # fallback: no pid file → kill every viewer
    else: sys.exit(__doc__)

if __name__ == "__main__":
    main(sys.argv[1:])
