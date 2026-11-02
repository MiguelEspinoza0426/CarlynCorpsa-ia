"""Uses a Windows desktop window (e.g. the live view of Hik-Connect or iVMS-4200) as if it were a camera.

Source: "window:<part of the title>" or "window:<part of the title>:x,y,w,h".
  - The region is relative to the inner area of the window (without borders or title bar), so it
    still works if the window moves. Without a region the whole inner area is captured.
  - The window must be visible on the desktop of the node (not minimized, not covered, session unlocked):
    what is on screen is captured.
"""
import ctypes
import logging
import re
import sys
import threading
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)

CAPTURE_FPS = 15
SUPPORTED = sys.platform == "win32"

if SUPPORTED:
    from ctypes import wintypes

    import mss

    user32 = ctypes.windll.user32
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # real coordinates with Windows display scaling
    except (AttributeError, OSError):
        user32.SetProcessDPIAware()


def parse(spec: str) -> tuple[str, tuple[int, int, int, int] | None]:
    """'iVMS-4200:10,20,640,360' -> ('iVMS-4200', (10, 20, 640, 360)); without a region -> (title, None)."""
    match = re.fullmatch(r"(.+):(\d+),(\d+),(\d+),(\d+)", spec)
    if match:
        return match.group(1), tuple(int(v) for v in match.groups()[1:])
    return spec, None


def _visible_windows() -> list[tuple[int, str]]:
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _):
        if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buffer, length + 1)
                found.append((hwnd, buffer.value))
        return True

    user32.EnumWindows(callback, 0)
    return found


def _find_window(title: str) -> int | None:
    """First visible window whose title contains `title` (case-insensitive)."""
    return next((hwnd for hwnd, text in _visible_windows() if title.lower() in text.lower()), None)


def _client_area(hwnd) -> tuple[int, int, int, int]:
    """Inner rectangle of the window (without borders or title bar) in screen coordinates."""
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    return origin.x, origin.y, rect.right, rect.bottom


def list_windows() -> list[dict]:
    """Visible windows of the node's desktop that can be used as a source (title and inner size)."""
    if not SUPPORTED:
        return []
    windows = []
    for hwnd, title in _visible_windows():
        if title == "Program Manager":   # the desktop itself
            continue
        _, _, width, height = _client_area(hwnd)
        if width >= 160 and height >= 120:
            windows.append({"title": title, "width": width, "height": height})
    return sorted(windows, key=lambda w: w["title"].lower())


def capture(title: str, region: tuple[int, int, int, int] | None = None, sct=None) -> np.ndarray | None:
    """Current image of the window (BGR), or None if it is not visible."""
    if not SUPPORTED:
        return None
    hwnd = _find_window(title)
    if hwnd is None:
        return None
    x, y, w, h = _client_area(hwnd)
    if region:
        rx, ry, rw, rh = region
        x, y, w, h = x + rx, y + ry, min(rw, w - rx), min(rh, h - ry)
    if w <= 0 or h <= 0:
        return None
    if sct is None:
        with mss.mss() as own:
            return capture(title, region, own)
    image = np.asarray(sct.grab({"left": x, "top": y, "width": w, "height": h}))
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


class WindowReader:
    """Same interface as CameraReader: read() -> (frame, ts), close(), connected."""

    def __init__(self, spec: str):
        if not SUPPORTED:
            raise RuntimeError("La captura de ventanas solo está disponible en Windows")
        self.title, self.region = parse(spec)
        self.frame = None
        self.ts = 0.0
        self.connected = False
        self.error: str | None = None
        self.active = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True, name=f"window-{self.title}")
        self.thread.start()

    def _loop(self):
        with mss.mss() as sct:
            while self.active:
                started = time.time()
                try:
                    frame = capture(self.title, self.region, sct)
                except Exception as ex:  # noqa: BLE001 - e.g. the desktop is locked; retried
                    frame = None
                    log.debug("[window %s] capture failed: %s", self.title, ex)
                if frame is None:
                    if self.connected or self.error is None:
                        log.warning("[window %s] not visible, waiting", self.title)
                    self.connected, self.error = False, f"La ventana '{self.title}' no está visible en el equipo de inferencia"
                else:
                    if not self.connected:
                        log.info("[window %s] capturing", self.title)
                    self.connected, self.error = True, None
                    with self.lock:
                        self.frame, self.ts = frame, time.time()
                time.sleep(max(0.0, 1 / CAPTURE_FPS - (time.time() - started)))

    def read(self):
        with self.lock:
            return self.frame, self.ts

    def close(self):
        self.active = False
        self.thread.join(timeout=3)
