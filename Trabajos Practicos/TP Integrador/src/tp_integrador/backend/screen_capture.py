import cv2
import numpy as np


class ScreenCapture:
    """Adaptador simple para leer la pantalla con la misma interfaz que VideoCapture."""

    def __init__(self, monitor_index: int = 1, width: int | None = 1280, height: int | None = 720) -> None:
        self.width = width
        self.height = height
        self._opened = False
        self._sct = None
        self._monitor = None

        try:
            import mss

            self._sct = mss.mss()
            monitors = self._sct.monitors
            if not monitors:
                return
            monitor_index = min(max(monitor_index, 0), len(monitors) - 1)
            self._monitor = monitors[monitor_index]
            self._opened = True
        except Exception:
            self.release()

    def isOpened(self) -> bool:
        return self._opened and self._sct is not None and self._monitor is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.isOpened():
            return False, None
        try:
            frame_bgra = np.asarray(self._sct.grab(self._monitor))
        except Exception:
            return False, None
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        if self.width is not None and self.height is not None:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return True, frame_bgr

    def release(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
        self._sct = None
        self._monitor = None
        self._opened = False


class WindowCapture:
    """Captura una ventana por coincidencia parcial de titulo."""

    def __init__(self, title: str, width: int | None = 1280, height: int | None = 720) -> None:
        self.title = title.strip()
        self.width = width
        self.height = height
        self._opened = False
        self._sct = None
        self._window = None

        try:
            import mss
            import pygetwindow as gw

            if not self.title:
                return
            windows = [window for window in gw.getWindowsWithTitle(self.title) if window.width > 0 and window.height > 0]
            if not windows:
                return
            self._window = windows[0]
            self._sct = mss.mss()
            self._opened = True
        except Exception:
            self.release()

    def isOpened(self) -> bool:
        return self._opened and self._sct is not None and self._window is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.isOpened():
            return False, None
        if getattr(self._window, "isMinimized", False):
            return False, None
        region = {
            "left": max(0, int(self._window.left)),
            "top": max(0, int(self._window.top)),
            "width": max(1, int(self._window.width)),
            "height": max(1, int(self._window.height)),
        }
        try:
            frame_bgra = np.asarray(self._sct.grab(region))
        except Exception:
            return False, None
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        if self.width is not None and self.height is not None:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return True, frame_bgr

    def release(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
        self._sct = None
        self._window = None
        self._opened = False


def open_screen_capture(
    monitor_index: int = 1,
    width: int | None = 1280,
    height: int | None = 720,
) -> ScreenCapture:
    return ScreenCapture(monitor_index=monitor_index, width=width, height=height)


def open_window_capture(
    title: str,
    width: int | None = 1280,
    height: int | None = 720,
) -> WindowCapture:
    return WindowCapture(title=title, width=width, height=height)


def list_available_windows() -> list[str]:
    """Devuelve titulos de ventanas visibles que pueden capturarse."""
    try:
        import pygetwindow as gw
    except Exception:
        return []

    titles = []
    seen = set()
    for window in gw.getAllWindows():
        title = window.title.strip()
        if not title or title in seen:
            continue
        if window.width <= 0 or window.height <= 0 or getattr(window, "isMinimized", False):
            continue
        seen.add(title)
        titles.append(title)
    return sorted(titles, key=str.lower)
