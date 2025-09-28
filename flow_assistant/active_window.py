"""Active window inspection utilities for Flow Assistant."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


@dataclass(slots=True)
class ActiveWindowInfo:
    """Details describing the current foreground window."""

    handle: int
    title: str
    process_name: str
    process_path: Optional[Path]
    timestamp: datetime
    rect: tuple[int, int, int, int]

    @property
    def app_label(self) -> str:
        if self.process_name:
            return self.process_name
        if self.process_path:
            return self.process_path.name
        return "Unknown"


class ActiveWindowProvider:
    """Windows-specific helper that retrieves foreground window metadata."""

    def __init__(self) -> None:
        self._platform_checked = False
        self._supported = False
        self._init_platform()

    def _init_platform(self) -> None:
        from sys import platform

        if not platform.startswith("win"):
            logger.warning("ActiveWindowProvider currently supports only Windows platforms.")
            return
        self._supported = True
        try:
            import ctypes
            from ctypes import wintypes

            self._ctypes = ctypes  # type: ignore[attr-defined]
            self._wintypes = wintypes  # type: ignore[attr-defined]
            user32 = ctypes.windll.user32
            self._get_foreground_window = user32.GetForegroundWindow
            self._get_window_text_length = user32.GetWindowTextLengthW
            self._get_window_text = user32.GetWindowTextW
            self._get_window_rect = user32.GetWindowRect
            self._get_window_thread_process_id = user32.GetWindowThreadProcessId
            kernel32 = ctypes.windll.kernel32
            self._open_process = kernel32.OpenProcess
            self._close_handle = kernel32.CloseHandle
            try:
                self._psapi = ctypes.windll.psapi
            except OSError:  # pragma: no cover - optional dependency
                self._psapi = None
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.exception("Failed to initialise ActiveWindowProvider: %s", exc)
            self._supported = False
        finally:
            self._platform_checked = True

    def is_supported(self) -> bool:
        if not self._platform_checked:
            self._init_platform()
        return self._supported

    def current(self) -> Optional[ActiveWindowInfo]:
        if not self.is_supported():
            return None
        hwnd = self._get_foreground_window()
        if not hwnd:
            return None
        title = self._window_title(hwnd)
        pid = self._window_process_id(hwnd)
        process_name, process_path = self._process_details(pid)
        rect = self._window_rect(hwnd)
        return ActiveWindowInfo(
            handle=int(hwnd),
            title=title,
            process_name=process_name,
            process_path=process_path,
            timestamp=datetime.now(timezone.utc),
            rect=rect,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _window_title(self, hwnd: int) -> str:
        length = self._get_window_text_length(hwnd)
        if length == 0:
            # Some windows (console, UWP) may return zero length even when text exists.
            length = 1024
        buffer = self._ctypes.create_unicode_buffer(length + 1)
        self._get_window_text(hwnd, buffer, length + 1)
        return buffer.value.strip()

    def _window_process_id(self, hwnd: int) -> int:
        pid = self._wintypes.DWORD()
        self._get_window_thread_process_id(hwnd, self._ctypes.byref(pid))
        return int(pid.value)

    def _window_rect(self, hwnd: int) -> tuple[int, int, int, int]:
        rect = self._wintypes.RECT()
        result = self._get_window_rect(hwnd, self._ctypes.byref(rect))
        if result == 0:
            return (0, 0, 0, 0)
        return (rect.left, rect.top, rect.right, rect.bottom)

    def _process_details(self, pid: int) -> tuple[str, Optional[Path]]:
        name = ""
        path: Optional[Path] = None
        if pid <= 0:
            return name, path
        if psutil is not None:
            try:
                proc = psutil.Process(pid)
                name = proc.name()
                exe = proc.exe()
                path = Path(exe) if exe else None
                return name, path
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):  # type: ignore[attr-defined]
                pass
        if getattr(self, "_psapi", None) is None:
            return name, path
        process_handle = self._open_process(0x0410, False, pid)  # PROCESS_QUERY_INFORMATION | PROCESS_VM_READ
        if not process_handle:
            return name, path
        try:
            buffer = self._ctypes.create_unicode_buffer(260)
            chars = self._psapi.GetModuleFileNameExW(process_handle, 0, buffer, len(buffer))
            if chars:
                candidate = Path(buffer.value)
                path = candidate
                name = candidate.name
        finally:
            self._close_handle(process_handle)
        return name, path
