"""Utilities for capturing desktop screenshots."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import mss  # type: ignore
    from mss import tools as mss_tools  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    mss = None  # type: ignore
    mss_tools = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from PIL import ImageGrab  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ImageGrab = None  # type: ignore

CaptureBackend = Callable[[Path], None]


class ScreenCapturer:
    """Captures screenshots to a local directory with graceful fallbacks."""

    def __init__(
        self,
        *,
        output_dir: Path | str | None = None,
        prefix: str = "capture",
        image_format: str = "png",
        monitor_index: int = 1,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("captures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.monitor_index = monitor_index
        backend, resolved_format = self._select_backend(image_format.lower())
        self._backend: Optional[CaptureBackend] = backend
        self.image_format = resolved_format

    def capture(self) -> Path | None:
        """Capture the current screen and return the saved file path."""

        if not self._backend:
            logger.warning("Screen capture backend is not available.")
            return None
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.prefix}_{timestamp}.{self.image_format}"
        destination = self.output_dir / filename
        try:
            self._backend(destination)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.exception("Screen capture failed: %s", exc)
            return None
        if not destination.exists():
            logger.error("Screen capture backend did not produce a file: %s", destination)
            return None
        return destination

    # ------------------------------------------------------------------
    # Backend selection helpers
    # ------------------------------------------------------------------

    def _select_backend(self, preferred_format: str) -> Tuple[Optional[CaptureBackend], str]:
        fmt = preferred_format or "png"
        if mss is not None and mss_tools is not None:
            return self._capture_with_mss, fmt
        if ImageGrab is not None:
            return self._capture_with_pillow, fmt
        if self._is_windows():  # fallback to Win32 GDI; outputs BMP
            return self._capture_with_gdi, "bmp"
        return None, fmt

    @staticmethod
    def _is_windows() -> bool:
        from sys import platform

        return platform.startswith("win")

    # ------------------------------------------------------------------
    # Concrete backend implementations
    # ------------------------------------------------------------------

    def _capture_with_mss(self, destination: Path) -> None:
        assert mss is not None and mss_tools is not None  # for type checkers
        with mss.mss() as sct:
            monitors = sct.monitors
            index = min(max(self.monitor_index, 1), len(monitors) - 1)
            shot = sct.grab(monitors[index])
            mss_tools.to_png(shot.rgb, shot.size, output=str(destination))

    def _capture_with_pillow(self, destination: Path) -> None:
        assert ImageGrab is not None  # for type checkers
        image = ImageGrab.grab()  # type: ignore[attr-defined]
        image.save(destination, format=self.image_format.upper())

    def _capture_with_gdi(self, destination: Path) -> None:
        """Capture the virtual desktop using Win32 GDI and write a BMP image."""

        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        user32.SetProcessDPIAware()
        left = user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
        top = user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN
        width = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
        height = user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN

        if width <= 0 or height <= 0:
            raise RuntimeError("Unable to determine screen bounds")

        desktop_dc = user32.GetDC(0)
        mem_dc = gdi32.CreateCompatibleDC(desktop_dc)
        bitmap = gdi32.CreateCompatibleBitmap(desktop_dc, width, height)
        if not bitmap:
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(0, desktop_dc)
            raise RuntimeError("CreateCompatibleBitmap failed")

        try:
            gdi32.SelectObject(mem_dc, bitmap)
            srccopy = 0x00CC0020
            if gdi32.BitBlt(mem_dc, 0, 0, width, height, desktop_dc, left, top, srccopy) == 0:
                raise RuntimeError("BitBlt failed")

            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ("biSize", wintypes.DWORD),
                    ("biWidth", wintypes.LONG),
                    ("biHeight", wintypes.LONG),
                    ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD),
                    ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD),
                    ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG),
                    ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD),
                ]

            class BITMAPINFO(ctypes.Structure):
                _fields_ = [
                    ("bmiHeader", BITMAPINFOHEADER),
                    ("bmiColors", wintypes.DWORD * 3),
                ]

            bmi = BITMAPINFO()
            bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.bmiHeader.biWidth = width
            bmi.bmiHeader.biHeight = height
            bmi.bmiHeader.biPlanes = 1
            bmi.bmiHeader.biBitCount = 32
            bmi.bmiHeader.biCompression = 0  # BI_RGB
            bmi.bmiHeader.biSizeImage = width * height * 4

            buf_size = bmi.bmiHeader.biSizeImage
            pixel_buffer = (ctypes.c_ubyte * buf_size)()
            bits = gdi32.GetDIBits(
                mem_dc,
                bitmap,
                0,
                height,
                ctypes.byref(pixel_buffer),
                ctypes.byref(bmi),
                0,
            )
            if bits == 0:
                raise RuntimeError("GetDIBits failed")

            import struct

            file_header_size = 14
            info_header_size = ctypes.sizeof(BITMAPINFOHEADER)
            offset = file_header_size + info_header_size
            file_size = offset + buf_size
            bmp_header = struct.pack(
                "<2sIHHI",
                b"BM",
                file_size,
                0,
                0,
                offset,
            )
            info_header = struct.pack(
                "<IIIHHIIIIII",
                info_header_size,
                width,
                height,
                1,
                32,
                0,
                buf_size,
                0,
                0,
                0,
                0,
            )

            destination.write_bytes(bmp_header + info_header + bytes(pixel_buffer))
        finally:
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(0, desktop_dc)
