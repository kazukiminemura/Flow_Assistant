"""Live Flow Assistant monitor using active window context."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from flow_assistant.active_window import ActiveWindowInfo, ActiveWindowProvider
from flow_assistant.models import ContextSnapshot
from flow_assistant.pipeline import FlowAssistant
from flow_assistant.ocr import extract_text, summarize_text

try:
    from PIL import ImageGrab  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ImageGrab = None  # type: ignore

logger = logging.getLogger(__name__)


def load_requirement_text() -> str:
    path = Path("docs/requiement.md")
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def ingest_requirements(assistant: FlowAssistant) -> None:
    requirement_text = load_requirement_text()
    if requirement_text:
        assistant.ingest_document(
            path_or_url="docs/requiement.md",
            title="Flow Assistant 要件定義",
            text=requirement_text,
        )


def build_snapshot(info: ActiveWindowInfo, *, recognized_text: str = "") -> ContextSnapshot:
    return ContextSnapshot(
        ts=info.timestamp,
        app=info.app_label,
        window_title=info.title,
        selected_text=recognized_text,
        participants=[],
        screenshot_path=None,
    )


def render_cards(cards: Iterable) -> list[str]:
    rendered: list[str] = []
    for card in cards:
        rendered.append(card.render_text())
    return rendered


def capture_window_image(info: ActiveWindowInfo):
    if ImageGrab is None:
        logger.debug("ImageGrab is unavailable; skipping window capture")
        return None
    left, top, right, bottom = info.rect
    if right <= left or bottom <= top:
        return None
    try:
        return ImageGrab.grab(bbox=(left, top, right, bottom))
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.debug("Window capture failed: %s", exc)
        return None






def summarize_context(
    info: ActiveWindowInfo,
    cards: Iterable,
    *,
    ocr_summary: Optional[str] = None,
) -> str:
    summary_parts: list[str] = []
    summary_parts.append(f"{info.app_label} window '{info.title}' is active")
    if info.process_path:
        summary_parts.append(f"executable: {info.process_path}")
    card_list = list(cards) if not isinstance(cards, list) else cards
    if card_list:
        summary_parts.append(f"suggestions: {len(card_list)}")
        top_card = card_list[0]
        summary_parts.append(
            f"top suggestion: {top_card.suggestion.suggestion_type} -> {top_card.suggestion.payload}"
        )
    else:
        summary_parts.append("suggestions: none")
    if ocr_summary:
        summary_parts.append(f"ocr summary: {ocr_summary}")
    return " | ".join(summary_parts)


def monitor_loop(*, interval: float, print_no_change: bool) -> None:
    assistant = FlowAssistant()
    ingest_requirements(assistant)
    provider = ActiveWindowProvider()
    if not provider.is_supported():
        sys.stderr.write("Active window monitoring is not supported on this platform.\n")
        return
    last_signature: tuple[int, str] | None = None
    try:
        while True:
            info = provider.current()
            if info is None:
                if print_no_change:
                    print("[info] Unable to read active window")
                time.sleep(interval)
                continue
            signature = (info.handle, info.title)
            if signature == last_signature and not print_no_change:
                time.sleep(interval)
                continue
            last_signature = signature
            timestamp_local = info.timestamp.astimezone().strftime("%H:%M:%S")
            print(f"[{timestamp_local}] {info.app_label} -> {info.title}")
            if info.process_path:
                print(f"  path: {info.process_path}")
            window_image = capture_window_image(info)
            recognized_text = ""
            ocr_summary: Optional[str] = None
            if window_image is not None:
                recognized_text = extract_text(window_image)
                if recognized_text:
                    ocr_summary = summarize_text(recognized_text)
            snapshot = build_snapshot(info, recognized_text=recognized_text)
            cards = assistant.observe(snapshot)
            card_list = list(cards) if not isinstance(cards, list) else cards
            print(f"  summary: {summarize_context(info, card_list, ocr_summary=ocr_summary)}")
            if recognized_text:
                trimmed = recognized_text if len(recognized_text) <= 200 else recognized_text[:197].rstrip() + "…"
                print(f"  ocr-text: {trimmed}")
            if ocr_summary:
                print(f"  ocr-summary: {ocr_summary}")
            rendered_cards = render_cards(card_list)
            if rendered_cards:
                print("  suggestions:")
                for card_text in rendered_cards:
                    for line in card_text.splitlines():
                        print(f"    {line}")
            else:
                print("  suggestions: (none)")
            print()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped monitoring.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow Assistant live monitor")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.5,
        help="Polling interval in seconds",
    )
    parser.add_argument(
        "--print-no-change",
        action="store_true",
        help="Always print status even when the foreground window does not change",
    )
    args = parser.parse_args()
    monitor_loop(interval=max(0.1, args.interval), print_no_change=args.print_no_change)


if __name__ == "__main__":
    main()
