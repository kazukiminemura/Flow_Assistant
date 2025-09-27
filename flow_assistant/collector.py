"""Context collection abstractions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Iterable, Iterator

from .models import ContextSnapshot


@dataclass(slots=True)
class CollectorConfig:
    """Configuration for the context collector ring buffer."""

    max_snapshots: int = 128


class ContextCollector:
    """Maintains a rolling buffer of recent context snapshots."""

    def __init__(self, config: CollectorConfig | None = None) -> None:
        self.config = config or CollectorConfig()
        self._buffer: Deque[ContextSnapshot] = deque(maxlen=self.config.max_snapshots)

    def ingest(self, snapshot: ContextSnapshot) -> None:
        self._buffer.append(snapshot)

    def extend(self, snapshots: Iterable[ContextSnapshot]) -> None:
        for snapshot in snapshots:
            self.ingest(snapshot)

    def latest(self) -> ContextSnapshot | None:
        return self._buffer[-1] if self._buffer else None

    def iter_recent(self) -> Iterator[ContextSnapshot]:
        return iter(reversed(self._buffer))


def snapshot_from_dict(payload: dict) -> ContextSnapshot:
    """Helper to construct a snapshot from a dictionary."""

    ts_raw = payload.get("ts")
    if isinstance(ts_raw, (int, float)):
        ts = datetime.fromtimestamp(ts_raw)
    elif isinstance(ts_raw, str):
        ts = datetime.fromisoformat(ts_raw)
    elif isinstance(ts_raw, datetime):
        ts = ts_raw
    else:
        ts = datetime.utcnow()
    return ContextSnapshot(
        ts=ts,
        app=payload.get("app", "Unknown"),
        window_title=payload.get("window_title", ""),
        selected_text=payload.get("selected_text", ""),
        participants=list(payload.get("participants", [])),
    )
