"""Data models aligned with Flow Assistant requirement definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass(slots=True)
class Document:
    """Represents indexed material retrievable for grounding suggestions."""

    doc_id: str
    path_or_url: str
    title: str
    summary: str
    embedding: Sequence[float]
    updated_at: datetime


@dataclass(slots=True)
class ContextSnapshot:
    """Captures the current working context observed by collectors."""

    ts: datetime
    app: str
    window_title: str
    selected_text: str = ""
    participants: List[str] = field(default_factory=list)
    screenshot_path: Optional[Path] = None


@dataclass(slots=True)
class Suggestion:
    """Proactive recommendation surfaced as an assist card."""

    sid: str
    ts: datetime
    suggestion_type: str
    payload: str
    sources: List[str]
    score: float
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Episode:
    """Tracks lifecycle of a suggestion from display to user outcome."""

    eid: str
    sid: str
    user_action: str
    result_status: str
    rollback_ref: Optional[str] = None
    recorded_at: datetime = field(default_factory=datetime.utcnow)
