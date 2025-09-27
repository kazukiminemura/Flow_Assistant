"""Simple online learner tracking suggestion outcomes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

from .models import Episode, Suggestion


@dataclass(slots=True)
class SuggestionStats:
    accepted: int = 0
    shown: int = 0

    def adoption_rate(self) -> float:
        if not self.shown:
            return 0.0
        return self.accepted / self.shown


class LearningEngine:
    """Updates per-action priors using user feedback."""

    def __init__(self) -> None:
        self._stats: Dict[str, SuggestionStats] = defaultdict(SuggestionStats)

    def adjust_score(self, suggestion: Suggestion) -> None:
        key = suggestion.metadata.get("action_hint", suggestion.suggestion_type)
        stats = self._stats[key]
        booster = min(stats.adoption_rate(), 0.3)
        penalty = 0.1 if stats.shown and stats.adoption_rate() < 0.1 else 0.0
        suggestion.score = max(0.0, min(1.0, suggestion.score + booster - penalty))
        stats.shown += 1

    def record_episode(self, episode: Episode, suggestion: Suggestion) -> None:
        key = suggestion.metadata.get("action_hint", suggestion.suggestion_type)
        stats = self._stats[key]
        if episode.user_action.lower() == "adopt":
            stats.accepted += 1

    def stats_snapshot(self) -> Dict[str, SuggestionStats]:
        return dict(self._stats)
