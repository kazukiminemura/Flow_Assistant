"""Reporting utilities for Flow Assistant."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, List

from .models import Episode, Suggestion


@dataclass(slots=True)
class Report:
    title: str
    summary_lines: List[str]

    def render_text(self) -> str:
        return "\n".join([self.title, "-" * len(self.title), *self.summary_lines])


def daily_report(episodes: Iterable[Episode], suggestions: dict[str, Suggestion], *, target_date: date | None = None) -> Report:
    target_date = target_date or datetime.utcnow().date()
    day_episodes = [ep for ep in episodes if ep.recorded_at.date() == target_date]
    if not day_episodes:
        return Report(title=f"{target_date} Daily Report", summary_lines=["No activity recorded."])
    adopted = [ep for ep in day_episodes if ep.user_action.lower() == "adopt"]
    adoption_rate = len(adopted) / len(day_episodes)
    breakdown = Counter(suggestions[ep.sid].suggestion_type for ep in day_episodes if ep.sid in suggestions)
    lines = [
        f"Total suggestions: {len(day_episodes)}",
        f"Adoption rate: {adoption_rate:.0%}",
    ]
    for suggestion_type, count in breakdown.most_common():
        lines.append(f"- {suggestion_type}: {count}")
    return Report(title=f"{target_date} Daily Report", summary_lines=lines)


def weekly_report(episodes: Iterable[Episode], suggestions: dict[str, Suggestion], *, end_date: date | None = None) -> Report:
    end_date = end_date or datetime.utcnow().date()
    start_date = end_date - timedelta(days=6)
    week_episodes = [ep for ep in episodes if start_date <= ep.recorded_at.date() <= end_date]
    if not week_episodes:
        return Report(title=f"Week ending {end_date}", summary_lines=["No activity recorded."])
    adoption_counts = Counter()
    noise_counts = Counter()
    type_counts = Counter()
    for ep in week_episodes:
        suggestion = suggestions.get(ep.sid)
        if not suggestion:
            continue
        type_counts[suggestion.suggestion_type] += 1
        if ep.user_action.lower() == "adopt":
            adoption_counts[suggestion.suggestion_type] += 1
        elif ep.user_action.lower() == "ignore":
            noise_counts[suggestion.suggestion_type] += 1
    lines = [
        f"Span: {start_date} - {end_date}",
        f"Total suggestions: {len(week_episodes)}",
    ]
    for suggestion_type, total in type_counts.most_common():
        adopted = adoption_counts[suggestion_type]
        noise = noise_counts[suggestion_type]
        lines.append(
            f"- {suggestion_type}: {total} shown, {adopted} adopted, {noise} ignored"
        )
    return Report(title=f"Week ending {end_date}", summary_lines=lines)
