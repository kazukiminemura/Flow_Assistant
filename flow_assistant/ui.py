"""Presentation helpers for Flow Assistant assist cards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .models import Document, Suggestion


@dataclass(slots=True)
class AssistCard:
    suggestion: Suggestion
    context_preview: str
    sources: List[Document]

    def render_text(self) -> str:
        lines = [
            f"[{self.suggestion.suggestion_type}] {self.suggestion.payload}",
            f"  score: {self.suggestion.score:.2f}",
        ]
        if self.context_preview:
            lines.append(f"  context: {self.context_preview}")
        if self.sources:
            lines.append("  sources:")
            for doc in self.sources:
                lines.append(f"    - {doc.title} ({doc.path_or_url})")
        else:
            lines.append("  sources: (なし)")
        return "\n".join(lines)


def build_cards(suggestions: Iterable[Suggestion], *, context_preview: str, sources_map: dict[str, List[Document]]) -> List[AssistCard]:
    cards: List[AssistCard] = []
    for suggestion in suggestions:
        cards.append(
            AssistCard(
                suggestion=suggestion,
                context_preview=context_preview,
                sources=sources_map.get(suggestion.sid, []),
            )
        )
    return cards
