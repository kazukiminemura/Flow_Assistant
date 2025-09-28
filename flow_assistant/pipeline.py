"""End-to-end orchestration for Flow Assistant."""

from __future__ import annotations

from typing import Dict, List

from .actions import ActionExecutor
from .collector import ContextCollector
from .index import DocumentIndex
from .learning import LearningEngine
from .models import ContextSnapshot, Document, Episode, Suggestion
from .preprocessor import Preprocessor
from .rag import RAGEngine
from .reporting import daily_report, weekly_report
from .triggers import DEFAULT_RULES, TriggerEngine
from .ui import AssistCard, build_cards


class FlowAssistant:
    """Coordinates the modules required for proactive assistance."""

    def __init__(
        self,
        *,
        collector: ContextCollector | None = None,
        trigger_engine: TriggerEngine | None = None,
        index: DocumentIndex | None = None,
        preprocessor: Preprocessor | None = None,
        action_executor: ActionExecutor | None = None,
        learning: LearningEngine | None = None,
    ) -> None:
        self.collector = collector or ContextCollector()
        self.preprocessor = preprocessor or Preprocessor()
        self.index = index or DocumentIndex()
        self.rag = RAGEngine(self.index, self.preprocessor)
        self.trigger_engine = trigger_engine or TriggerEngine(DEFAULT_RULES)
        self.action_executor = action_executor or ActionExecutor()
        self.learning = learning or LearningEngine()
        self.episodes: List[Episode] = []
        self.suggestion_registry: Dict[str, Suggestion] = {}

    def observe(self, snapshot: ContextSnapshot) -> List[AssistCard]:
        """Process a new context snapshot and emit assist cards."""

        self.collector.ingest(snapshot)
        suggestions = self.trigger_engine.evaluate(snapshot)
        context_preview = self._build_context_preview(snapshot)
        sources_map: Dict[str, List[Document]] = {}
        for suggestion in suggestions:
            self.learning.adjust_score(suggestion)
            if snapshot.screenshot_path:
                suggestion.metadata.setdefault("context_screenshot", str(snapshot.screenshot_path))
            context_text = self._context_text(snapshot)
            documents = self.rag.retrieve_support(suggestion, context_text=context_text)
            existing_sources = set(suggestion.sources)
            for doc in documents:
                if doc.doc_id not in existing_sources:
                    suggestion.sources.append(doc.doc_id)
                    existing_sources.add(doc.doc_id)
            sources_map[suggestion.sid] = documents
            self.suggestion_registry[suggestion.sid] = suggestion
        cards = build_cards(
            suggestions,
            context_preview=context_preview,
            sources_map=sources_map,
            screenshot_path=snapshot.screenshot_path,
        )
        return cards

    def ingest_document(self, *, path_or_url: str, title: str, text: str) -> None:
        self.rag.ingest_document(path_or_url=path_or_url, title=title, text=text)

    def record_user_action(self, sid: str, action: str) -> Episode | None:
        suggestion = self.suggestion_registry.get(sid)
        if not suggestion:
            return None
        action_lower = action.lower()
        outcome = None
        if action_lower == "adopt":
            action_hint = suggestion.metadata.get("action_hint", "")
            outcome = self.action_executor.execute(suggestion, action_hint)
        episode = self.action_executor.to_episode(
            suggestion,
            user_action=action_lower,
            outcome=outcome,
        )
        self.episodes.append(episode)
        self.learning.record_episode(episode, suggestion)
        return episode

    def daily_report_text(self) -> str:
        return daily_report(self.episodes, self.suggestion_registry).render_text()

    def weekly_report_text(self) -> str:
        return weekly_report(self.episodes, self.suggestion_registry).render_text()

    @staticmethod
    def _build_context_preview(snapshot: ContextSnapshot, *, limit: int = 120) -> str:
        text = snapshot.selected_text.strip() or snapshot.window_title.strip()
        if len(text) > limit:
            return text[: limit - 3] + "..."
        return text

    @staticmethod
    def _context_text(snapshot: ContextSnapshot) -> str:
        parts = [snapshot.window_title, snapshot.selected_text]
        if snapshot.participants:
            parts.append("Participants: " + ", ".join(snapshot.participants))
        return "\n".join(part for part in parts if part)

    def close(self) -> None:
        self.index.close()
