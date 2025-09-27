"""Action execution stubs for Flow Assistant."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .models import Episode, Suggestion
from .utils import generate_id


@dataclass(slots=True)
class ActionOutcome:
    status: str
    detail: str = ""


class ActionExecutor:
    """Executes the real-world side effects for accepted suggestions.

    Current implementation simulates the effects and returns structured
    outcomes that can be logged and used by the learning module.
    """

    def execute(self, suggestion: Suggestion, action_hint: str) -> ActionOutcome:
        mapping = {
            "chart_generate": "Excel用の臨時グラフを生成しました",
            "open_related_docs": "関連ドキュメントを5件開きました",
            "show_git_history": "関連するコミット履歴を表示しました",
            "draft_calendar": "ToDoとカレンダー案をドラフトしました",
        }
        detail = mapping.get(action_hint, "アクションをシミュレートしました")
        return ActionOutcome(status="executed", detail=detail)

    def to_episode(self, suggestion: Suggestion, user_action: str, outcome: ActionOutcome | None, rollback_ref: str | None = None) -> Episode:
        return Episode(
            eid=generate_id("ep"),
            sid=suggestion.sid,
            user_action=user_action,
            result_status=outcome.status if outcome else "skipped",
            rollback_ref=rollback_ref,
            recorded_at=datetime.utcnow(),
        )
