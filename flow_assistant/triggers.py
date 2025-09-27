"""Rule-based trigger engine for proactive suggestions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, List

from .models import ContextSnapshot, Suggestion
from .utils import generate_id


@dataclass(slots=True)
class SuggestionTemplate:
    """Intermediate representation produced by trigger rules."""

    suggestion_type: str
    payload: str
    action_hint: str
    sources: List[str]
    base_score: float = 0.5


Trigger = Callable[[ContextSnapshot], SuggestionTemplate | None]


class TriggerEngine:
    """Evaluates context snapshots against rule set."""

    def __init__(self, rules: Iterable[Trigger]) -> None:
        self.rules = list(rules)

    def evaluate(self, snapshot: ContextSnapshot) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        for rule in self.rules:
            template = rule(snapshot)
            if not template:
                continue
            suggestion = Suggestion(
                sid=generate_id("sgg"),
                ts=datetime.utcnow(),
                suggestion_type=template.suggestion_type,
                payload=template.payload,
                sources=template.sources,
                score=template.base_score,
                metadata={"action_hint": template.action_hint},
            )
            suggestions.append(suggestion)
        return suggestions


def excel_chart_rule(snapshot: ContextSnapshot) -> SuggestionTemplate | None:
    if snapshot.app.lower() != "excel" and "excel" not in snapshot.window_title.lower():
        return None
    digits = sum(c.isdigit() for c in snapshot.selected_text)
    commas = snapshot.selected_text.count(",")
    if digits < 4 and commas < 2:
        return None
    payload = "選択中のデータからグラフ（棒/折れ線/散布）を作成しますか？"
    return SuggestionTemplate(
        suggestion_type="visualization",
        payload=payload,
        action_hint="chart_generate",
        sources=["rule:excel_chart"],
        base_score=0.65,
    )


def browser_reference_rule(snapshot: ContextSnapshot) -> SuggestionTemplate | None:
    app = snapshot.app.lower()
    browser_apps = {"edge", "chrome", "firefox", "safari", "opera", "browser"}
    if app not in browser_apps and not any(br in snapshot.window_title.lower() for br in browser_apps):
        return None
    proper_nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", snapshot.selected_text)
    if not proper_nouns:
        return None
    payload = f"『{proper_nouns[0]}』に関連する資料をローカルから提示しますか？"
    return SuggestionTemplate(
        suggestion_type="reference",
        payload=payload,
        action_hint="open_related_docs",
        sources=["rule:browser_reference"],
        base_score=0.55 + min(len(proper_nouns) * 0.05, 0.15),
    )


def vscode_similarity_rule(snapshot: ContextSnapshot) -> SuggestionTemplate | None:
    markers = ["vscode", "visual studio code"]
    if snapshot.app.lower() not in {"vscode", "code"} and not any(m in snapshot.window_title.lower() for m in markers):
        return None
    functions = re.findall(r"def\s+([a-zA-Z_][\w]*)", snapshot.selected_text)
    duplicates = {fn for fn in functions if functions.count(fn) > 1}
    if not duplicates:
        return None
    name = sorted(duplicates)[0]
    payload = f"関数 {name} の他の出現箇所の修正やIssue履歴を確認しますか？"
    return SuggestionTemplate(
        suggestion_type="code_assist",
        payload=payload,
        action_hint="show_git_history",
        sources=["rule:vscode_similarity"],
        base_score=0.6,
    )


def pdf_contract_rule(snapshot: ContextSnapshot) -> SuggestionTemplate | None:
    if "pdf" not in snapshot.app.lower() and "pdf" not in snapshot.window_title.lower():
        return None
    has_deadline = bool(re.search(r"(\d{4}[/-]\d{1,2}[/-]\d{1,2}|\b\d{1,2}/\d{1,2}\b)", snapshot.selected_text))
    has_amount = bool(re.search(r"(¥\s?\d+[\d,]*|USD|JPY|\d+万円)", snapshot.selected_text))
    if not (has_deadline and has_amount):
        return None
    payload = "契約書の期日と金額を元にToDoとカレンダー案を作成しますか？"
    return SuggestionTemplate(
        suggestion_type="tasking",
        payload=payload,
        action_hint="draft_calendar",
        sources=["rule:pdf_contract"],
        base_score=0.7,
    )


DEFAULT_RULES: List[Trigger] = [
    excel_chart_rule,
    browser_reference_rule,
    vscode_similarity_rule,
    pdf_contract_rule,
]
