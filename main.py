"""CLI demo for Flow Assistant prototype."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import dedent

from flow_assistant.collector import snapshot_from_dict
from flow_assistant.models import ContextSnapshot
from flow_assistant.pipeline import FlowAssistant


def load_requirement_text() -> str:
    path = Path("docs/requiement.md")
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def sample_contexts(base_time: datetime) -> list[ContextSnapshot]:
    payloads = [
        {
            "ts": (base_time + timedelta(minutes=0)).isoformat(),
            "app": "Excel",
            "window_title": "Budget_Q1.xlsx",
            "selected_text": "Product,Jan,Feb\nAlpha,1200,1500\nBeta,900,1100",
        },
        {
            "ts": (base_time + timedelta(minutes=5)).isoformat(),
            "app": "Edge",
            "window_title": "Research - Generative AI Market",
            "selected_text": "OpenAI has announced new updates at the DevDay event.",
        },
        {
            "ts": (base_time + timedelta(minutes=12)).isoformat(),
            "app": "VSCode",
            "window_title": "app/services/report.py - Visual Studio Code",
            "selected_text": dedent(
                """
                def build_report(data):
                    # TODO: consolidate metrics
                    pass

                def build_report(data):
                    return renderer.render(data)
                """
            ),
        },
        {
            "ts": (base_time + timedelta(minutes=25)).isoformat(),
            "app": "PDF",
            "window_title": "Contract_2025.pdf",
            "selected_text": "締結期限: 2025/01/15\n支払い金額: ¥1,200,000 (JPY)",
        },
    ]
    return [snapshot_from_dict(payload) for payload in payloads]


def run_demo() -> None:
    assistant = FlowAssistant()
    requirement_text = load_requirement_text()
    if requirement_text:
        assistant.ingest_document(
            path_or_url="docs/requiement.md",
            title="Flow Assistant 要件定義",
            text=requirement_text,
        )
    base_time = datetime.utcnow().replace(second=0, microsecond=0)
    contexts = sample_contexts(base_time)
    print("Flow Assistant Demo")
    print("===================")
    for snapshot in contexts:
        print(f"\n[Context] {snapshot.app} | {snapshot.window_title}")
        if snapshot.selected_text:
            print("Excerpt:\n" + snapshot.selected_text)
        cards = assistant.observe(snapshot)
        if not cards:
            print("  -> No suggestion triggered")
            continue
        for card in cards:
            print("\nAssist Card")
            print(card.render_text())
            action = input("Action? (adopt/ignore/snooze, default=ignore): ").strip() or "ignore"
            episode = assistant.record_user_action(card.suggestion.sid, action)
            if episode:
                print(f"  -> Recorded episode {episode.eid} with status {episode.result_status}")
    print("\nReports")
    print("-------")
    print(assistant.daily_report_text())
    print()
    print(assistant.weekly_report_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow Assistant CLI")
    parser.add_argument("command", nargs="?", default="demo", choices=["demo"], help="Command to run")
    args = parser.parse_args()
    if args.command == "demo":
        run_demo()


if __name__ == "__main__":
    main()
