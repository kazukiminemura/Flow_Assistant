"""Lightweight preprocessing leveraging on-device operations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .utils import stable_hash

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(slots=True)
class PreprocessResult:
    summary: str
    embedding: list[float]


class Preprocessor:
    """Performs summarisation and embedding.

    This module intentionally uses simple heuristics to stay aligned with the
    requirement of on-device, dependency-light processing while keeping the
    interface open for future NPU acceleration.
    """

    def __init__(self, embedding_size: int = 16) -> None:
        self.embedding_size = embedding_size

    def run(self, text: str) -> PreprocessResult:
        summary = self._summarise(text)
        embedding = list(stable_hash(summary or text, length=self.embedding_size))
        return PreprocessResult(summary=summary, embedding=embedding)

    @staticmethod
    def _summarise(text: str, max_sentences: int = 2) -> str:
        stripped = text.strip()
        if not stripped:
            return ""
        sentences = _SENTENCE_SPLIT.split(stripped)
        if len(sentences) <= max_sentences:
            return " ".join(sentences[:max_sentences]).strip()
        # Prioritise sentences with key verbs typically tied to actions.
        scored = []
        for sentence in sentences:
            verbs = re.findall(r"\b(create|update|review|discuss|analyze|提案|確認|予定)\b", sentence, flags=re.IGNORECASE)
            score = len(verbs) * 2 + len(sentence)
            scored.append((score, sentence))
        top = sorted(scored, reverse=True)[:max_sentences]
        ordered = [sentence for _, sentence in sorted(top, key=lambda pair: sentences.index(pair[1]))]
        return " ".join(ordered).strip()
