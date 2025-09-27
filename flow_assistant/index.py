"""SQLite-backed document index with lightweight vector support."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from math import sqrt
from typing import Iterable, List, Sequence

from .models import Document
from .utils import deserialize_vector, serialize_vector


class DocumentIndex:
    """Persist documents and provide approximate semantic lookup."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with closing(self.conn.cursor()) as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    path_or_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_updated
                ON documents(updated_at)
                """
            )
            self.conn.commit()

    def upsert(self, document: Document) -> None:
        payload = (
            document.doc_id,
            document.path_or_url,
            document.title,
            document.summary,
            serialize_vector(document.embedding),
            document.updated_at.isoformat(),
        )
        with closing(self.conn.cursor()) as cur:
            cur.execute(
                """
                INSERT INTO documents (doc_id, path_or_url, title, summary, embedding, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    path_or_url=excluded.path_or_url,
                    title=excluded.title,
                    summary=excluded.summary,
                    embedding=excluded.embedding,
                    updated_at=excluded.updated_at
                """,
                payload,
            )
            self.conn.commit()

    def bulk_upsert(self, documents: Iterable[Document]) -> None:
        for document in documents:
            self.upsert(document)

    def fetch(self, doc_id: str) -> Document | None:
        with closing(self.conn.cursor()) as cur:
            cur.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
            row = cur.fetchone()
        if not row:
            return None
        return self._row_to_document(row)

    def search(self, query_vector: Sequence[float], *, limit: int = 5) -> List[Document]:
        with closing(self.conn.cursor()) as cur:
            cur.execute("SELECT * FROM documents")
            rows = cur.fetchall()
        scored: list[tuple[float, Document]] = []
        for row in rows:
            document = self._row_to_document(row)
            score = self._cosine_similarity(query_vector, document.embedding)
            scored.append((score, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = [doc for score, doc in scored[:limit] if score > 0]
        return top

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> Document:
        return Document(
            doc_id=row["doc_id"],
            path_or_url=row["path_or_url"],
            title=row["title"],
            summary=row["summary"],
            embedding=deserialize_vector(row["embedding"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        if len(vec_a) != len(vec_b):
            # Align lengths by truncating to the shortest vector.
            length = min(len(vec_a), len(vec_b))
            vec_a = vec_a[:length]
            vec_b = vec_b[:length]
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sqrt(sum(a * a for a in vec_a))
        norm_b = sqrt(sum(b * b for b in vec_b))
        if not norm_a or not norm_b:
            return 0.0
        return dot / (norm_a * norm_b)

    def close(self) -> None:
        self.conn.close()
