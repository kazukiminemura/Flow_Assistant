"""Retrieval augmented generation glue for Flow Assistant."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List

from .index import DocumentIndex
from .models import Document, Suggestion
from .preprocessor import PreprocessResult, Preprocessor
from .utils import generate_id


class RAGEngine:
    """Manages document ingestion and retrieval for grounding suggestions."""

    def __init__(self, index: DocumentIndex, preprocessor: Preprocessor) -> None:
        self.index = index
        self.preprocessor = preprocessor

    def ingest_document(self, *, path_or_url: str, title: str, text: str) -> Document:
        result: PreprocessResult = self.preprocessor.run(text)
        document = Document(
            doc_id=generate_id("doc"),
            path_or_url=path_or_url,
            title=title,
            summary=result.summary or text[:200],
            embedding=result.embedding,
            updated_at=datetime.utcnow(),
        )
        self.index.upsert(document)
        return document

    def ingest_many(self, documents: Iterable[tuple[str, str, str]]) -> List[Document]:
        stored = []
        for path_or_url, title, text in documents:
            stored.append(self.ingest_document(path_or_url=path_or_url, title=title, text=text))
        return stored

    def retrieve_support(self, suggestion: Suggestion, *, context_text: str, limit: int = 3) -> List[Document]:
        query = f"{suggestion.payload}\n{context_text}"
        query_embedding = self.preprocessor.run(query).embedding
        return self.index.search(query_embedding, limit=limit)
