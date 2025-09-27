"""Flow Assistant core package implementing on-device proactive assistant pipeline."""

from .models import Document, ContextSnapshot, Suggestion, Episode
from .pipeline import FlowAssistant

__all__ = [
    "FlowAssistant",
    "Document",
    "ContextSnapshot",
    "Suggestion",
    "Episode",
]
