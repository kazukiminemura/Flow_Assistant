"""Flow Assistant core package implementing on-device proactive assistant pipeline."""

from .active_window import ActiveWindowInfo, ActiveWindowProvider
from .models import Document, ContextSnapshot, Suggestion, Episode
from .ocr import extract_text, summarize_text
from .pipeline import FlowAssistant
from .screen_capture import ScreenCapturer

__all__ = [
    "FlowAssistant",
    "ActiveWindowProvider",
    "ActiveWindowInfo",
    "ScreenCapturer",
    "extract_text",
    "summarize_text",
    "Document",
    "ContextSnapshot",
    "Suggestion",
    "Episode",
]
