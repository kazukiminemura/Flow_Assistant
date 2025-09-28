"""OCR helpers for extracting text from window captures using OpenVINO."""

from __future__ import annotations

import logging
import os
import re
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import openvino.runtime as ov  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ov = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from PIL import Image  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore

logger = logging.getLogger(__name__)

_DEFAULT_ALPHABET = os.getenv(
    "FLOW_ASSISTANT_OCR_ALPHABET",
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" " ,.:-_/\\()[]{}@#%&+*=;!?\"'",
)
_DEFAULT_BLANK_ID = int(os.getenv("FLOW_ASSISTANT_OCR_BLANK_ID", "0"))

_DEFAULT_MODEL_URLS = {
    "xml": os.getenv(
        "FLOW_ASSISTANT_OCR_MODEL_XML_URL",
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.2/models_bin/1/"
        "text-recognition-0014/FP16/text-recognition-0014.xml",
    ),
    "bin": os.getenv(
        "FLOW_ASSISTANT_OCR_MODEL_BIN_URL",
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.2/models_bin/1/"
        "text-recognition-0014/FP16/text-recognition-0014.bin",
    ),
}


@dataclass(slots=True)
class OpenVinoOCRConfig:
    """Configuration for the OpenVINO OCR pipeline."""

    recognition_model: Optional[Path]
    device: str = os.getenv("FLOW_ASSISTANT_OCR_DEVICE", "CPU")
    alphabet: str = _DEFAULT_ALPHABET
    blank_id: int = _DEFAULT_BLANK_ID

    @classmethod
    def from_env(cls) -> "OpenVinoOCRConfig":
        model_path = os.getenv("FLOW_ASSISTANT_OCR_RECOGNITION_MODEL")
        if model_path:
            recognition_model = Path(model_path).expanduser()
        else:
            default_root = Path(os.getenv("FLOW_ASSISTANT_MODEL_DIR", "models")).expanduser()
            recognition_model = default_root / "ocr" / "text-recognition-0014.xml"
        return cls(recognition_model=recognition_model)


class OpenVINOTextRecognizer:
    """Thin wrapper around an OpenVINO text recognition network."""

    def __init__(self, config: OpenVinoOCRConfig) -> None:
        if ov is None:
            raise RuntimeError("OpenVINO runtime is not installed")
        if Image is None:
            raise RuntimeError("Pillow is required for OpenVINO OCR support")
        if not config.recognition_model:
            raise FileNotFoundError(
                "Recognition model path is not configured. Set FLOW_ASSISTANT_OCR_RECOGNITION_MODEL"
            )
        self._ensure_model_files(config.recognition_model)
        bin_path = config.recognition_model.with_suffix(".bin")
        if not config.recognition_model.exists() or not bin_path.exists():
            raise FileNotFoundError(
                "OpenVINO OCR model files are missing even after attempted download."
            )

        self._alphabet = config.alphabet
        self._blank_id = config.blank_id
        self._core = ov.Core()
        model = self._core.read_model(str(config.recognition_model))
        self._compiled = self._core.compile_model(model, config.device)
        self._input = self._compiled.input(0)
        self._output = self._compiled.output(0)
        input_shape = list(self._input.shape)  # type: ignore[call-arg]
        if len(input_shape) != 4:
            raise ValueError(f"Unsupported recognition model input shape: {input_shape}")
        self._channels = input_shape[1]
        self._target_height = input_shape[2]
        self._target_width = input_shape[3]

    @staticmethod
    def _ensure_model_files(model_path: Path) -> None:
        xml_path = model_path
        bin_path = model_path.with_suffix(".bin")
        if xml_path.exists() and bin_path.exists():
            return
        xml_url = _DEFAULT_MODEL_URLS["xml"]
        bin_url = _DEFAULT_MODEL_URLS["bin"]
        if not xml_url or not bin_url:
            raise FileNotFoundError("OpenVINO OCR model URLs are not configured")
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not xml_path.exists():
                _download_file(xml_url, xml_path)
            if not bin_path.exists():
                _download_file(bin_url, bin_path)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.warning("Failed to download OpenVINO OCR model: %s", exc)
            if xml_path.exists():
                xml_path.unlink()
            if bin_path.exists():
                bin_path.unlink()
            raise

    def run(self, image) -> str:
        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
        prepared = self._prepare_input(pil_image)
        outputs = self._compiled({self._input: prepared})
        logits = outputs[self._output]
        return self._decode(logits).strip()

    def _prepare_input(self, image: "Image.Image") -> np.ndarray:
        if self._channels == 1:
            image = image.convert("L")
        else:
            image = image.convert("RGB")
        resized = image.resize((self._target_width, self._target_height))
        array = np.asarray(resized, dtype=np.float32)
        if self._channels == 1:
            array = array[np.newaxis, :, :]
        else:
            array = np.transpose(array, (2, 0, 1))  # HWC -> CHW
        array = array / 255.0
        array = array[np.newaxis, ...]  # add batch dimension
        return array

    def _decode(self, logits: np.ndarray) -> str:
        if logits.ndim == 3:
            token_ids = logits.argmax(axis=2)[0]
        elif logits.ndim == 2:
            token_ids = logits.argmax(axis=1)
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}")
        chars: list[str] = []
        prev = None
        for idx in token_ids:
            idx = int(idx)
            if idx == self._blank_id:
                prev = None
                continue
            if idx == prev:
                continue
            if 0 <= idx < len(self._alphabet):
                chars.append(self._alphabet[idx])
            else:
                logger.debug("Token id %s is outside of alphabet range", idx)
            prev = idx
        return "".join(chars)


_PIPELINE: Optional[OpenVINOTextRecognizer] = None
_PIPELINE_FAILED = False


def _download_file(url: str, destination: Path) -> None:
    logger.info("Downloading OpenVINO OCR model from %s", url)
    with urllib.request.urlopen(url) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise RuntimeError(f"Download failed with status {status}: {url}")
        data = response.read()
    if not data:
        raise RuntimeError(f"Downloaded file is empty: {url}")
    if destination.suffix.lower() == ".xml":
        snippet = data.lstrip()[:64]
        if not snippet.startswith(b"<?xml"):
            raise RuntimeError("Downloaded XML does not appear to be valid IR; received HTML/other content")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def _get_pipeline() -> Optional[OpenVINOTextRecognizer]:
    global _PIPELINE, _PIPELINE_FAILED
    if _PIPELINE_FAILED:
        return None
    if _PIPELINE is None:
        config = OpenVinoOCRConfig.from_env()
        try:
            _PIPELINE = OpenVINOTextRecognizer(config)
            logger.info("Loaded OpenVINO OCR model from %s", config.recognition_model)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.warning("OpenVINO OCR disabled: %s", exc)
            _PIPELINE_FAILED = True
            return None
    return _PIPELINE


def extract_text(image) -> str:
    """Extract text from the given image using the OpenVINO recognizer."""

    pipeline = _get_pipeline()
    if pipeline is None:
        return ""
    try:
        return pipeline.run(image)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.debug("OpenVINO OCR inference failed: %s", exc)
        return ""


_SENTENCE_BOUNDARY = re.compile(r"(?<=[\u3002\uFF0E\uFF01\uFF1F.!?])\s+")


def summarize_text(text: str, *, max_sentences: int = 2, max_chars: int = 200) -> str:
    """Produce a lightweight summary from OCR text."""

    cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not cleaned_lines:
        return ""
    cleaned = " ".join(cleaned_lines)
    sentences = _SENTENCE_BOUNDARY.split(cleaned)
    summary_parts: list[str] = []
    for sentence in sentences:
        if not sentence:
            continue
        summary_parts.append(sentence)
        current = " ".join(summary_parts)
        if len(summary_parts) >= max_sentences or len(current) >= max_chars:
            break
    summary = " ".join(summary_parts).strip()
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rstrip() + "..."
    return summary
