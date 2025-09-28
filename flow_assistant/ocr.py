"""OCR helpers for extracting text from window captures using OpenVINO."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from shutil import which

try:  # pragma: no cover - optional dependency
    import openvino.runtime as ov  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ov = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from PIL import Image, ImageOps  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore

logger = logging.getLogger(__name__)

_DEBUG_OCR_OUTPUT = os.getenv("FLOW_ASSISTANT_OCR_DEBUG", "").strip().lower() in {"1", "true", "yes"}

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
        if _download_with_omz(xml_path):
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
    image = image.convert("L") if self._channels == 1 else image.convert("RGB")
    width, height = image.size
    if width == 0 or height == 0:
        width, height = self._target_width, self._target_height
    scale = self._target_height / height
    new_width = max(1, int(round(width * scale)))
    if new_width > self._target_width:
        new_width = self._target_width
    resized = image.resize((new_width, self._target_height), Image.BICUBIC)
    if self._channels == 1:
        canvas = Image.new("L", (self._target_width, self._target_height), color=255)
    else:
        canvas = Image.new("RGB", (self._target_width, self._target_height), color=(255, 255, 255))
    canvas.paste(resized, (0, 0))
    array = np.asarray(canvas, dtype=np.float32)
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


def _download_with_omz(model_path: Path) -> bool:
    downloader = which("omz_downloader")
    if not downloader:
        return False
    precision = os.getenv("FLOW_ASSISTANT_OCR_MODEL_PRECISION", "FP16")
    model_name = os.getenv("FLOW_ASSISTANT_OCR_MODEL_NAME", "text-recognition-0014")
    output_root = model_path.parent
    command = [
        downloader,
        "--name",
        model_name,
        "--precision",
        precision,
        "--output_dir",
        str(output_root),
    ]
    logger.info("Running omz_downloader to fetch %s", model_name)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime safeguard
        logger.warning("omz_downloader failed: %s", exc.stderr.strip())
        return False
    xml_candidate = next((c for c in output_root.rglob(f"{model_name}.xml")), None)
    if not xml_candidate:
        logger.warning("omz_downloader did not produce expected IR files")
        return False
    bin_candidate = xml_candidate.with_suffix(".bin")
    if not bin_candidate.exists():
        logger.warning("omz_downloader result missing BIN file")
        return False
    target_xml = model_path
    target_bin = model_path.with_suffix(".bin")
    target_xml.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(xml_candidate, target_xml)
    shutil.copy2(bin_candidate, target_bin)
    logger.info("Copied OpenVINO OCR model to %s", target_xml)
    return True


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
            raise RuntimeError(
                "Downloaded XML does not appear to be valid IR; received HTML/other content"
            )
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
    if pipeline is None or Image is None:
        return ""
    if isinstance(image, Image.Image):
        pil_image = image
    else:
        pil_image = Image.fromarray(np.asarray(image))
    pil_image = ImageOps.autocontrast(pil_image)
    segments = _segment_text_lines(pil_image)
    if not segments:
        segments = [pil_image]
    decoded_lines: list[str] = []
    for idx, segment in enumerate(segments):
        try:
            text_line = pipeline.run(ImageOps.autocontrast(segment))
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.debug("OpenVINO OCR inference failed on segment %s: %s", idx, exc)
            continue
        if not text_line:
            continue
        decoded_lines.append(text_line)
        if _DEBUG_OCR_OUTPUT:
            print(f"[OCR DEBUG] line {idx}: {text_line}")
    if not decoded_lines:
        try:
            fallback = pipeline.run(ImageOps.autocontrast(pil_image))
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.debug("OpenVINO OCR fallback failed: %s", exc)
            fallback = ""
        if fallback:
            decoded_lines.append(fallback)
    if _DEBUG_OCR_OUTPUT and not decoded_lines:
        print("[OCR DEBUG] no text detected")
    return "\n".join(decoded_lines)


_DEF_MIN_LINE_HEIGHT = 12
_DEF_LINE_MARGIN = 4


def _segment_text_lines(image: "Image.Image") -> list["Image.Image"]:
    gray = np.asarray(ImageOps.autocontrast(image.convert("L")), dtype=np.float32)
    if gray.size == 0:
        return []
    inverted = 255.0 - gray
    row_profile = inverted.mean(axis=1)
    if not np.any(row_profile):
        return []
    threshold = max(row_profile.mean() * 0.5, row_profile.max() * 0.15)
    mask = row_profile > threshold
    min_height = max(int(gray.shape[0] * 0.04), _DEF_MIN_LINE_HEIGHT)
    margin = max(int(gray.shape[0] * 0.02), _DEF_LINE_MARGIN)
    segments: list[Image.Image] = []
    start = None
    for idx, val in enumerate(mask):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            if idx - start >= min_height:
                top = max(0, start - margin)
                bottom = min(gray.shape[0], idx + margin)
                segments.append(_trim_horizontal_margins(image.crop((0, top, image.width, bottom))))
            start = None
    if start is not None and len(mask) - start >= min_height:
        top = max(0, start - margin)
        bottom = len(mask)
        segments.append(_trim_horizontal_margins(image.crop((0, top, image.width, bottom))))
    filtered = [seg for seg in segments if seg.height >= max(min_height // 2, 8) and seg.width > 8]
    return filtered


def _trim_horizontal_margins(image: "Image.Image") -> "Image.Image":
    gray = np.asarray(ImageOps.autocontrast(image.convert("L")), dtype=np.float32)
    if gray.size == 0:
        return image
    inverted = 255.0 - gray
    col_profile = inverted.mean(axis=0)
    if not np.any(col_profile):
        return image
    threshold = max(col_profile.mean() * 0.5, col_profile.max() * 0.15)
    mask = col_profile > threshold
    if not np.any(mask):
        return image
    indices = np.where(mask)[0]
    margin = max(int(image.width * 0.02), 2)
    left = max(0, int(indices[0]) - margin)
    right = min(image.width, int(indices[-1]) + margin + 1)
    if right - left < 8:
        return image
    return image.crop((left, 0, right, image.height))


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

