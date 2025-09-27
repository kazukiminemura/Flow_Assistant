"""Utilities supporting Flow Assistant modules."""

from __future__ import annotations

import hashlib
import json
import os
import random
import string
import time
from typing import Iterable, Sequence


_ID_ALPHABET = string.ascii_lowercase + string.digits


def generate_id(prefix: str, *, size: int = 8) -> str:
    """Generate a short unique identifier with a readable prefix."""

    suffix = "".join(random.choice(_ID_ALPHABET) for _ in range(size))
    return f"{prefix}_{suffix}"


def stable_hash(text: str, *, length: int = 16) -> Sequence[float]:
    """Produce deterministic pseudo-embedding from text using hashing.

    The embedding is not semantically meaningful, but gives us a stable vector
    for approximate similarity ranking that runs entirely on-device.
    """

    if length <= 0:
        raise ValueError("length must be positive")

    # Use SHA256 to seed deterministic pseudo-random projections.
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    floats = []
    for idx in range(length):
        # Combine digest bytes into signed values between -1 and 1.
        b1 = digest[idx % len(digest)]
        b2 = digest[(idx * 7 + 13) % len(digest)]
        val = ((b1 << 8) + b2) % 2001
        floats.append((val - 1000) / 1000)
    return floats


def timestamp_ms() -> int:
    """Return current UTC timestamp in milliseconds."""

    return int(time.time() * 1000)


def serialize_vector(vector: Sequence[float]) -> str:
    """Serialize vector to JSON for persistence."""

    return json.dumps(list(vector), ensure_ascii=True)


def deserialize_vector(serialized: str) -> Sequence[float]:
    """Read vector from stored JSON text."""

    return json.loads(serialized)
