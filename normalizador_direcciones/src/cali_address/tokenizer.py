"""Character tokenizer of the address encoder (numpy only, no torch).

Lives apart from :mod:`cali_address.model` so that the ONNX serving backend can tokenize without
importing torch. ``model`` re-exports every name, so existing imports keep working.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CHARS", "VOCAB", "VOCAB_SIZE", "PAD", "UNK", "MAX_LEN", "encode_text", "encode_batch"]

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 #-.,/'()|"
PAD = 0
UNK = 1
VOCAB = {ch: i + 2 for i, ch in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS) + 2
MAX_LEN = 48

_TABLE = np.full(256, UNK, dtype=np.uint8)
for _ch, _i in VOCAB.items():
    _TABLE[ord(_ch)] = _i


def encode_text(text: str, max_len: int = MAX_LEN) -> np.ndarray:
    """Encode one already-uppercased ASCII string into a fixed-length id array."""
    out = np.zeros(max_len, dtype=np.uint8)
    b = text.encode("ascii", "replace")[:max_len]
    if b:
        arr = np.frombuffer(b, dtype=np.uint8)
        out[: len(arr)] = _TABLE[arr]
    return out


def encode_batch(texts, max_len: int = MAX_LEN) -> np.ndarray:
    out = np.zeros((len(texts), max_len), dtype=np.uint8)
    for i, t in enumerate(texts):
        out[i] = encode_text(t, max_len)
    return out
