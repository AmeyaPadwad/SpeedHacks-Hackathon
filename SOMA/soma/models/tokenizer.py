"""Deterministic hashed word-piece tokenizer for OCR text and memory summaries.

No vocabulary file to ship or leak; IDs are stable across runs (blake2b, not
Python's salted ``hash``). ID 0 is PAD, ID 1 is UNK/empty.
"""

from __future__ import annotations

import hashlib
import re

PAD_ID = 0
UNK_ID = 1
_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def token_id(token: str, vocab_size: int) -> int:
    h = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")
    return 2 + h % (vocab_size - 2)


def encode(text: str, vocab_size: int, max_len: int | None = None) -> list[int]:
    ids = [token_id(t, vocab_size) for t in tokenize(text)]
    if max_len is not None:
        ids = ids[:max_len]
    return ids


def hash_bucket(name: str, n_buckets: int) -> int:
    """Stable bucket for categorical strings such as app names."""
    h = int.from_bytes(hashlib.blake2b((name or "").lower().encode("utf-8"), digest_size=8).digest(), "little")
    return h % n_buckets
