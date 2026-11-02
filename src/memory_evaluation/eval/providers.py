"""Deterministic embeddings for the memory eval, bound through the embeddings port (DB-840, DB-843).

Memory embeds through ``ports.embedder``, which it reads on every call, so the eval
binds a token-hash embedder for the run and puts the previous binding back after.
No module attribute is patched, and nothing reaches the network: the default OpenAI
adapter is simply never asked.

The token-hash embedder is lexical, not semantic: cosine similarity tracks shared
words. That is enough for read-after-write checks whose cue shares words with the
stored fact, and it is reproducible byte for byte.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from kai_memory import ports

EMBEDDER_NAME = "token-hash-v1"
EMBED_DIM = 1536

_TOKEN = re.compile(r"\w+", re.UNICODE)


def token_hash_embedding(text: str) -> list[float]:
    """A unit vector from signed token hashes; never the zero vector."""
    vec = [0.0] * EMBED_DIM
    tokens = [t for t in _TOKEN.findall(text.casefold()) if len(t) >= 3]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBED_DIM
        vec[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        # No usable token (or the hashes cancelled out): a fixed unit vector keeps
        # the store's zero-vector guard from turning this into a NULL embedding.
        vec[0], norm = 1.0, 1.0
    return [v / norm for v in vec]


class TokenHashEmbedder:
    """The embeddings port, deterministically: one token-hash vector per text."""

    name = EMBEDDER_NAME
    timeout_s = None
    max_retries = 0

    async def embed(self, texts: list[str], *, api_key: str | None = None) -> list[list[float]]:
        return [token_hash_embedding(t) for t in texts]


@contextmanager
def deterministic_embeddings() -> Iterator[str]:
    """Bind the token-hash embedder for the block; yields its name."""
    previous = ports.bind(embedder=TokenHashEmbedder())
    try:
        yield EMBEDDER_NAME
    finally:
        ports.bind(**previous)


@contextmanager
def evaluation_embeddings(adapter: Any = None, descriptor: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Explicitly bind one adapter for the whole replay; never downgrade an injected adapter."""
    if adapter is None:
        adapter = TokenHashEmbedder()
        descriptor = {"name": EMBEDDER_NAME, "kind": "lexical", "dimension": EMBED_DIM}
    elif not descriptor or descriptor.get("kind") not in {"lexical", "real"} or not descriptor.get("name"):
        raise ValueError("injected embedding adapter requires name and lexical/real kind descriptor")
    with ports.scoped({"embedder": adapter}, inherit=True):
        yield dict(descriptor)
