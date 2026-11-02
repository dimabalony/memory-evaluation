"""The memory eval's deterministic embedder, bound through the embeddings port (DB-840, DB-843)."""

from __future__ import annotations

import asyncio
import math

from kai_memory import ports
from memory_evaluation.eval.providers import (
    EMBED_DIM,
    TokenHashEmbedder,
    deterministic_embeddings,
    token_hash_embedding,
)
from kai_memory.store import cognitive_memory, embeddings


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_token_hash_embedding_is_deterministic_unit_and_never_zero():
    a = token_hash_embedding("Teodora works at Quillmoor Labs")
    assert a == token_hash_embedding("Teodora works at Quillmoor Labs")
    assert len(a) == EMBED_DIM
    assert math.isclose(math.sqrt(sum(v * v for v in a)), 1.0, rel_tol=1e-9)
    for empty in ("", "a b", "🙂"):
        assert any(token_hash_embedding(empty))


def test_token_hash_embedding_tracks_shared_words():
    stored = token_hash_embedding("Teodora Valk started at Quillmoor Labs as a supply chain analyst")
    related = token_hash_embedding("Where does Teodora Valk work now? Quillmoor?")
    unrelated = token_hash_embedding("the boiler engineer arrives on Wednesday morning")
    assert _cosine(stored, related) > _cosine(stored, unrelated)


def test_the_block_binds_the_embedder_port_and_restores_the_previous_binding(monkeypatch):
    monkeypatch.setattr(ports, "_BOUND", {})
    default = ports.embedder
    with deterministic_embeddings() as name:
        assert name == "token-hash-v1"
        assert isinstance(ports.embedder, TokenHashEmbedder)
        # Memory's own embedding functions, as every caller imported them, now use it.
        vec = asyncio.run(cognitive_memory.generate_embedding("hello there"))
        assert vec == token_hash_embedding("hello there")
        batch = asyncio.run(embeddings.generate_embeddings_batch(["a cat", "a dog"]))
        assert batch == [token_hash_embedding("a cat"), token_hash_embedding("a dog")]
    assert ports._BOUND == {}
    assert ports.embedder is default


def test_an_outer_binding_survives_the_block(monkeypatch):
    outer = TokenHashEmbedder()
    monkeypatch.setattr(ports, "_BOUND", {"embedder": outer})
    with deterministic_embeddings():
        assert ports.embedder is not outer
    assert ports.embedder is outer
