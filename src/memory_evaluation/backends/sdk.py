"""Recoleva adapter. The historical SDK lifecycle instrument remains canonical."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import kai_memory
from kai_memory.retrieval.memory_priming import MemoryPrimingService
from kai_memory.store.cognitive_memory_constants import DEFAULT_RETRIEVAL_WEIGHTS

from memory_evaluation.contracts import BackendDescriptor, Retrieval
from memory_evaluation.eval.disposable_db import DisposableDB
from memory_evaluation.eval.extractors import RecordedExtractor
from memory_evaluation.eval.providers import evaluation_embeddings
from memory_evaluation.eval.runner import _chat_context, _pinned_admins, _replay, run_write_readback, store_fingerprint
from memory_evaluation.eval.standalone import recorded_host
from memory_evaluation.runner import sha256


def _verify_installed_sdk(version: str, artifact_sha256: str) -> None:
    distribution = importlib.metadata.distribution("recoleva")
    if distribution.version != version:
        raise ValueError("installed SDK version differs from requested version")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    archive = origin.get("archive_info", {})
    digest = archive.get("hashes", {}).get("sha256")
    if digest is None and archive.get("hash", "").startswith("sha256="):
        digest = archive["hash"].removeprefix("sha256=")
    if digest != artifact_sha256:
        raise ValueError("installed SDK archive differs from requested artifact")
    package = Path(kai_memory.__file__).resolve().parent
    installed_root = Path(distribution.locate_file("kai_memory")).resolve()
    if (
        package != installed_root
        or "site-packages" not in package.parts
        or not package.is_relative_to(Path(sys.prefix).resolve())
    ):
        raise ValueError("SDK import is shadowed or outside this interpreter's installation")
    for name, module in tuple(sys.modules.items()):
        if name == "kai_memory" or name.startswith("kai_memory."):
            origin = vars(module).get("__file__") if module is not None else None
            if origin and not Path(origin).resolve().is_relative_to(package):
                raise ValueError("SDK submodule import is shadowed")
    recorded = set()
    for entry in distribution.files or ():
        name = str(entry)
        if not name.startswith("kai_memory/") or "__pycache__" in Path(name).parts:
            continue
        path = Path(distribution.locate_file(entry)).resolve()
        if not path.is_relative_to(package) or entry.hash is None or entry.hash.mode != "sha256":
            raise ValueError("SDK payload has no verifiable installation record")
        actual = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b"=").decode()
        if actual != entry.hash.value:
            raise ValueError("installed SDK payload differs from installation record")
        recorded.add(path)
    actual_paths = {p.resolve() for p in package.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    if not recorded or actual_paths != recorded:
        raise ValueError("SDK payload inventory differs from installation record")


class SDKBackend:
    """Use an already bootstrapped disposable local database and exact SDK artifact."""

    def __init__(self, db: DisposableDB, *, artifact_sha256: str, version: str = "0.3.0"):
        if re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None:
            raise ValueError("exact SDK wheel SHA-256 is required")
        _verify_installed_sdk(version, artifact_sha256)
        self.db = db
        self.artifact_sha256 = artifact_sha256
        self.version = version

    def describe(self):
        return BackendDescriptor(
            "recoleva",
            self.version,
            self.artifact_sha256,
            ("read_to_answer", "sdk_lifecycle"),
            {"extractor": "recorded", "retrieval": "last-message"},
        )

    @asynccontextmanager
    async def open_fixture(self, fixture, config):
        now = datetime.fromisoformat(config.clock)
        if now.tzinfo is None:
            raise ValueError("fixture clock must include a timezone")
        with recorded_host(now), evaluation_embeddings(), _pinned_admins(fixture):
            async with self.db.session() as session:
                result = {
                    "id": fixture["id"],
                    "status": "ok",
                    "error": None,
                    "labels": {},
                    "checkpoints": [],
                    "prompt_checks": [],
                    "snapshot": None,
                    "readback": [],
                    "messages_saved": 0,
                    "fail_fast": True,
                }
                await _replay(session, fixture, RecordedExtractor(), result)
                if any(cp.get("store_errors") for cp in result["checkpoints"]):
                    raise RuntimeError("SDK replay failed")
                yield _SDKSession(session, fixture)

    async def write_readback(self, fixture_set, **kwargs):
        """Return the unmodified historical raw report; caller binds SDK host context."""
        return await run_write_readback(fixture_set, self.db, **kwargs)


class _SDKSession:
    def __init__(self, session, fixture):
        self.session, self.fixture = session, fixture

    async def state_receipt(self):
        return sha256(await store_fingerprint(self.session))

    async def retrieve(self, request, question):
        participants = {p["key"]: p for p in self.fixture.get("participants") or []}
        asker = participants.get(question.get("asker"))
        priming = MemoryPrimingService(self.session, weights_override=DEFAULT_RETRIEVAL_WEIGHTS)
        context, ids = await priming.prime(
            request.question,
            limit=question.get("k", 8),
            chat_ctx=_chat_context(self.fixture, asker),
            record_access=False,
        )
        return Retrieval(context, tuple(str(i) for i in ids))
