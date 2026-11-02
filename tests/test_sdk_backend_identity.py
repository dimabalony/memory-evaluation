import importlib.metadata
import json
from types import SimpleNamespace

import pytest

from memory_evaluation.backends import sdk
from memory_evaluation.eval.disposable_db import DisposableDB


def installed_identity():
    distribution = importlib.metadata.distribution("recoleva")
    origin = json.loads(distribution.read_text("direct_url.json"))
    return distribution, origin["archive_info"].get("hashes", {}).get("sha256") or origin["archive_info"][
        "hash"
    ].removeprefix("sha256=")


def backend(digest, version="0.3.0"):
    return sdk.SDKBackend(
        DisposableDB("postgresql+asyncpg://postgres@localhost/evaluation", production_url=None),
        artifact_sha256=digest,
        version=version,
    )


def test_reject_false_archive_and_version_claims():
    _, digest = installed_identity()
    with pytest.raises(ValueError, match="archive differs"):
        backend("0" * 64)
    with pytest.raises(ValueError, match="version differs"):
        backend(digest, "9.9.9")
    assert backend(digest).describe().artifact_sha256 == digest


def test_reject_source_shadow(monkeypatch, tmp_path):
    _, digest = installed_identity()
    monkeypatch.setattr(sdk.kai_memory, "__file__", str(tmp_path / "kai_memory/__init__.py"))
    with pytest.raises(ValueError, match="shadowed"):
        backend(digest)


def test_reject_modified_payload_record(monkeypatch):
    distribution, digest = installed_identity()
    files = list(distribution.files)
    index = next(i for i, p in enumerate(files) if str(p) == "kai_memory/__init__.py")
    changed = importlib.metadata.PackagePath(str(files[index]))
    changed.hash = importlib.metadata.FileHash("sha256=" + "A" * 43)
    files[index] = changed
    wrapper = SimpleNamespace(
        version=distribution.version,
        files=files,
        read_text=distribution.read_text,
        locate_file=distribution.locate_file,
    )
    monkeypatch.setattr(sdk.importlib.metadata, "distribution", lambda _: wrapper)
    with pytest.raises(ValueError, match="payload differs"):
        backend(digest)


def test_reject_shadowed_submodule(monkeypatch, tmp_path):
    _, digest = installed_identity()
    monkeypatch.setitem(sdk.sys.modules, "kai_memory.injected", SimpleNamespace(__file__=str(tmp_path / "outside.py")))
    with pytest.raises(ValueError, match="submodule import is shadowed"):
        backend(digest)
