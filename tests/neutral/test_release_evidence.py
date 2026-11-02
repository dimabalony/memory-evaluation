import importlib.util
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("release_evidence", ROOT / "scripts/release_evidence.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


@pytest.mark.parametrize(
    "bad", ["kai_memory/__init__.py", "app/private.py", "../escape.py", "x.data/purelib/hidden.py"]
)
def test_refuse_unowned_or_indirect_wheel_payload(tmp_path, bad):
    wheel = tmp_path / "memory_evaluation-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("memory_evaluation/__init__.py", "")
        archive.writestr(bad, "")
    with pytest.raises(ValueError):
        evidence.installed_payloads(wheel)


def test_missing_archive_origin_cannot_attest_installation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evidence.importlib.metadata, "distribution", lambda _: SimpleNamespace(read_text=lambda _: None)
    )
    with pytest.raises(ValueError, match="provenance missing"):
        evidence.verify_archive_origin(tmp_path / "unused.whl")


def test_zero_skip_gate_requires_executed_cases(tmp_path):
    path = tmp_path / "postgres.xml"
    for body in ("", "<testcase><skipped/></testcase>", "<testcase><failure/></testcase>"):
        path.write_text(f"<testsuite>{body}</testsuite>")
        with pytest.raises(ValueError):
            evidence.junit(path, zero_skips=True)
