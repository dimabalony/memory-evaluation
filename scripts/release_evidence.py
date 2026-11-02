"""Bind release artifacts, installed payloads and executed deterministic proof."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def junit(path, *, zero_skips=False):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    result = {
        "tests": len(cases),
        **{
            key: sum(c.find(tag) is not None for c in cases)
            for key, tag in (
                ("skipped", "skipped"),
                ("failures", "failure"),
                ("errors", "error"),
            )
        },
    }
    result["xfailed"] = sum(
        c.find("skipped") is not None and c.find("skipped").get("type") == "pytest.xfail" for c in cases
    )
    result["skipped"] -= result["xfailed"]
    if not cases or result["failures"] or result["errors"] or (zero_skips and (result["skipped"] or result["xfailed"])):
        raise ValueError(f"incomplete or failing proof: {path.name}: {result}")
    result["junit_sha256"] = sha256(path)
    return result


def installed_payloads(wheel, distribution="memory_evaluation", namespace="memory_evaluation"):
    match = re.fullmatch(rf"{re.escape(distribution)}-([A-Za-z0-9_.!+]+)-py3-none-any\.whl", wheel.name)
    if not match:
        raise ValueError("unreviewed wheel filename")
    metadata = f"{distribution}-{match.group(1)}.dist-info"
    with zipfile.ZipFile(wheel) as archive:
        if any(stat.S_ISLNK(i.external_attr >> 16) for i in archive.infolist()):
            raise ValueError("symlink wheel payload")
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("duplicate wheel files")
        for name in names:
            parsed = PurePosixPath(name)
            if (
                str(parsed) != name
                or parsed.is_absolute()
                or ".." in parsed.parts
                or any(p.endswith(".data") for p in parsed.parts)
                or any(p.endswith(".dist-info") for p in parsed.parts[1:])
                or "\\" in name
            ):
                raise ValueError("unsafe wheel path")
            if not name.startswith(namespace + "/") and (
                len(parsed.parts) < 2
                or parsed.parts[0] != metadata
                or any(p.endswith(".dist-info") for p in parsed.parts[1:])
            ):
                raise ValueError("unreviewed installation payload")
        expected = {n: hashlib.sha256(archive.read(n)).hexdigest() for n in names if n.startswith(namespace + "/")}
    if not expected or namespace + "/__init__.py" not in expected:
        raise ValueError("missing SDK payload")
    module = __import__(namespace)
    package = Path(module.__file__).resolve().parent
    if "site-packages" not in package.parts or not package.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("SDK must load from this interpreter's site-packages")
    actual = {
        namespace + "/" + p.relative_to(package).as_posix(): sha256(p)
        for p in package.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if actual != expected:
        raise ValueError("installed payload differs from exact wheel")
    return expected


def verify_archive_origin(wheel, distribution="memory-evaluation"):
    raw = importlib.metadata.distribution(distribution).read_text("direct_url.json")
    if not raw:
        raise ValueError("installed archive provenance missing")
    origin = json.loads(raw)
    archive = origin.get("archive_info", {})
    actual = archive.get("hashes", {}).get("sha256")
    if actual is None and archive.get("hash", "").startswith("sha256="):
        actual = archive["hash"].removeprefix("sha256=")
    if actual != sha256(wheel):
        raise ValueError("installed archive hash differs from proved wheel")
    return {"archive_sha256": actual, "verified": True}


def build_evidence(dist, reports, source_commit, sdk_wheel):
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be a full Git SHA")
    (wheel,) = dist.glob("*.whl")
    (sdist,) = dist.glob("*.tar.gz")
    version = importlib.metadata.version("memory-evaluation")
    if (
        wheel.name != f"memory_evaluation-{version}-py3-none-any.whl"
        or sdist.name != f"memory_evaluation-{version}.tar.gz"
    ):
        raise ValueError("artifact names do not match installed version")
    sdk = json.loads((Path(__file__).resolve().parents[1] / "sdk-artifact.json").read_text())
    if sha256(sdk_wheel) != sdk["sha256"]:
        raise ValueError("SDK wheel differs from pinned release")
    neutral = json.loads((reports / "neutral-environment.json").read_text())
    if neutral["archive"]["archive_sha256"] != sha256(wheel) or neutral["forbidden_available"]:
        raise ValueError("neutral proof used a different wheel or available SDK/ORM")
    payloads = installed_payloads(wheel)
    if neutral["installed_payloads"] != payloads:
        raise ValueError("neutral and SDK environments installed different evaluation payloads")
    scenarios = {}
    for name in (
        "lifecycle-raw.json", "lifecycle-scoreboard.json", "lifecycle-mutation.json",
        "lifecycle-parity.json", "lifecycle-retrieval-scores.json",
    ):
        path = reports / name
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or not value:
            raise ValueError("missing lifecycle evidence")
        if name == "lifecycle-parity.json":
            root = Path(__file__).resolve().parents[1]
            contract_path = root / "tests/baselines/lifecycle-isolated-v2.json"
            contract = json.loads(contract_path.read_text())
            if (
                value.get("isolated_baseline_passed") is not True
                or value.get("historical_exact_parity") is not False
                or value.get("semantic_candidates_complete") is not True
                or value.get("historical_differences") != contract["differences"]
                or value.get("instrument_contract_sha256") != sha256(contract_path)
                or value.get("baseline_sha256") != contract["historical_baseline_sha256"]
            ):
                raise ValueError("isolated lifecycle proof or declared historical difference changed")
        if name == "lifecycle-mutation.json" and (value["residual_rows"] or value["changed_tables"]):
            raise ValueError("lifecycle left residual data")
        scenarios[name] = sha256(path)
    return {
        "scenarios": scenarios,
        "format": "memory-evaluation-release/v1",
        "distribution": "memory-evaluation",
        "version": version,
        "source_commit": source_commit,
        "python_version": platform.python_version(),
        "ci_run_url": f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        if os.environ.get("GITHUB_REPOSITORY") and os.environ.get("GITHUB_RUN_ID")
        else None,
        "artifacts": {
            kind: {"filename": p.name, "sha256": sha256(p)} for kind, p in (("wheel", wheel), ("sdist", sdist))
        },
        "installed_payloads": payloads,
        "installed_archive": verify_archive_origin(wheel),
        "sdk": {
            **sdk,
            "installed_archive": verify_archive_origin(sdk_wheel, "recoleva"),
            "installed_payloads": installed_payloads(sdk_wheel, "recoleva", "kai_memory"),
        },
        "tests": {
            name: junit(reports / filename, zero_skips=True)
            for name, filename in (("portable", "unit.xml"), ("neutral", "neutral.xml"), ("postgres", "postgres.xml"))
        },
        "provider_calls": 0,
        "claims": "deterministic instrument and adapter proof; model quality and learning efficacy unmeasured",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--sdk-wheel", type=Path)
    parser.add_argument("--neutral", action="store_true")
    args = parser.parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    if args.neutral:
        import importlib.util

        (wheel,) = args.dist.glob("*.whl")
        forbidden = [name for name in ("kai_memory", "app", "sqlalchemy") if importlib.util.find_spec(name)]
        if forbidden:
            raise ValueError("neutral environment includes forbidden dependencies")
        report = {
            "forbidden_available": forbidden,
            "archive": verify_archive_origin(wheel),
            "installed_payloads": installed_payloads(wheel),
        }
        (args.reports / "neutral-environment.json").write_text(json.dumps(report, indent=2) + "\n")
        return
    if args.sdk_wheel is None or args.source_commit is None:
        parser.error("--sdk-wheel and --source-commit required for full proof")
    evidence = build_evidence(args.dist, args.reports, args.source_commit, args.sdk_wheel)
    receipt_archive = args.reports / "proof-receipts.tar.gz"
    with tarfile.open(receipt_archive, "w:gz") as archive:
        for name in ("unit.xml", "neutral.xml", "postgres.xml", "neutral-environment.json", *evidence["scenarios"]):
            archive.add(args.reports / name, arcname=name)
    evidence["proof_receipts"] = {"filename": receipt_archive.name, "sha256": sha256(receipt_archive)}
    (args.reports / "release.json").write_text(json.dumps(evidence, indent=2) + "\n")
    artifacts = [args.dist / v["filename"] for v in evidence["artifacts"].values()] + [receipt_archive]
    (args.reports / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in artifacts))


if __name__ == "__main__":
    main()
