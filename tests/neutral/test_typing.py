"""External consumers must see declared types from the installed release wheel."""

import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

import memory_evaluation

ROOT = Path(__file__).resolve().parents[2]


def test_typing_marker_survives_source_and_wheel_packaging():
    distribution = importlib.metadata.distribution("memory-evaluation")
    assert "memory_evaluation/py.typed" in {str(path) for path in distribution.files}
    package = Path(memory_evaluation.__file__).resolve().parent
    assert "site-packages" in package.parts
    assert (package / "py.typed").read_bytes() == b""
    assert (ROOT / "src/memory_evaluation/py.typed").read_bytes() == b""


def test_external_typed_consumer_and_invalid_calls(tmp_path):
    (tmp_path / "facade.py").write_text(
        "from memory_evaluation.benchmark_io import *\n"
        "from memory_evaluation.eval import *\n"
        "from memory_evaluation.elephant_scorer import *\n"
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "from pathlib import Path\n"
        "from typing import Any, assert_type\n"
        "from facade import dataset_version, load_fixture_set, select_fixture_subset, parse_elephant_response\n"
        "from memory_evaluation.eval.fixtures import FixtureSet\n"
        "from memory_evaluation.elephant_scorer import ElephantScore\n"
        "assert_type(dataset_version(Path('fixture.json')), int | str)\n"
        "assert_type(load_fixture_set(Path('fixtures')), FixtureSet)\n"
        "assert_type(parse_elephant_response('{}'), ElephantScore)\n"
        "assert_type(select_fixture_subset([], None, None), list[dict[str, Any]])\n"
        "def version(path: Path) -> int | str:\n"
        "    return dataset_version(path)\n"
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("MYPYPATH", None)
    command = [sys.executable, "-m", "mypy", "--strict", "--no-incremental", "consumer.py"]
    valid = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert valid.returncode == 0, valid.stdout + valid.stderr
    consumer.write_text(consumer.read_text() + "dataset_version('not-a-Path')\nparse_elephant_response(123)\n")
    invalid = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert invalid.returncode == 1, invalid.stdout + invalid.stderr
    assert invalid.stdout.count("[arg-type]") == 2, invalid.stdout
    assert "import-untyped" not in invalid.stdout
