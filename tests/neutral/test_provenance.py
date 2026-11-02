import ast
import hashlib
import json
from pathlib import Path

import memory_evaluation
from memory_evaluation import elephant_scorer

ROOT = Path(__file__).resolve().parents[2]


def test_fixtures_exactly_match_licensed_inventory():
    receipt = json.loads((ROOT / "FIXTURE_LICENSES.json").read_text())
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / "tests/fixtures").rglob("*.json")}
    assert actual == {row["path"] for row in receipt["fixtures"]}
    for row in receipt["fixtures"]:
        assert row["license"] == "GPL-3.0-or-later"
        assert row["sha256"] == hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest()


def test_canonical_instrument_functions_and_prompts_preserved():
    receipt = json.loads((ROOT / "CANONICAL_INSTRUMENTS.json").read_text())
    package = Path(memory_evaluation.__file__).parent
    for row in receipt["functions"]:
        node = next(
            n
            for n in ast.parse((package / row["module"]).read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == row["function"]
        )
        assert row["ast_sha256"] == hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    for name, digest in receipt["prompts_sha256"].items():
        assert hashlib.sha256(getattr(elephant_scorer, name).encode()).hexdigest() == digest
