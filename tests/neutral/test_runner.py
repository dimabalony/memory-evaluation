import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from dataclasses import replace

from memory_evaluation.backends.no_memory import NoMemoryBackend
from memory_evaluation.contracts import BackendDescriptor, Retrieval, RunConfig
from memory_evaluation.eval.fixtures import FixtureSet
from memory_evaluation.runner import evaluate

CONFIG = RunConfig("revision", "2026-01-01T00:00:00+00:00", 42, {"model": "recorded", "prompt_version": "test-v1"})
FIXTURE = {
    "id": "extension-example",
    "participants": [{"key": "user", "name": "Ada"}],
    "messages": [{"from": "user", "text": "The archive city is Lima."}],
    "expected": {"questions": [{"id": "city", "question": "Which city?", "history": "none", "expect_exact": "Lima"}]},
}
FIXTURES = FixtureSet("synthetic-test", [FIXTURE], "0" * 64)


class DictionaryBackend:
    """An external implementation: no SDK, ORM, or evaluation internals."""

    def describe(self):
        return BackendDescriptor("dictionary-example", "1")

    @asynccontextmanager
    async def open_fixture(self, fixture, config):
        self.value = fixture["messages"][0]["text"].split()[-1].rstrip(".")
        yield self

    async def retrieve(self, request, question):
        return Retrieval(self.value, ("example-row",))

    async def state_receipt(self):
        return self.value


async def answer(request):
    return request.memory_context or "I don't know."


async def test_same_runner_executes_independent_backends():
    absent = await evaluate(FIXTURES, NoMemoryBackend(), answer_fn=answer, config=CONFIG)
    present = await evaluate(FIXTURES, DictionaryBackend(), answer_fn=answer, config=CONFIG)
    assert absent["accuracy"] == 0
    assert present["accuracy"] == 1
    assert absent["rows"][0]["input_sha256"] == present["rows"][0]["input_sha256"]
    assert absent["write_readback"]["status"] == "unmeasured"
    assert absent["denominators"] == {"valid": 1, "invalid": 0, "unrun": 0, "unmeasured": 0}


async def test_failure_is_invalid_never_perfect():
    async def failure(_):
        raise RuntimeError("private content should not reach report")

    report = await evaluate(FIXTURES, NoMemoryBackend(), answer_fn=failure, config=CONFIG)
    assert report["accuracy"] is None
    assert report["denominators"]["invalid"] == 1
    assert "private content" not in json.dumps(report)


async def test_mutating_retrieval_invalidates_row():
    class Mutating(DictionaryBackend):
        async def retrieve(self, request, question):
            self.value = "changed"
            return Retrieval(self.value)

    report = await evaluate(FIXTURES, Mutating(), answer_fn=answer, config=CONFIG)
    assert report["denominators"]["invalid"] == 1
    assert report["accuracy"] is None


async def test_open_failure_tracks_unrun():
    class Broken(NoMemoryBackend):
        @asynccontextmanager
        async def open_fixture(self, fixture, config):
            raise RuntimeError("unavailable")
            yield self

    report = await evaluate(FIXTURES, Broken(), answer_fn=answer, config=CONFIG)
    assert report["denominators"]["unrun"] == 1
    assert report["accuracy"] is None


async def test_unsupported_instrument_is_unmeasured():
    class Unsupported(NoMemoryBackend):
        def describe(self):
            return BackendDescriptor("unsupported", "1", capabilities=())

    report = await evaluate(FIXTURES, Unsupported(), answer_fn=answer, config=CONFIG)
    assert report["denominators"]["unmeasured"] == 1
    assert report["accuracy"] is None


async def test_ambiguous_judge_is_invalid():
    async def judge(_):
        return "VERDICT: correct VERDICT: incorrect"

    report = await evaluate(
        FIXTURES, DictionaryBackend(), answer_fn=answer, judge_fn=judge, config=replace(CONFIG, judge={"model": "test"})
    )
    assert report["denominators"]["invalid"] == 1


def test_neutral_runtime_with_sdk_and_sqlalchemy_blocked():
    script = """
import importlib.abc, sys, asyncio
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'kai_memory', 'app', 'sqlalchemy'}:
            raise ImportError('blocked ' + fullname)
sys.meta_path.insert(0, Block())
from tests.neutral.test_runner import test_same_runner_executes_independent_backends
from memory_evaluation.elephant_scorer import build_scoring_prompt
from memory_evaluation.eval import load_fixture_set
asyncio.run(test_same_runner_executes_independent_backends())
assert 'USER PROMPT' in build_scoring_prompt('q', 'a', canonical=True)
assert not {'kai_memory', 'sqlalchemy', 'app'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())
