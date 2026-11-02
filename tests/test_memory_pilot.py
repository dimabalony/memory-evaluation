"""Offline-only DB-881 approval, monotone budget and injected transport proofs."""

from __future__ import annotations

import io
import json
import sys
from types import ModuleType
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_evaluation.eval.answering import PAIR_HASHES, check_answer
from memory_evaluation.eval.budget import PilotBudgetExceeded, PriceCeiling, SharedPilotBudget
from memory_evaluation.eval.fixtures import FixtureSet, load_fixture_set
from memory_evaluation.eval.pilot import (
    DEFAULT_IDS,
    EMBED_MODEL,
    LLM_MODEL,
    PILOT_ARMS,
    ApprovalError,
    ApprovedBedrock,
    PilotProviderError,
    build_plan,
    digest,
    execute_pilot,
    proposal_manifest,
    require_approval,
    validate_run,
)

ROOT = Path(__file__).resolve().parents[1]


def fixtures():
    return load_fixture_set(ROOT / "tests/fixtures/memory_lifecycle/v1", ids=list(DEFAULT_IDS))


def approved(tmp_path, *, max_calls=1000, max_usd="5"):
    subset = fixtures()
    plan = build_plan(
        subset,
        source_sha="a" * 40,
        source_tree_sha256="b" * 64,
        host="offline-test-host",
        runtime_identity={
            "format": "kai-memory-pilot-runtime/v1",
            "python": {"version": "3.13 synthetic", "executable": "/synthetic/python"},
            "sdk": {
                "api_version": "2",
                "version": "0.3.0",
                "module_path": "/synthetic/site-packages/kai_memory",
                "tree_sha256": "d" * 64,
                "installed_distribution_match": True,
            },
            "versions": {
                key: "synthetic-version"
                for key in ("SQLAlchemy", "pgvector", "pydantic", "Unidecode", "asyncpg", "boto3", "botocore")
            },
        },
        account="123456789012",
        aws_profile="synthetic-test-profile",
        aws_config_file="/synthetic/config",
        aws_credentials_file="/synthetic/credentials",
        max_calls=max_calls,
        max_usd=max_usd,
        approval_path=str(tmp_path / "approval.json"),
        report_path=str(tmp_path / "report.json"),
    )
    manifest = proposal_manifest(plan)
    manifest["approval"] = {
        "approved": True,
        "plan_sha256": digest(plan),
        "approved_by": "offline fake transport test only",
        "approved_at": "2026-01-01T00:00:00+00:00",
    }
    return subset, plan, manifest


class FakeSTS:
    def __init__(self, account="123456789012"):
        self.account, self.calls = account, 0

    def get_caller_identity(self):
        self.calls += 1
        return {"Account": self.account}


class FakeRuntime:
    """Answers depend on API role only; no expected fixture answers are consulted."""

    def __init__(self, response=None, error=None):
        self.calls, self.response, self.error = [], response, error

    def invoke_model(self, **request):
        self.calls.append(request)
        if self.error:
            raise self.error
        body = json.loads(request["body"])
        if self.response is not None:
            response = self.response
        elif "inputText" in body:
            response = {"embedding": [1.0] + [0.0] * 1535, "inputTextTokenCount": 7}
        else:
            text = (
                '```json\n{"memories": [], "social_experiences": []}\n```'
                if body["max_tokens"] == 11000
                else "Offline transport wiring answer."
            )
            response = {
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 13, "output_tokens": 9},
            }
        return {"body": io.BytesIO(json.dumps(response).encode())}


def provider(tmp_path, *, runtime=None, max_calls=1000, max_usd="5", account="123456789012"):
    _, plan, manifest = approved(tmp_path, max_calls=max_calls, max_usd=max_usd)
    budget = SharedPilotBudget(max_calls=max_calls, max_usd=max_usd)
    return ApprovedBedrock(
        require_approval(manifest, plan), sts=FakeSTS(account), runtime=runtime or FakeRuntime(), budget=budget
    ), budget


def test_default_proposal_is_pure_and_has_no_spending_permission():
    plan = build_plan(fixtures(), source_sha="a" * 40, source_tree_sha256="b" * 64, host="offline")
    proposal = proposal_manifest(plan)
    assert proposal["approval"] is None and plan["account"] is None
    assert plan["budget"]["max_usd"] is None
    assert plan["counts"]["answer_calls"] == 360
    assert plan["counts"]["real_extraction_calls"] == 12
    assert plan["models"] == {"answer": LLM_MODEL, "extraction": LLM_MODEL, "embedding": EMBED_MODEL}
    assert plan["region"] == "us-east-1"
    assert plan["policies"]["role_output_caps"] == {"answer": 384, "extraction": 11000}
    with pytest.raises(ApprovalError):
        require_approval(proposal, plan)


def test_approval_is_immutable_and_source_hash_changes_refuse(tmp_path):
    _, plan, manifest = approved(tmp_path)
    snapshot = require_approval(manifest, plan)
    changed = json.loads(json.dumps(plan))
    changed["source_sha"] = "c" * 40
    with pytest.raises(ApprovalError, match="hash"):
        require_approval(manifest, changed)
    plan["budget"]["max_usd"] = "999"
    assert snapshot.plan["budget"]["max_usd"] == "5"


def test_plan_rejects_empty_questions_held_out_and_missing_availability():
    subset = fixtures()
    held_out = json.loads(json.dumps(subset.fixtures))
    held_out[0]["split"] = "held_out"
    with pytest.raises(ValueError, match=r"^pilot fixtures must be a nonempty tuning-only subset$"):
        build_plan(
            FixtureSet(subset.path, held_out, subset.sha256),
            source_sha="a" * 40,
            source_tree_sha256="b" * 64,
            host="offline",
        )
    altered = json.loads(json.dumps(subset.fixtures))
    altered[0]["expected"]["questions"] = []
    invalid = FixtureSet(subset.path, altered, subset.sha256)
    with pytest.raises(ValueError, match="answer probes"):
        build_plan(invalid, source_sha="a" * 40, source_tree_sha256="b" * 64, host="offline")
    altered[0]["expected"]["questions"] = [{"id": "no-metadata"}]
    altered[1]["expected"]["questions"] = [{"id": "also-no-metadata"}]
    with pytest.raises(ValueError, match="all five"):
        build_plan(invalid, source_sha="a" * 40, source_tree_sha256="b" * 64, host="offline")


def test_budget_is_thread_safe_and_failures_never_refund():
    budget = SharedPilotBudget(max_calls=7, max_usd="1")
    price = PriceCeiling(Decimal(1), Decimal(5))

    def reserve(_):
        try:
            receipt = budget.reserve(
                role="answer", model=LLM_MODEL, body="猫".encode(), max_output_tokens=384, price=price
            )
            budget.observe(receipt, usage=None, error_type="SyntheticFailure")
            return True
        except PilotBudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=16) as pool:
        assert sum(pool.map(reserve, range(40))) == 7
    report = budget.describe()
    assert report["calls"] == 7 and Decimal(report["reserved_usd"]) > 0
    assert all(row["input_token_ceiling"] == 4099 and row["status"] == "failed" for row in report["records"])
    assert all(row["usage_status"] == "unknown" for row in report["records"])


def test_dollar_cap_refuses_before_admission():
    budget = SharedPilotBudget(max_calls=1000, max_usd="0.000001")
    with pytest.raises(PilotBudgetExceeded):
        budget.reserve(
            role="embedding",
            model=EMBED_MODEL,
            body=b"text",
            max_output_tokens=0,
            price=PriceCeiling(Decimal("0.1"), Decimal("0.1")),
        )
    assert budget.describe()["calls"] == 0


def test_account_mismatch_never_invokes_bedrock(tmp_path):
    runtime = FakeRuntime()
    with pytest.raises(PilotProviderError, match="STS"):
        provider(tmp_path, runtime=runtime, account="999999999999")
    assert runtime.calls == []


@pytest.mark.parametrize(
    "response",
    [
        {"embedding": [0.0] * 1536},
        {"embedding": [1.0] * 8},
        {"embedding": [float("nan")] + [0.0] * 1535},
        {"embedding": [True] + [0.0] * 1535},
    ],
)
async def test_malformed_embeddings_charge_and_stop_without_fallback(tmp_path, response):
    runtime = FakeRuntime(response=response)
    adapter, budget = provider(tmp_path, runtime=runtime)
    with pytest.raises(PilotProviderError):
        await adapter.embedding("a real-shaped request")
    with pytest.raises(PilotProviderError, match="stopped"):
        await adapter.text("answer", "must not dispatch", max_tokens=384)
    assert len(runtime.calls) == 1
    assert budget.describe()["records"][0]["status"] == "failed"


async def test_oversized_embedding_latches_before_any_paid_call(tmp_path):
    runtime = FakeRuntime()
    adapter, budget = provider(tmp_path, runtime=runtime)
    with pytest.raises(PilotProviderError, match="8192"):
        await adapter.embedding("猫" * 3000)
    with pytest.raises(PilotProviderError, match="stopped"):
        await adapter.text("answer", "later valid answer", max_tokens=384)
    assert runtime.calls == [] and budget.describe()["calls"] == 0


async def test_transport_failure_is_charged_once_and_never_retried(tmp_path):
    runtime = FakeRuntime(error=TimeoutError("offline synthetic failure"))
    adapter, budget = provider(tmp_path, runtime=runtime)
    with pytest.raises(TimeoutError):
        await adapter.text("answer", "test", max_tokens=384)
    with pytest.raises(PilotProviderError, match="stopped"):
        await adapter.embedding("later embedding")
    assert len(runtime.calls) == 1 and budget.describe()["calls"] == 1
    assert budget.observed(adapter.approved.plan["budget"]["price_ceilings"])["usage_status"] == "partial-unknown"


async def test_missing_usage_is_unknown_not_zero_and_cached_tokens_are_visible(tmp_path):
    response = {"content": [{"type": "text", "text": "A wire answer"}], "stop_reason": "end_turn"}
    adapter, budget = provider(tmp_path, runtime=FakeRuntime(response=response))
    assert await adapter.text("answer", "test", max_tokens=384) == "A wire answer"
    assert budget.describe()["records"][0]["provider_usage"] is None
    report = budget.observed(adapter.approved.plan["budget"]["price_ceilings"])
    assert report["usage_status"] == "partial-unknown"
    assert report["by_role"]["answer"]["calls_missing_usage"] == 1


def complete_result(subset):
    return {
        "results": [
            {
                "id": f["id"],
                "status": "ok",
                "messages_saved": len(f["messages"]),
                "checkpoints": [{"after": cp["after"]} for cp in f["checkpoints"]],
                "answers": [
                    {
                        "question_id": q["id"],
                        "arm": arm,
                        "answer": "Wire answer",
                        "error": None,
                        "paired_metadata": "m4/v1",
                        **{key: digest([f["id"], q["id"], key, "synthetic receipt"]) for key in PAIR_HASHES},
                        "store_after_sha256": digest([f["id"], q["id"], "store_sha256", "synthetic receipt"]),
                    }
                    for q in f["expected"]["questions"]
                    for arm in PILOT_ARMS
                ],
                "readback": [
                    {"id": probe["id"], "retrieved": [], "formatted": ""} for probe in f["expected"].get("readback", [])
                ],
            }
            for f in subset.fixtures
        ],
        "mutation_check": {
            "runs_checked": len(subset.fixtures),
            "tables_checked": 19,
            "residual_rows": 0,
            "changed_tables": [],
        },
    }


@pytest.mark.parametrize("fault", ["empty", "answer", "rollback", "readback"])
def test_empty_or_incomplete_harness_reports_are_not_complete(fault):
    subset = fixtures()
    report = complete_result(subset)
    if fault == "empty":
        report["results"] = []
    elif fault == "answer":
        report["results"][0]["answers"].pop()
    elif fault == "readback":
        report["results"][0]["readback"] = []
    else:
        report["mutation_check"]["changed_tables"] = ["synthetic_table"]
    with pytest.raises(PilotProviderError):
        validate_run(report, subset)


async def test_whole_fake_matrix_and_single_use_claim(tmp_path, monkeypatch):
    from memory_evaluation.eval import answering

    monkeypatch.setattr(answering, "render_answer_request", lambda request: request.question)
    subset, plan, manifest = approved(tmp_path)
    approval = Path(plan["evidence"]["approval_manifest"])
    approval.write_text(json.dumps(manifest))
    runtime, sts = FakeRuntime(), FakeSTS()
    seen = []

    async def harness(selected, db, **kwargs):
        seen.append(kwargs["embedding_descriptor"]["kind"])
        for fixture in selected.fixtures:
            for cp in fixture["checkpoints"]:
                extractor = kwargs["extractor"]
                extractor.arm("offline-checkpoint", json.dumps(cp["response"]))
                await extractor(None, "Independent extraction transport prompt", max_tokens=11000)
            await kwargs["embedding_adapter"].embed(["Independent embedding transport text"])
            for question in fixture["expected"]["questions"]:
                for arm in kwargs["arms"]:
                    await kwargs["answer_fn"](SimpleNamespace(question=question["question"], arm=arm))
        return complete_result(selected)

    result = await execute_pilot(
        manifest,
        current_plan=plan,
        fixtures=subset,
        db=object(),
        approval_path=approval,
        client_factory=lambda _: (sts, runtime),
        harness=harness,
        summarize_fn=lambda reports, _: {"n_reports": len(reports), "report_only": True},
    )
    assert result["status"] == "complete" and result["progress"]["completed_runs"] == 12
    assert seen.count("real") == 6 and seen.count("lexical") == 6
    assert result["budget"]["calls"] == len(runtime.calls) == 384
    assert result["observed_usage"]["by_role"]["answer"]["calls"] == 360
    assert result["observed_usage"]["by_role"]["extraction"]["calls"] == 12
    assert result["observed_usage"]["by_role"]["embedding"]["calls"] == 12
    assert all(row["diagnostics"]["n_reports"] == 3 for row in result["repeat_diagnostics"])
    assert Path(plan["evidence"]["report"]).exists() and Path(plan["evidence"]["journal"]).exists()
    with pytest.raises(FileExistsError):
        await execute_pilot(
            manifest,
            current_plan=plan,
            fixtures=subset,
            db=object(),
            approval_path=approval,
            client_factory=lambda _: pytest.fail("replay must not create clients"),
        )
    assert len(runtime.calls) == 384


async def test_unauthorized_and_forged_fixture_content_never_construct_clients(tmp_path):
    subset, plan, manifest = approved(tmp_path)
    approval = Path(plan["evidence"]["approval_manifest"])
    approval.write_text(json.dumps(manifest))
    subset.fixtures[0]["messages"][0]["text"] = "forged content under a stale set hash"
    with pytest.raises(ApprovalError, match="fixture content"):
        await execute_pilot(
            manifest,
            current_plan=plan,
            fixtures=subset,
            db=object(),
            approval_path=approval,
            client_factory=lambda _: pytest.fail("no client before fixture verification"),
        )
    assert not Path(plan["evidence"]["claim"]).exists()


async def test_first_harness_failure_stops_later_cells_and_preserves_report(tmp_path):
    subset, plan, manifest = approved(tmp_path)
    approval = Path(plan["evidence"]["approval_manifest"])
    approval.write_text(json.dumps(manifest))
    seen = []

    async def broken(*args, **kwargs):
        seen.append(True)
        return {"results": []}

    result = await execute_pilot(
        manifest,
        current_plan=plan,
        fixtures=subset,
        db=object(),
        approval_path=approval,
        client_factory=lambda _: (FakeSTS(), FakeRuntime()),
        harness=broken,
        summarize_fn=lambda reports, _: {"n_reports": len(reports)},
    )
    assert result["status"] == "failed" and result["progress"]["completed_runs"] == 0
    assert len(seen) == 1 and len(result["reports"]) == 1
    assert result["repeat_diagnostics"][0]["diagnostics"]["n_reports"] == 1
    assert json.loads(Path(plan["evidence"]["report"]).read_text())["status"] == "failed"


async def test_budget_exhaustion_stops_swallowed_errors_and_later_cells(tmp_path, monkeypatch):
    from memory_evaluation.eval import answering

    monkeypatch.setattr(answering, "render_answer_request", lambda request: request.question)
    subset, plan, manifest = approved(tmp_path, max_calls=31)
    approval = Path(plan["evidence"]["approval_manifest"])
    approval.write_text(json.dumps(manifest))
    runtime, seen = FakeRuntime(), []

    async def swallowing_harness(selected, db, **kwargs):
        seen.append(True)
        result = complete_result(selected)
        for row in result["results"]:
            for answer in row["answers"]:
                try:
                    await kwargs["answer_fn"](SimpleNamespace(question="independent wire question"))
                except (PilotProviderError, PilotBudgetExceeded) as error:
                    answer["error"] = type(error).__name__
        return result

    result = await execute_pilot(
        manifest,
        current_plan=plan,
        fixtures=subset,
        db=object(),
        approval_path=approval,
        client_factory=lambda _: (FakeSTS(), runtime),
        harness=swallowing_harness,
        summarize_fn=lambda reports, _: {"n_reports": len(reports)},
    )
    assert result["status"] == "failed"
    assert result["progress"] == {"planned_runs": 12, "attempted_runs": 2, "completed_runs": 1}
    assert len(seen) == 2 and len(runtime.calls) == result["budget"]["calls"] == 31
    assert Decimal(result["budget"]["reserved_usd"]) <= Decimal("5")
    assert result["repeat_diagnostics"][1]["diagnostics"]["status"] == "not-run"


def test_price_ceilings_cannot_understate_known_rates(tmp_path):
    _, plan, manifest = approved(tmp_path)
    plan["budget"]["price_ceilings"][LLM_MODEL]["input_usd_per_mtok"] = "0.01"
    manifest["plan_sha256"] = digest(plan)
    manifest["approval"]["plan_sha256"] = digest(plan)
    with pytest.raises(ApprovalError, match="price ceilings"):
        require_approval(manifest, plan)


def test_client_factory_is_lazy_explicit_and_disables_ambient_sources(tmp_path, monkeypatch):
    from memory_evaluation.eval.pilot import create_aws_clients

    _, plan, manifest = approved(tmp_path)
    authorization = require_approval(manifest, plan)
    for key in list(__import__("os").environ):
        if key.startswith("AWS_") or key == "BOTO_CONFIG":
            monkeypatch.delenv(key)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    removed, configs, clients = [], {}, []

    class Core:
        def __init__(self, profile):
            configs["profile"] = profile

        def set_config_variable(self, name, value):
            configs[name] = value

        def get_component(self, name):
            assert name == "credential_provider"
            return SimpleNamespace(remove=removed.append)

    class Session:
        def __init__(self, **kwargs):
            assert isinstance(kwargs["botocore_session"], Core)
            assert kwargs["region_name"] == "us-east-1"

        def client(self, service, **kwargs):
            clients.append((service, kwargs))
            return object()

    boto = ModuleType("boto3")
    boto.Session = Session
    core = ModuleType("botocore")
    core_session = ModuleType("botocore.session")
    core_session.Session = Core
    core.session = core_session
    config = ModuleType("botocore.config")
    config.Config = lambda **kwargs: SimpleNamespace(**kwargs)
    for name, module in {
        "boto3": boto,
        "botocore": core,
        "botocore.session": core_session,
        "botocore.config": config,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    create_aws_clients(authorization)
    assert configs == {
        "profile": "synthetic-test-profile",
        "config_file": "/synthetic/config",
        "credentials_file": "/synthetic/credentials",
    }
    assert {"env", "iam-role", "container-role", "boto-config", "ec2-credentials-file", "login"} <= set(removed)
    assert [service for service, _ in clients] == ["sts", "bedrock-runtime"]
    for service, kwargs in clients:
        assert kwargs["endpoint_url"] == f"https://{service}.us-east-1.amazonaws.com"
        assert kwargs["config"].retries["total_max_attempts"] == 1
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "synthetic-sentinel-no-secret")
    with pytest.raises(ApprovalError, match="clean environment"):
        create_aws_clients(authorization)
    assert len(clients) == 2


@pytest.mark.parametrize("field", [*PAIR_HASHES, "store_after_sha256", "paired_metadata"])
def test_run_refuses_invalid_paired_receipts(field):
    subset = fixtures()
    report = complete_result(subset)
    answer = report["results"][0]["answers"][1]
    answer[field] = "legacy" if field == "paired_metadata" else "f" * 64
    with pytest.raises(PilotProviderError, match="paired arm receipts"):
        validate_run(report, subset)


def test_run_refuses_legacy_hash_only_rows_even_when_two_hashes_agree():
    subset = fixtures()
    report = complete_result(subset)
    for result in report["results"]:
        for answer in result["answers"]:
            answer.pop("paired_metadata")
    with pytest.raises(PilotProviderError, match="paired arm receipts"):
        validate_run(report, subset)


@pytest.mark.parametrize("actual", [[16, 8], [8, 8], [8, 17]])
def test_run_requires_exact_checkpoint_sequence(actual):
    subset = fixtures()
    fixture = subset.fixtures[0]
    checkpoint = fixture["checkpoints"][0]
    fixture["checkpoints"] = [{**checkpoint, "after": 8}, {**checkpoint, "after": 16}]
    report = complete_result(subset)
    report["results"][0]["checkpoints"] = [{"after": after} for after in actual]
    with pytest.raises(PilotProviderError, match="extraction checkpoints"):
        validate_run(report, subset)


def test_model_quality_miss_remains_a_valid_complete_pipeline_run():
    subset = fixtures()
    report = complete_result(subset)
    assert check_answer(subset.fixtures[0]["expected"]["questions"][0], "Wire answer")["verdict"] == "incorrect"
    validate_run(report, subset)


def test_runtime_module_hash_change_invalidates_approval(tmp_path):
    _, plan, manifest = approved(tmp_path)
    altered = json.loads(json.dumps(plan))
    altered["runtime_identity"]["sdk"]["tree_sha256"] = "e" * 64
    with pytest.raises(ApprovalError, match="hash"):
        require_approval(manifest, altered)


def test_source_override_or_missing_runtime_identity_cannot_execute(tmp_path):
    _, plan, manifest = approved(tmp_path)
    plan["runtime_identity"]["sdk"]["installed_distribution_match"] = False
    manifest["plan_sha256"] = digest(plan)
    manifest["approval"]["plan_sha256"] = digest(plan)
    with pytest.raises(ApprovalError, match="actual installed SDK"):
        require_approval(manifest, plan)


def assert_interrupted_evidence(plan, error_type):
    saved = json.loads(Path(plan["evidence"]["report"]).read_text())
    assert saved["status"] == "interrupted" and saved["error_type"] == error_type
    assert saved["progress"] == {"planned_runs": 12, "attempted_runs": 1, "completed_runs": 0}
    assert saved["budget"]["calls"] == 1 and Decimal(saved["budget"]["reserved_usd"]) > 0
    assert saved["observed_usage"]["by_role"]["answer"]["calls"] == 1
    assert json.loads(Path(plan["evidence"]["claim"]).read_text())["plan_sha256"] == digest(plan)
    journal = [json.loads(line) for line in Path(plan["evidence"]["journal"]).read_text().splitlines()]
    assert [event["event"] for event in journal] == ["approved-start", "reserve", "observe"]


async def test_keyboard_interrupt_persists_interruption_and_reraises_without_continuation(tmp_path):
    from memory_evaluation.eval.answering import AnswerRequest

    subset, plan, manifest = approved(tmp_path)
    approval = Path(plan["evidence"]["approval_manifest"])
    approval.write_text(json.dumps(manifest))
    runtime, seen = FakeRuntime(), []
    interruption = KeyboardInterrupt("synthetic operator interruption")

    async def interrupted_harness(selected, db, **kwargs):
        seen.append(True)
        await kwargs["answer_fn"](AnswerRequest("f", "q", "no_memory", "Independent wire prompt.", None, (), "", ()))
        raise interruption

    with pytest.raises(KeyboardInterrupt) as error:
        await execute_pilot(
            manifest,
            current_plan=plan,
            fixtures=subset,
            db=object(),
            approval_path=approval,
            client_factory=lambda _: (FakeSTS(), runtime),
            harness=interrupted_harness,
        )
    assert error.value is interruption
    assert len(seen) == len(runtime.calls) == 1
    assert_interrupted_evidence(plan, "KeyboardInterrupt")
    with pytest.raises(FileExistsError):
        await execute_pilot(
            manifest,
            current_plan=plan,
            fixtures=subset,
            db=object(),
            approval_path=approval,
            client_factory=lambda _: pytest.fail("interrupted single-use plan must not dispatch again"),
        )
    assert len(runtime.calls) == 1


async def test_task_cancellation_persists_interruption_and_preserves_single_use_budget(tmp_path):
    import asyncio
    from memory_evaluation.eval.answering import AnswerRequest

    subset, plan, manifest = approved(tmp_path)
    approval = Path(plan["evidence"]["approval_manifest"])
    approval.write_text(json.dumps(manifest))
    runtime, seen = FakeRuntime(), []
    ready = asyncio.Event()

    async def waiting_harness(selected, db, **kwargs):
        seen.append(True)
        await kwargs["answer_fn"](AnswerRequest("f", "q", "no_memory", "Independent wire prompt.", None, (), "", ()))
        ready.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(
        execute_pilot(
            manifest,
            current_plan=plan,
            fixtures=subset,
            db=object(),
            approval_path=approval,
            client_factory=lambda _: (FakeSTS(), runtime),
            harness=waiting_harness,
        )
    )
    await asyncio.wait_for(ready.wait(), timeout=5)
    task.cancel("synthetic cancellation")
    with pytest.raises(asyncio.CancelledError, match="synthetic cancellation"):
        await task
    assert len(seen) == len(runtime.calls) == 1
    assert_interrupted_evidence(plan, "CancelledError")
    with pytest.raises(FileExistsError):
        await execute_pilot(
            manifest,
            current_plan=plan,
            fixtures=subset,
            db=object(),
            approval_path=approval,
            client_factory=lambda _: pytest.fail("cancelled single-use plan must not dispatch again"),
        )
    assert len(runtime.calls) == 1


@pytest.mark.parametrize("outcome", ["complete", "failed", "interrupted"])
async def test_default_repeat_diagnostics_bind_only_a_scoped_fixture_clock(tmp_path, outcome):
    from kai_memory import ports

    subset, plan, manifest = approved(tmp_path)
    approval = Path(plan["evidence"]["approval_manifest"])
    approval.write_text(json.dumps(manifest))
    calls = 0

    async def harness(selected, db, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2 and outcome != "complete":
            if outcome == "interrupted":
                raise KeyboardInterrupt("synthetic interruption")
            raise RuntimeError("synthetic failure")
        result = complete_result(selected)
        result.update(
            arms=list(PILOT_ARMS),
            fixture_set={"sha256": selected.sha256},
            answerer={"name": "synthetic"},
            extractor={"name": "synthetic"},
            embedder={"name": "synthetic"},
        )
        return result

    previous_resolver = ports.set_default_resolver(None)
    previous_bindings = ports.bind(
        **{
            name: ports.UNBOUND
            for name in ports.__all__
            if name not in {"UNBOUND", "bind", "scoped", "set_default_resolver"}
        }
    )
    try:
        with ports.scoped({}, inherit=False):
            with pytest.raises(RuntimeError):
                ports.clock()
            kwargs = dict(
                current_plan=plan,
                fixtures=subset,
                db=object(),
                approval_path=approval,
                client_factory=lambda _: (FakeSTS(), FakeRuntime()),
                harness=harness,
            )
            if outcome == "interrupted":
                with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
                    await execute_pilot(manifest, **kwargs)
                result = json.loads(Path(plan["evidence"]["report"]).read_text())
            else:
                result = await execute_pilot(manifest, **kwargs)
            with pytest.raises(RuntimeError):
                ports.clock()
    finally:
        ports.bind(**previous_bindings)
        ports.set_default_resolver(previous_resolver)
    assert result["status"] == outcome
    populated = [row["diagnostics"] for row in result["repeat_diagnostics"] if row["diagnostics"].get("repeat_count")]
    assert len(populated) == (4 if outcome == "complete" else 1)
    for summary in populated:
        assert summary["repeat_count"] == (3 if outcome == "complete" else 1)
        assert set(summary["per_arm"]) == set(PILOT_ARMS)
        assert all(arm["n_valid"] > 0 for arm in summary["per_arm"].values())


@pytest.mark.parametrize("outcome", ["complete", "failed", "interrupted"])
async def test_default_repeat_diagnostics_restore_the_callers_clock(tmp_path, monkeypatch, outcome):
    from datetime import UTC, datetime

    from kai_memory import ports
    from memory_evaluation.eval import diagnostics
    from memory_evaluation.eval.pilot import fixture_now

    subset, plan, manifest = approved(tmp_path)
    approval = Path(plan["evidence"]["approval_manifest"])
    approval.write_text(json.dumps(manifest))
    calls = 0
    caller_now = datetime(2042, 6, 7, tzinfo=UTC)

    real_summarize = diagnostics.summarize_repeats
    summary_clocks = []

    def summarize_with_clock_evidence(*args, **kwargs):
        summary_clocks.append(ports.clock())
        return real_summarize(*args, **kwargs)

    monkeypatch.setattr(diagnostics, "summarize_repeats", summarize_with_clock_evidence)

    def caller_clock():
        return caller_now

    async def harness(selected, db, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2 and outcome != "complete":
            if outcome == "interrupted":
                raise KeyboardInterrupt("synthetic interruption")
            raise RuntimeError("synthetic failure")
        result = complete_result(selected)
        result.update(
            arms=list(PILOT_ARMS),
            fixture_set={"sha256": selected.sha256},
            answerer={"name": "synthetic"},
            extractor={"name": "synthetic"},
            embedder={"name": "synthetic"},
        )
        return result

    with ports.scoped({"clock": caller_clock}, inherit=False):
        assert ports.clock is caller_clock
        assert ports.clock() is caller_now
        kwargs = dict(
            current_plan=plan,
            fixtures=subset,
            db=object(),
            approval_path=approval,
            client_factory=lambda _: (FakeSTS(), FakeRuntime()),
            harness=harness,
        )
        if outcome == "interrupted":
            with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
                await execute_pilot(manifest, **kwargs)
            result = json.loads(Path(plan["evidence"]["report"]).read_text())
            assert result["error_type"] == "KeyboardInterrupt"
            assert len(result["reports"]) == 1
        else:
            result = await execute_pilot(manifest, **kwargs)
        assert ports.clock is caller_clock
        assert ports.clock() is caller_now
    assert result["status"] == outcome
    populated = [row["diagnostics"] for row in result["repeat_diagnostics"] if row["diagnostics"].get("repeat_count")]
    assert len(populated) == (4 if outcome == "complete" else 1)
    assert all(summary["per_arm"]["windowed"]["n_valid"] > 0 for summary in populated)
    assert summary_clocks
    assert all(value == fixture_now(subset.fixtures[0]) for value in summary_clocks)
