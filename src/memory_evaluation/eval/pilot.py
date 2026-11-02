"""Approval-gated generic-host Bedrock pilot; importing/planning never creates AWS clients.

Optional boto dependencies are imported only by the explicitly approved client
factory. This is a report-only quality pilot, not a deployment or a calibrated
semantic evaluation. Haiku uses the exact global inference profile from US East;
there is no host/model fallback, SDK retry, prompt cache or embedding cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from itertools import combinations
from typing import Any, cast

from memory_evaluation.eval.budget import PriceCeiling, SharedPilotBudget, positive_decimal
from memory_evaluation.eval.extractors import ExtractionCall, RecordedExtractor, UnexpectedExtractionCall
from memory_evaluation.eval.fixtures import AVAILABILITIES, FixtureSet, content_sha256

PLAN_FORMAT = "kai-memory-pilot-plan/v1"
APPROVAL_FORMAT = "kai-memory-pilot-approval/v1"
REPORT_FORMAT = "kai-memory-pilot-report/v1"
LLM_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
EMBED_MODEL = "amazon.titan-embed-text-v1"
PILOT_ARMS = ("no_memory", "last_message", "windowed")
CELLS = (("recorded", "lexical"), ("recorded", "real"), ("real", "lexical"), ("real", "real"))
DEFAULT_IDS = ("lc-05-volunteer-garden", "lc-06-choir-rehearsal")
MAX_OUTPUT_TOKENS = 11000
MAX_RESPONSE_BYTES = 2_000_000


class ApprovalError(ValueError):
    """The immutable proposed plan lacks matching explicit approval."""


class PilotProviderError(RuntimeError):
    """An account, endpoint or model response violated the approved contract."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def fixture_now(fixture: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(fixture["start_at"]) + timedelta(minutes=fixture["messages"][-1]["minute"] + 1)


def validate_prices(prices: dict[str, dict[str, str]]) -> None:
    """Frozen 2026-10-28 official list rates are the minimum admitted ceilings."""
    if set(prices) != {LLM_MODEL, EMBED_MODEL}:
        raise ValueError("exact pinned model prices required")
    llm = PriceCeiling.from_dict(prices[LLM_MODEL])
    embed = PriceCeiling.from_dict(prices[EMBED_MODEL])
    if llm.input_usd_per_mtok < 1 or llm.output_usd_per_mtok < 5 or embed.input_usd_per_mtok < positive_decimal("0.10"):
        raise ValueError("price ceilings cannot be lower than frozen official list rates")


def build_plan(
    fixtures: FixtureSet,
    *,
    source_sha: str,
    source_tree_sha256: str,
    host: str,
    account: str | None = None,
    region: str = "us-east-1",
    repeats: int = 3,
    max_calls: int = 1000,
    max_usd: str | None = None,
    aws_profile: str | None = None,
    aws_config_file: str | None = None,
    aws_credentials_file: str | None = None,
    database: dict[str, Any] | None = None,
    prices: dict[str, dict[str, str]] | None = None,
    approval_path: str | None = None,
    report_path: str | None = None,
    runtime_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pure proposal builder: no database, client, credential lookup or paid calls.

    Exact answer/extraction counts are known; embedding calls depend on extracted
    state. A hard total call cap covers every embedding text, across all cells.
    A low cap/budget can intentionally abort before completing the matrix.
    """
    if repeats != 3:
        raise ValueError("this pilot requires three report-only repeats")
    if not fixtures.fixtures or any(f.get("split") != "tuning" for f in fixtures.fixtures):
        raise ValueError("pilot fixtures must be a nonempty tuning-only subset")
    if len({f["id"] for f in fixtures.fixtures}) != len(fixtures.fixtures):
        raise ValueError("pilot fixture IDs must be unique")
    if any(not f.get("expected", {}).get("questions") for f in fixtures.fixtures):
        raise ValueError("every pilot fixture needs answer probes")
    if {q.get("availability") for f in fixtures.fixtures for q in f["expected"]["questions"]} != AVAILABILITIES:
        raise ValueError("pilot selection must cover all five question availability classes")
    if len(fixtures.fixtures) < 2:
        raise ValueError("pilot needs at least two distinct tuning fixtures")
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha) or not re.fullmatch(r"[0-9a-f]{64}", source_tree_sha256):
        raise ValueError("exact source commit and source-tree hash required")
    if (
        (account is not None and not re.fullmatch(r"[0-9]{12}", account))
        or not re.fullmatch(r"[a-z]{2}-[a-z]+-\d", region)
        or not host
    ):
        raise ValueError("explicit account, region and host required")
    if type(max_calls) is not int or max_calls < 1:
        raise ValueError("max_calls must be positive")
    if max_usd is not None:
        positive_decimal(max_usd)
    questions = sum(len(f["expected"].get("questions", [])) for f in fixtures.fixtures)
    checkpoints = sum(len(f["checkpoints"]) for f in fixtures.fixtures)
    answer_calls = questions * len(PILOT_ARMS) * len(CELLS) * repeats
    extraction_calls = checkpoints * 2 * repeats
    prices = prices or {
        LLM_MODEL: {"input_usd_per_mtok": "1", "output_usd_per_mtok": "5"},
        EMBED_MODEL: {"input_usd_per_mtok": "0.10", "output_usd_per_mtok": "0"},
    }
    validate_prices(prices)
    return {
        "format": PLAN_FORMAT,
        "issue": "DB-881",
        "source_sha": source_sha,
        "source_tree_sha256": source_tree_sha256,
        "runtime_identity": runtime_identity,
        "host": host,
        "fixture_set_sha256": fixtures.sha256,
        "fixtures": [
            {"id": f["id"], "sha256": f["_provenance"]["content_sha256"], "clock": fixture_now(f).isoformat()}
            for f in fixtures.fixtures
        ],
        "account": account,
        "region": region,
        "identity": {"profile": aws_profile, "config_file": aws_config_file, "credentials_file": aws_credentials_file},
        "models": {"answer": LLM_MODEL, "extraction": LLM_MODEL, "embedding": EMBED_MODEL},
        "routing": "exact AWS global inference profile via regional Bedrock endpoint; no fallback",
        "repeats": repeats,
        "arms": list(PILOT_ARMS),
        "cells": [{"extraction": extraction, "embedding": embedding} for extraction, embedding in CELLS],
        "host_adapter": "standalone.recorded_host + explicit pilot adapters; no Kai application adapter",
        "policies": {
            "history_window": 30,
            "clock": "fixture start + final message minute + 1",
            "stage_c": "disabled: no reasoning-summary injection in pilot",
            "parser": "SDK 0.3.0 default _parse_object/_parse_array",
            "judge": "lexical diagnostics only; no semantic calibration",
            "retries": 0,
            "cache": "disabled; every embedding text is sent and charged",
            "temperature": 0,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "role_output_caps": {"answer": 384, "extraction": 11000},
            "endpoint_envelope_tokens": 4096,
            "max_request_bytes": 100_000,
            "max_embedding_text_bytes": 8192,
            "embedding_dimensions": 1536,
            "abort_on_provider_or_harness_error": True,
        },
        "evidence": {
            "approval_manifest": approval_path,
            "report": report_path,
            "journal": report_path + ".calls.jsonl" if report_path else None,
            "claim": approval_path + ".claim" if approval_path else None,
        },
        "database": database or {"host": "localhost", "port": 5432, "database": "memory_pilot"},
        "budget": {
            "max_calls": max_calls,
            "max_usd": max_usd,
            "price_ceilings": prices,
            "price_status": "frozen list-price ceilings; no cache discount",
            "sources": [
                "https://www-cdn.anthropic.com/files/4zrzovbb/website/3684c2faafb97418665782cea0001f439f74b1d2.pdf",
                "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/us-east-1/index.json",
            ],
        },
        "counts": {
            "fixtures": len(fixtures.fixtures),
            "questions": questions,
            "checkpoints": checkpoints,
            "answer_calls": answer_calls,
            "real_extraction_calls": extraction_calls,
            "known_model_calls": answer_calls + extraction_calls,
            "embedding_calls": "data-dependent, bounded by shared paid-call cap",
            "total_paid_call_cap": max_calls,
        },
        "limits": [
            "native account model access unverified until approved execution",
            "global inference can route across AWS regions within the approved account; synthetic data only",
            "reservation ceiling uses UTF-8 byte bound plus endpoint envelope and approved price ceilings",
            "three repeats are diagnostics, not acceptance evidence",
            "generic host only; no Kai adapter parity or human-reviewed semantic calibration",
        ],
    }


def proposal_manifest(plan: dict[str, Any]) -> dict[str, Any]:
    return {"format": APPROVAL_FORMAT, "plan": plan, "plan_sha256": digest(plan), "approval": None}


@dataclass(frozen=True)
class ApprovedPilot:
    """Canonical immutable bytes prevent caller mutation after approval checks."""

    canonical_plan: bytes
    sha256: str
    approved_by: str
    approved_at: str

    @property
    def plan(self) -> dict[str, Any]:
        return cast("dict[str, Any]", json.loads(self.canonical_plan))


def require_approval(manifest: dict[str, Any], current_plan: dict[str, Any]) -> ApprovedPilot:
    """Verify full plan equality before constructing any AWS client or reading identity files."""
    if manifest.get("format") != APPROVAL_FORMAT or not isinstance(manifest.get("plan"), dict):
        raise ApprovalError("a complete approval manifest is required")
    plan = manifest["plan"]
    sha = digest(plan)
    if manifest.get("plan_sha256") != sha or digest(current_plan) != sha:
        raise ApprovalError("approved plan hash does not match current source, fixtures, host or configuration")
    approval = manifest.get("approval")
    if not isinstance(approval, dict) or approval.get("approved") is not True or approval.get("plan_sha256") != sha:
        raise ApprovalError("explicit human approval for this exact plan hash is required")
    if not isinstance(approval.get("approved_by"), str) or not approval["approved_by"].strip():
        raise ApprovalError("approval needs an explicit approver")
    try:
        at = datetime.fromisoformat(approval.get("approved_at", ""))
    except (ValueError, TypeError):
        raise ApprovalError("approval needs an offset-aware timestamp") from None
    if at.tzinfo is None:
        raise ApprovalError("approval needs an offset-aware timestamp")
    if plan.get("models") != {"answer": LLM_MODEL, "extraction": LLM_MODEL, "embedding": EMBED_MODEL}:
        raise ApprovalError("model IDs must match the pinned pilot models")
    if plan.get("repeats") != 3 or plan.get("arms") != list(PILOT_ARMS):
        raise ApprovalError("pilot repeats and paired arms changed")
    if plan.get("cells") != [{"extraction": x, "embedding": y} for x, y in CELLS]:
        raise ApprovalError("all four extraction/embedding cells are required")
    if not isinstance(plan.get("account"), str) or not re.fullmatch(r"[0-9]{12}", plan["account"]):
        raise ApprovalError("approval must name an exact AWS account")
    identity = plan.get("identity") or {}
    if not identity.get("profile") or any(
        not isinstance(identity.get(k), str) or not Path(identity[k]).is_absolute()
        for k in ("config_file", "credentials_file")
    ):
        raise ApprovalError("approval must name the exact AWS profile and absolute config/credentials file paths")
    runtime = plan.get("runtime_identity") or {}
    sdk = runtime.get("sdk") or {}
    python = runtime.get("python") or {}
    versions = runtime.get("versions") or {}
    required_versions = {"SQLAlchemy", "pgvector", "pydantic", "Unidecode", "asyncpg", "boto3", "botocore"}
    if (
        runtime.get("format") != "kai-memory-pilot-runtime/v1"
        or sdk.get("api_version") != "2"
        or not sdk.get("installed_distribution_match")
        or not isinstance(sdk.get("tree_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", sdk["tree_sha256"]) is None
        or any(not isinstance(value, str) or not value for value in (sdk.get("version"), python.get("version")))
        or any(
            not isinstance(value, str) or not Path(value).is_absolute()
            for value in (sdk.get("module_path"), python.get("executable"))
        )
        or any(not isinstance(versions.get(key), str) or not versions[key] for key in required_versions)
    ):
        raise ApprovalError(
            "approval requires actual installed SDK/API, Python and dependency/provider runtime identity"
        )
    evidence = plan.get("evidence") or {}
    if any(
        not isinstance(evidence.get(k), str) or not Path(evidence[k]).is_absolute()
        for k in ("approval_manifest", "report", "journal", "claim")
    ):
        raise ApprovalError("approval must bind absolute manifest, single-use claim, report and journal paths")
    if (
        evidence["journal"] != evidence["report"] + ".calls.jsonl"
        or evidence["claim"] != evidence["approval_manifest"] + ".claim"
        or len(set(evidence.values())) != 4
    ):
        raise ApprovalError("execution evidence paths are inconsistent")
    budget = plan.get("budget") or {}
    try:
        SharedPilotBudget(max_calls=budget["max_calls"], max_usd=budget["max_usd"])
        validate_prices(budget["price_ceilings"])
    except (KeyError, ValueError, TypeError):
        raise ApprovalError(
            "positive explicit total budget, call cap and exact model price ceilings required"
        ) from None
    return ApprovedPilot(canonical_bytes(plan), sha, approval["approved_by"], approval["approved_at"])


def create_aws_clients(approved: ApprovedPilot) -> tuple[Any, Any]:
    """Only approved execution imports boto and resolves the explicitly named profile.

    No environment, instance-role or container-role fallback identity is accepted.
    The profile may use its own explicitly configured SSO/assume-role provider.
    """
    forbidden = {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_CA_BUNDLE",
        "BOTO_CONFIG",
        "AWS_CREDENTIAL_FILE",
    }
    if forbidden.intersection(os.environ) or os.environ.get("AWS_EC2_METADATA_DISABLED") != "true":
        raise ApprovalError("execute in an explicit clean environment with AWS_EC2_METADATA_DISABLED=true")
    import boto3
    import botocore.session
    from botocore.config import Config

    plan = approved.plan
    identity = plan["identity"]
    core = botocore.session.Session(profile=identity["profile"])
    core.set_config_variable("config_file", identity["config_file"])
    core.set_config_variable("credentials_file", identity["credentials_file"])
    resolver = core.get_component("credential_provider")
    for provider in (
        "env",
        "iam-role",
        "container-role",
        "assume-role-with-web-identity",
        "boto-config",
        "ec2-credentials-file",
        "login",
    ):
        resolver.remove(provider)
    session = boto3.Session(botocore_session=core, region_name=plan["region"])
    config = Config(retries={"total_max_attempts": 1, "mode": "standard"}, connect_timeout=10, read_timeout=120)
    region = plan["region"]
    sts = session.client("sts", region_name=region, endpoint_url=f"https://sts.{region}.amazonaws.com", config=config)
    runtime = session.client(
        "bedrock-runtime",
        region_name=region,
        endpoint_url=f"https://bedrock-runtime.{region}.amazonaws.com",
        config=config,
    )
    return sts, runtime


def _usage(body: dict[str, Any], *, embedding: bool) -> dict[str, int] | None:
    raw = (
        {"input_tokens": body["inputTextTokenCount"]}
        if embedding and "inputTextTokenCount" in body
        else body.get("usage")
    )
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise PilotProviderError("malformed provider usage")
    fields = ("input_tokens",) if embedding else ("input_tokens", "output_tokens")
    if any(type(raw.get(k)) is not int or raw[k] < 0 for k in fields):
        raise PilotProviderError("malformed provider usage")
    counters = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    result = {k: raw[k] for k in counters if k in raw}
    if any(type(v) is not int or v < 0 for v in result.values()):
        raise PilotProviderError("malformed provider cache usage")
    # Absent counters remain absent/unknown, never invented zeros.
    return result


class ApprovedBedrock:
    """Injectable synchronous boto transport with one shared all-role budget."""

    def __init__(self, approved: ApprovedPilot, *, sts: Any, runtime: Any, budget: SharedPilotBudget) -> None:
        self.approved = approved
        self.runtime = runtime
        self.budget = budget
        self._verified = False
        self._failure: str | None = None
        identity = sts.get_caller_identity()
        if not isinstance(identity, dict) or identity.get("Account") != approved.plan["account"]:
            raise PilotProviderError("STS account does not match the approved account; no inference sent")
        self._verified = True

    def stop(self, reason: str) -> None:
        self._failure = reason
        raise PilotProviderError(reason)

    async def _invoke(self, role: str, model: str, payload: dict[str, Any], max_tokens: int, validator: Any) -> Any:
        if self._failure is not None:
            raise PilotProviderError("pilot provider is stopped after an earlier failed call")
        if not self._verified:
            self.stop("account verification required")
        plan = self.approved.plan
        if model != plan["models"]["embedding" if role == "embedding" else role]:
            self.stop("unapproved model")
        body = canonical_bytes(payload)
        if len(body) > plan["policies"]["max_request_bytes"]:
            self.stop("request exceeds the approved byte ceiling")
        try:
            reservation = self.budget.reserve(
                role=role,
                model=model,
                body=body,
                max_output_tokens=max_tokens,
                price=PriceCeiling.from_dict(plan["budget"]["price_ceilings"][model]),
            )
        except Exception as error:
            self._failure = type(error).__name__
            raise
        usage = None
        try:
            response = await asyncio.to_thread(
                self.runtime.invoke_model,
                modelId=model,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            stream = response["body"]
            try:
                data = await asyncio.to_thread(stream.read, MAX_RESPONSE_BYTES + 1)
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            if len(data) > MAX_RESPONSE_BYTES:
                raise PilotProviderError("provider response exceeds the bounded envelope")
            parsed = json.loads(data)
            if not isinstance(parsed, dict):
                raise PilotProviderError("provider response must be an object")
            usage = _usage(parsed, embedding=role == "embedding")
            result = validator(parsed)
        except BaseException as error:
            self._failure = type(error).__name__
            self.budget.observe(reservation, usage=usage, error_type=type(error).__name__)
            raise
        self.budget.observe(reservation, usage=usage)
        return result

    @staticmethod
    def parse_text(response: dict[str, Any]) -> str:
        content = response.get("content")
        if response.get("stop_reason") not in {"end_turn", "stop_sequence"} or not isinstance(content, list):
            raise PilotProviderError("malformed or truncated text response")
        if not content or any(
            not isinstance(c, dict) or c.get("type") != "text" or not isinstance(c.get("text"), str) for c in content
        ):
            raise PilotProviderError("text response contains unsupported blocks")
        answer = "".join(c["text"] for c in content)
        if not answer.strip():
            raise PilotProviderError("empty text response")
        return answer

    @staticmethod
    def parse_embedding(response: dict[str, Any]) -> list[float]:
        vector = response.get("embedding")
        if (
            not isinstance(vector, list)
            or len(vector) != 1536
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)
            or not any(v != 0 for v in vector)
        ):
            raise PilotProviderError("embedding response must contain 1536 finite nonzero vector components")
        return [float(v) for v in vector]

    async def text(self, role: str, prompt: str, *, max_tokens: int, validate_response: Any = None) -> str:
        if role not in {"answer", "extraction"} or not isinstance(prompt, str) or not prompt.strip():
            self.stop("a text call requires its approved role and a nonempty prompt")
        if (
            type(max_tokens) is not int
            or not 1 <= max_tokens <= self.approved.plan["policies"]["role_output_caps"][role]
        ):
            self.stop("output token limit exceeds approval")

        def validate(response):
            text = self.parse_text(response)
            if validate_response is not None:
                validate_response(text)
            return text

        return cast(
            "str",
            await self._invoke(
                role,
                LLM_MODEL,
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                },
                max_tokens,
                validate,
            ),
        )

    async def embedding(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 8192:
            self.stop("embedding text must be nonempty and at most 8192 UTF-8 bytes")
        return cast(
            "list[float]", await self._invoke("embedding", EMBED_MODEL, {"inputText": text}, 0, self.parse_embedding)
        )


class PilotEmbedder:
    name = EMBED_MODEL
    timeout_s = 120
    max_retries = 0

    def __init__(self, provider: ApprovedBedrock) -> None:
        self.provider = provider

    async def embed(self, texts: list[str], *, api_key: str | None = None) -> list[list[float]]:
        if api_key is not None:
            self.provider.stop("pilot embedding uses only the approved AWS identity")
        return [await self.provider.embedding(text) for text in texts]


class PilotExtractor:
    def __init__(self, provider: ApprovedBedrock) -> None:
        self.provider = provider
        self.calls: list[ExtractionCall] = []
        self._checkpoint: str | None = None
        self._consumed = False

    def describe(self) -> dict[str, Any]:
        return {"name": "approved-bedrock", "model": LLM_MODEL, "quality": "real extractor; uncalibrated"}

    def arm(self, checkpoint: str, response: str | None) -> None:
        self._checkpoint, self._consumed = checkpoint, False

    def consumed(self) -> bool:
        return self._consumed

    async def __call__(self, provider: Any, prompt: str, max_tokens: int = 2048, span: Any = None, **_: Any) -> str:
        if self._checkpoint is None or self._consumed:
            raise UnexpectedExtractionCall("pilot extraction needs one explicit checkpoint")
        self._consumed = True
        from kai_memory.host_calls import parse_json_object_response

        def validate(raw):
            parsed = parse_json_object_response(raw)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("memories"), list):
                raise PilotProviderError("real extraction must return a JSON object with memories")

        response = await self.provider.text("extraction", prompt, max_tokens=max_tokens, validate_response=validate)
        self.calls.append(
            ExtractionCall(self._checkpoint, prompt, hashlib.sha256(prompt.encode()).hexdigest(), response)
        )
        return response


class PilotAnswerer:
    def __init__(self, provider: ApprovedBedrock) -> None:
        self.provider = provider

    def describe(self) -> dict[str, Any]:
        from memory_evaluation.eval.answering import ANSWER_PROMPT

        return {
            "name": "approved-bedrock",
            "model": LLM_MODEL,
            "temperature": 0,
            "max_tokens": 384,
            "prompt_sha256": hashlib.sha256(ANSWER_PROMPT.encode()).hexdigest(),
            "quality": "report-only lexical diagnostics; no human calibration",
        }

    async def __call__(self, request: Any) -> str:
        from memory_evaluation.eval.answering import render_answer_request

        return await self.provider.text("answer", render_answer_request(request), max_tokens=384)


def validate_run(result: dict[str, Any], fixtures: FixtureSet) -> None:
    """Completeness is pipeline validity, never a threshold on measured model quality."""
    from memory_evaluation.eval.answering import comparable_pair

    mutation = result.get("mutation_check") or {}
    if (
        mutation.get("residual_rows") != 0
        or mutation.get("changed_tables") != []
        or mutation.get("runs_checked", 0) < len(fixtures.fixtures)
        or mutation.get("tables_checked", 0) <= 0
    ):
        raise PilotProviderError("missing or invalid rollback/mutation evidence")
    results = result.get("results")
    if not isinstance(results, list) or len(results) != len(fixtures.fixtures):
        raise PilotProviderError("incomplete fixture results")
    by_id = {r.get("id"): r for r in results}
    if set(by_id) != {f["id"] for f in fixtures.fixtures} or len(by_id) != len(results):
        raise PilotProviderError("duplicate, missing or unknown fixture results")
    for fixture in fixtures.fixtures:
        row = by_id[fixture["id"]]
        if row.get("status") != "ok" or row.get("messages_saved") != len(fixture["messages"]):
            raise PilotProviderError("failed or incomplete conversation replay")
        checkpoints = row.get("checkpoints") or []
        if [cp.get("after") for cp in checkpoints] != [cp["after"] for cp in fixture["checkpoints"]] or any(
            cp.get("error") or cp.get("store_errors") for cp in checkpoints
        ):
            raise PilotProviderError("failed or incomplete extraction checkpoints")
        answers = row.get("answers") or []
        pairs = {(a.get("question_id"), a.get("arm")) for a in answers}
        expected = {(q["id"], arm) for q in fixture["expected"]["questions"] for arm in PILOT_ARMS}
        if (
            len(answers) != len(expected)
            or pairs != expected
            or any(a.get("error") or not isinstance(a.get("answer"), str) or not a["answer"].strip() for a in answers)
        ):
            raise PilotProviderError("failed, missing or duplicate answers")
        by_question: dict[str, dict[str, dict[str, Any]]] = {}
        for answer in answers:
            by_question.setdefault(answer["question_id"], {})[answer["arm"]] = answer
        for group in by_question.values():
            if any(answer.get("paired_metadata") != "m4/v1" for answer in group.values()) or any(
                not comparable_pair(group[left], group[right]) for left, right in combinations(PILOT_ARMS, 2)
            ):
                raise PilotProviderError("incomparable or missing M4 paired arm receipts")
        readback = row.get("readback") or []
        expected_readback = {probe["id"] for probe in fixture["expected"].get("readback", [])}
        if (
            {probe.get("id") for probe in readback} != expected_readback
            or len(readback) != len(expected_readback)
            or any(
                probe.get("error")
                or not isinstance(probe.get("retrieved"), list)
                or not isinstance(probe.get("formatted"), str)
                for probe in readback
            )
        ):
            raise PilotProviderError("failed or incomplete readback probes")


@contextmanager
def execution_files(evidence: dict[str, str], plan_sha256: str):
    """Exclusive local durability gates complete before any external call."""
    with Path(evidence["claim"]).open("x", encoding="utf-8") as claim:
        claim.write(
            canonical_bytes({"plan_sha256": plan_sha256, "rule": "single use; uncertain retry refused"}).decode()
        )
        claim.flush()
        os.fsync(claim.fileno())
    with (
        Path(evidence["journal"]).open("x", encoding="utf-8") as journal,
        Path(evidence["report"]).open("x", encoding="utf-8") as report,
    ):
        yield journal, report


async def execute_pilot(
    manifest: dict[str, Any],
    *,
    current_plan: dict[str, Any],
    fixtures: FixtureSet,
    db: Any,
    approval_path: Path,
    client_factory: Any = create_aws_clients,
    harness: Any = None,
    summarize_fn: Any = None,
) -> dict[str, Any]:
    """Claim an approved plan ONCE, before any AWS or DB action; persist every admitted call.

    A failed/uncertain run cannot reset its budget by changing output paths or
    replaying this manifest. There is no automatic resume or refreshed allowance.
    """
    approved = require_approval(manifest, current_plan)
    plan = approved.plan
    evidence = plan["evidence"]
    if str(approval_path.resolve()) != evidence["approval_manifest"]:
        raise ApprovalError("approval manifest path differs from the approved single-use execution path")
    if digest(json.loads(approval_path.read_text())) != digest(manifest):
        raise ApprovalError("approval file does not contain the supplied immutable manifest")
    current_fixtures = [
        {"id": f["id"], "sha256": content_sha256(f), "clock": fixture_now(f).isoformat()} for f in fixtures.fixtures
    ]
    if fixtures.sha256 != plan["fixture_set_sha256"] or current_fixtures != plan["fixtures"]:
        raise ApprovalError("fixture content or identity changed after approval")
    reports: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "runtime_identity": plan["runtime_identity"],
        "source_sha": plan["source_sha"],
        "plan_sha256": approved.sha256,
        "status": "running",
        "reports": reports,
        "acceptance": "report-only",
        "progress": {"planned_runs": len(CELLS) * 3, "attempted_runs": 0, "completed_runs": 0},
    }
    with execution_files(evidence, approved.sha256) as (journal_stream, report_stream):

        def journal(event):
            journal_stream.write(canonical_bytes(event).decode() + "\n")
            journal_stream.flush()
            os.fsync(journal_stream.fileno())

        def persist():
            report["budget"] = budget.describe()
            report["observed_usage"] = budget.observed(plan["budget"]["price_ceilings"])
            report_stream.seek(0)
            report_stream.truncate()
            report_stream.write(canonical_bytes(report).decode() + "\n")
            report_stream.flush()
            os.fsync(report_stream.fileno())

        budget = SharedPilotBudget(
            max_calls=plan["budget"]["max_calls"], max_usd=plan["budget"]["max_usd"], journal=journal
        )
        journal({"event": "approved-start", "plan_sha256": approved.sha256})
        persist()
        try:
            sts, runtime = client_factory(approved)
            provider = ApprovedBedrock(approved, sts=sts, runtime=runtime, budget=budget)
            if plan["database"].get("create_schema"):
                from memory_evaluation.eval.standalone import HISTORY

                await db.create_schema(HISTORY)
            if harness is None:
                from memory_evaluation.eval.answering import run_read_to_answer

                harness = run_read_to_answer
            from kai_memory import ports
            from memory_evaluation.eval.providers import TokenHashEmbedder
            from memory_evaluation.eval.standalone import recorded_host
            from kai_memory.sdk import _parse_array, _parse_object

            for cell in plan["cells"]:
                for repeat in range(plan["repeats"]):
                    report["progress"]["attempted_runs"] += 1
                    extractor = RecordedExtractor() if cell["extraction"] == "recorded" else PilotExtractor(provider)
                    embedder = TokenHashEmbedder() if cell["embedding"] == "lexical" else PilotEmbedder(provider)
                    descriptor = {"name": embedder.name, "kind": cell["embedding"], "cache": "disabled"}
                    with (
                        recorded_host(fixture_now(fixtures.fixtures[0])),
                        ports.scoped(
                            {
                                "parse_json_object_response": _parse_object,
                                "parse_json_array_response": _parse_array,
                            },
                            inherit=True,
                        ),
                    ):
                        result = await harness(
                            fixtures,
                            db,
                            answer_fn=PilotAnswerer(provider),
                            extractor=extractor,
                            arms=PILOT_ARMS,
                            embedding_adapter=embedder,
                            embedding_descriptor=descriptor,
                            fixture_clock=fixture_now,
                            fail_fast=True,
                        )
                    reports.append({"cell": cell, "repeat": repeat + 1, "report": result})
                    validate_run(result, fixtures)
                    report["progress"]["completed_runs"] += 1
                    persist()
            report["status"] = "complete"
        except Exception as error:
            report.update(status="failed", error_type=type(error).__name__)
        except BaseException as error:
            report.update(status="interrupted", error_type=type(error).__name__)
            raise
        finally:
            if summarize_fn is None:
                from memory_evaluation.eval.diagnostics import summarize_repeats

                summarize_fn = summarize_repeats
            diagnostics = []
            for cell in plan["cells"]:
                partial = [r["report"] for r in reports if r["cell"] == cell]
                try:
                    from kai_memory import ports

                    with ports.scoped({"clock": lambda: fixture_now(fixtures.fixtures[0])}, inherit=True):
                        summary = summarize_fn(partial, fixtures) if partial else {"status": "not-run", "n_reports": 0}
                except Exception as error:
                    summary = {
                        "status": "invalid-partial",
                        "n_reports": len(partial),
                        "error_type": type(error).__name__,
                    }
                diagnostics.append({"cell": cell, "diagnostics": summary})
            report["repeat_diagnostics"] = diagnostics
            persist()
    return report
