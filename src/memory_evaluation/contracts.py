"""Backend extension contract. Payloads are plain values, never SDK/ORM objects."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

from memory_evaluation.answers import AnswerRequest


@dataclass(frozen=True)
class BackendDescriptor:
    name: str
    version: str
    artifact_sha256: str | None = None
    capabilities: tuple[str, ...] = ("read_to_answer",)
    configuration: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    source_revision: str
    clock: str
    seed: int
    answerer: dict[str, Any]
    judge: dict[str, Any] | None = None


@dataclass(frozen=True)
class Retrieval:
    context: str
    trace_ids: tuple[str, ...] = ()


class FixtureSession(Protocol):
    async def retrieve(self, request: AnswerRequest, question: dict[str, Any]) -> Retrieval: ...
    async def state_receipt(self) -> str: ...


class Backend(Protocol):
    def describe(self) -> BackendDescriptor: ...
    def open_fixture(
        self, fixture: dict[str, Any], config: RunConfig
    ) -> AbstractAsyncContextManager[FixtureSession]: ...
