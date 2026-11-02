"""Extractors for the write-to-readback eval: the ``llm_text_fn`` an ExtractionService calls.

Both stand where ``call_llm_text`` stands in production, so the real prompt render,
JSON parsing, slot validation, entity resolution and contradiction handling all run.

* ``RecordedExtractor`` returns the fixture's recorded Haiku response for the
  current checkpoint. It is the CI extractor: no network, no cost. Its scores
  measure the pipeline around the model, never the model ("pipeline-only").
* ``LiveExtractor`` calls the real model through ``ports.call_llm_text`` under a
  call cap and a dollar cap. It runs only from ``scripts/run_memory_eval.py --live``
  against a local database, on demand, never in CI.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, cast

from kai_memory import ports

LIVE_MODEL = "haiku-5.5"
# Routing-aware budget estimates per million tokens (input, output). Haiku 5.5
# can fall back to Bedrock Haiku 4.5 ($1/$5), which costs more than either 5.5
# prompt tier ($0.10/$0.50 <=100K; $0.50/$2.50 >100K). Reserve at the highest
# known leaf rate and assume no cache discount. These are character-based
# estimates, not provider usage or guaranteed dollar caps. Unknown models need
# explicit prices; custom prices remain authoritative.
USD_PER_MTOK: dict[str, tuple[float, float]] = {"haiku-4.5": (1.0, 5.0), "haiku-5.5": (1.0, 5.0)}
_CHARS_PER_TOKEN = 4


class UnexpectedExtractionCall(RuntimeError):
    """Extraction asked the model for a response the fixture did not record."""


class BudgetExceeded(RuntimeError):
    """A live call would pass the run's call or dollar cap."""


@dataclass
class CallBudget:
    """A call cap and a dollar cap for live model calls, checked before each call.

    ``reserve`` refuses a call that could pass either cap, counting the worst case
    (the whole prompt plus ``max_tokens`` of output), and books that worst case at
    once, so a call that raises after consuming tokens stays charged. ``spend``
    settles a call that returned to the estimate for what it actually returned."""

    max_calls: int
    max_usd: float
    usd_per_mtok: tuple[float, float]
    chars_per_token: int = _CHARS_PER_TOKEN
    calls: int = 0
    spent_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.max_calls < 1 or self.max_usd <= 0:
            raise ValueError("a live run needs a positive call cap and dollar cap")

    @classmethod
    def for_model(
        cls, model: str, *, max_calls: int, max_usd: float, usd_per_mtok: tuple[float, float] | None = None
    ) -> CallBudget:
        prices = usd_per_mtok or USD_PER_MTOK.get(model)
        if prices is None:
            raise ValueError(f"no list price known for {model!r}; pass usd_per_mtok=(input, output)")
        return cls(
            max_calls=max_calls,
            max_usd=max_usd,
            usd_per_mtok=prices,
            # Haiku 5.5's tokenizer produces ~30% more tokens than 4.5.
            chars_per_token=3 if model == LIVE_MODEL else _CHARS_PER_TOKEN,
        )

    def _usd(self, input_chars: int, output_chars: int) -> float:
        usd_in, usd_out = self.usd_per_mtok
        return (input_chars * usd_in + output_chars * usd_out) / self.chars_per_token / 1_000_000

    def reserve(self, prompt_chars: int, max_tokens: int) -> float:
        """Admit one call and charge its worst case now; returns the amount charged."""
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"live call cap reached ({self.max_calls})")
        worst_case = self._usd(prompt_chars, max_tokens * self.chars_per_token)
        if self.spent_usd + worst_case > self.max_usd:
            raise BudgetExceeded(
                f"next call could cost ${worst_case:.4f}; ${self.spent_usd:.4f} of ${self.max_usd:.2f} already spent"
            )
        self.calls += 1
        self.spent_usd += worst_case
        return worst_case

    def spend(self, prompt_chars: int, output_chars: int, *, reserved: float) -> None:
        """Replace a returned call's reserved worst case with its estimated cost."""
        self.spent_usd += self._usd(prompt_chars, output_chars) - reserved

    def describe(self) -> dict[str, Any]:
        return {
            "max_calls": self.max_calls,
            "max_usd": self.max_usd,
            "calls": self.calls,
            "spent_usd_estimate": round(self.spent_usd, 4),
            "usd_per_mtok": list(self.usd_per_mtok),
            "chars_per_token_estimate": self.chars_per_token,
        }


@dataclass
class ExtractionCall:
    """One model call made during a checkpoint."""

    checkpoint: str
    prompt: str
    prompt_sha256: str
    response: str | None


@dataclass
class _Armed:
    checkpoint: str
    response: str | None
    consumed: bool = False


@dataclass
class RecordedExtractor:
    """Serves one recorded response per checkpoint, and refuses any other call."""

    calls: list[ExtractionCall] = field(default_factory=list)
    _armed: _Armed | None = None

    def describe(self) -> dict[str, Any]:
        return {"name": "recorded", "model": None, "quality": "pipeline-only"}

    def arm(self, checkpoint: str, response: str | None) -> None:
        self._armed = _Armed(checkpoint, response)

    def consumed(self) -> bool:
        return self._armed is not None and self._armed.consumed

    async def __call__(
        self, provider: Any, prompt: str, max_tokens: int = 2048, span: Any = None, **_: Any
    ) -> str | None:
        armed = self._armed
        if armed is None or armed.consumed:
            raise UnexpectedExtractionCall("extraction made a model call with no recorded response left for it")
        armed.consumed = True
        self.calls.append(ExtractionCall(armed.checkpoint, prompt, _sha256(prompt), armed.response))
        return armed.response


class LiveExtractor:
    """Calls the real model under explicit caps; records what it returned."""

    def __init__(self, max_calls: int, max_usd: float) -> None:
        self.budget = CallBudget.for_model(LIVE_MODEL, max_calls=max_calls, max_usd=max_usd)
        self.calls: list[ExtractionCall] = []
        self._checkpoint = ""
        self._consumed = False

    def describe(self) -> dict[str, Any]:
        return {"name": "live", "model": LIVE_MODEL, "quality": "extractor-and-pipeline", **self.budget.describe()}

    def arm(self, checkpoint: str, response: str | None) -> None:
        # The recorded response is ignored: the live model answers for itself.
        self._checkpoint, self._consumed = checkpoint, False

    def consumed(self) -> bool:
        return self._consumed

    async def __call__(
        self,
        provider: Any,
        prompt: str,
        max_tokens: int = 2048,
        span: Any = None,
        *,
        thinking_mode: str = "disabled",
        **kwargs: Any,
    ) -> str | None:
        reserved = self.budget.reserve(len(prompt), max_tokens)
        response = await ports.call_llm_text(
            provider, prompt, max_tokens=max_tokens, span=span, thinking_mode=thinking_mode, **kwargs
        )
        self.budget.spend(len(prompt), len(response or ""), reserved=reserved)
        self._consumed = True
        self.calls.append(ExtractionCall(self._checkpoint, prompt, _sha256(prompt), response))
        return cast("str | None", response)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
