"""One monotone reservation ledger for an explicitly approved pilot (DB-881).

Reservations use UTF-8 request bytes as a conservative text-token ceiling plus
an endpoint-envelope allowance. They are never released, including on failure.
Observed provider usage is separate evidence, never extra spending permission.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


class PilotBudgetExceeded(RuntimeError):
    """A call was refused before reaching a paid endpoint."""


def positive_decimal(value: str | int | Decimal) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("budget and price values must be finite positive decimals") from None
    if not result.is_finite() or result <= 0:
        raise ValueError("budget and price values must be finite positive decimals")
    return result


@dataclass(frozen=True)
class PriceCeiling:
    input_usd_per_mtok: Decimal
    output_usd_per_mtok: Decimal

    def __post_init__(self) -> None:
        positive_decimal(self.input_usd_per_mtok)
        if not self.output_usd_per_mtok.is_finite() or self.output_usd_per_mtok < 0:
            raise ValueError("output price must be finite and nonnegative")

    @classmethod
    def from_dict(cls, value: dict[str, str]) -> PriceCeiling:
        return cls(positive_decimal(value["input_usd_per_mtok"]), Decimal(str(value["output_usd_per_mtok"])))


@dataclass(frozen=True)
class Reservation:
    call_id: int
    role: str
    model: str
    input_token_ceiling: int
    output_token_ceiling: int
    reserved_usd: Decimal


class SharedPilotBudget:
    """Thread-safe, fail-closed ceiling shared across cells, repeats and roles.

    No refund/settlement method exists. ``observe`` records token usage, including
    cache counters if present, without modifying the reservation total. The USD
    cap bounds reservations at the frozen approved price ceilings, not AWS bills
    under an unspecified future tariff. No cache discount is assumed.
    """

    def __init__(self, *, max_calls: int, max_usd: str, envelope_tokens: int = 4096, journal: Any = None) -> None:
        if type(max_calls) is not int or max_calls <= 0:
            raise ValueError("max_calls must be a positive integer")
        if type(envelope_tokens) is not int or envelope_tokens < 4096:
            raise ValueError("endpoint envelope allowance must be at least 4096 tokens")
        self.max_calls = max_calls
        self.max_usd = positive_decimal(max_usd)
        self.envelope_tokens = envelope_tokens
        self._journal = journal
        self._lock = threading.Lock()
        self._reserved = Decimal(0)
        self._calls: list[dict[str, Any]] = []

    def reserve(
        self, *, role: str, model: str, body: bytes, max_output_tokens: int, price: PriceCeiling
    ) -> Reservation:
        if role not in {"answer", "extraction", "embedding"}:
            raise ValueError("unknown paid-call role")
        if not isinstance(body, bytes) or type(max_output_tokens) is not int or max_output_tokens < 0:
            raise ValueError("a call needs serialized request bytes and a nonnegative output-token cap")
        inputs = len(body) + self.envelope_tokens
        charge = (inputs * price.input_usd_per_mtok + max_output_tokens * price.output_usd_per_mtok) / 1_000_000
        with self._lock:
            if len(self._calls) >= self.max_calls or self._reserved + charge > self.max_usd:
                raise PilotBudgetExceeded("shared pilot reservation cap reached; call was not sent")
            call = Reservation(len(self._calls) + 1, role, model, inputs, max_output_tokens, charge)
            self._reserved += charge
            self._calls.append(
                {
                    "call_id": call.call_id,
                    "role": role,
                    "model": model,
                    "input_token_ceiling": inputs,
                    "output_token_ceiling": max_output_tokens,
                    "reserved_usd": str(charge),
                    "status": "reserved",
                    "provider_usage": None,
                    "request_sha256": hashlib.sha256(body).hexdigest(),
                    "usage_status": "unknown",
                    "cache_policy": "no cache controls requested; no discount assumed",
                }
            )
            if self._journal is not None:
                self._journal({"event": "reserve", **self._calls[-1]})
            return call

    def observe(self, reservation: Reservation, *, usage: dict[str, int] | None, error_type: str | None = None) -> None:
        with self._lock:
            index = reservation.call_id - 1
            if not 0 <= index < len(self._calls) or self._calls[index]["status"] != "reserved":
                raise ValueError("unknown or already observed reservation")
            row = self._calls[index]
            row["status"] = "failed" if error_type else "succeeded"
            row["error_type"] = error_type
            row["provider_usage"] = dict(usage) if usage is not None else None
            row["usage_status"] = "provider-reported" if usage is not None else "unknown"
            if self._journal is not None:
                self._journal({"event": "observe", **row})
            # No subtraction here: consumed/failed calls never reopen permission.

    def observed(self, prices: dict[str, dict[str, str]]) -> dict[str, Any]:
        """Provider-reported usage/cost estimates; missing/failed usage stays unknown."""
        with self._lock:
            groups: dict[str, dict[str, Any]] = {}
            for row in self._calls:
                item = groups.setdefault(
                    row["role"],
                    {
                        "calls": 0,
                        "calls_with_usage": 0,
                        "calls_missing_usage": 0,
                        "failed_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "observed_cost_usd_estimate": Decimal(0),
                    },
                )
                item["calls"] += 1
                item["failed_calls"] += row["status"] == "failed"
                usage = row["provider_usage"]
                if usage is None:
                    item["calls_missing_usage"] += 1
                    continue
                item["calls_with_usage"] += 1
                for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
                    item[key] += usage.get(key, 0)
                rates = PriceCeiling.from_dict(prices[row["model"]])
                # No cache discounts; cache writes use a conservative 2x input rate.
                input_equivalent = (
                    usage.get("input_tokens", 0)
                    + usage.get("cache_read_input_tokens", 0)
                    + 2 * usage.get("cache_creation_input_tokens", 0)
                )
                item["observed_cost_usd_estimate"] += (
                    input_equivalent * rates.input_usd_per_mtok
                    + usage.get("output_tokens", 0) * rates.output_usd_per_mtok
                ) / 1_000_000
            for item in groups.values():
                item["observed_cost_usd_estimate"] = str(item["observed_cost_usd_estimate"])
                item["usage_status"] = "partial-unknown" if item["calls_missing_usage"] else "provider-reported"
            return {
                "by_role": groups,
                "observed_cost_usd_estimate": str(
                    sum((Decimal(g["observed_cost_usd_estimate"]) for g in groups.values()), Decimal(0))
                ),
                "usage_status": "partial-unknown"
                if any(g["calls_missing_usage"] for g in groups.values())
                else "provider-reported"
                if groups
                else "no-paid-calls",
                "billing_claim": "observed usage estimate at frozen rates, not an invoice; unknown usage is excluded",
                "cache_counters": "missing counters remain unknown per call; aggregate sums known counters",
            }

    def describe(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_calls": self.max_calls,
                "max_usd": str(self.max_usd),
                "calls": len(self._calls),
                "reserved_usd": str(self._reserved),
                "remaining_reservation_usd": str(self.max_usd - self._reserved),
                "input_bound": "serialized UTF-8 bytes + endpoint envelope tokens",
                "endpoint_envelope_tokens": self.envelope_tokens,
                "billing_claim": "reservation cap at approved price ceilings; observed usage separate",
                "records": [dict(row) for row in self._calls],
            }
