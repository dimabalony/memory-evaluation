"""How a labelled expectation matches a stored trace (shared by the runner and the scorer).

A trace view is the runner's plain-dict projection of one ``memory_traces`` row.
Matching is case-insensitive. Latin and Cyrillic values retain word boundaries;
Han/Kana permit unspaced lexical spans. This does not infer meaning or morphology.
"""

from __future__ import annotations

import re
from typing import Any


def _contains_all(haystack: str, needles: list[str]) -> bool:
    folded = haystack.casefold()
    return all(text_contains(folded, n) for n in needles)


def matching_fact(spec: dict[str, Any], view: dict[str, Any]) -> dict[str, Any] | None:
    """The first fact of ``view`` whose value carries ``spec['fact']['value_contains']``."""
    want = ((spec.get("fact") or {}).get("value_contains") or "").casefold()
    if not want:
        return None
    for fact in view.get("facts") or []:
        if text_contains(str(fact.get("value") or ""), want):
            return dict(fact)
    return None


def write_matches(write: dict[str, Any], view: dict[str, Any]) -> bool:
    """An expected write matches when every condition it declares holds."""
    if write.get("content_contains") and not _contains_all(view.get("content") or "", write["content_contains"]):
        return False
    return not (write.get("fact") and matching_fact(write, view) is None)


def prohibited_matches(spec: dict[str, Any], view: dict[str, Any]) -> bool:
    """A prohibited write is hit when its content OR its fact value shows up anywhere."""
    if spec.get("content_contains") and _contains_all(view.get("content") or "", spec["content_contains"]):
        return True
    return bool(spec.get("fact")) and matching_fact(spec, view) is not None


def _han_or_kana(char: str) -> bool:
    point = ord(char)
    return (
        0x3400 <= point <= 0x4DBF
        or 0x4E00 <= point <= 0x9FFF
        or 0x20000 <= point <= 0x2FA1F
        or 0x3040 <= point <= 0x30FF
        or 0x31F0 <= point <= 0x31FF
        or 0xFF66 <= point <= 0xFF9D
    )


def text_contains(text: str, needle: str) -> bool:
    """Bound space-delimited words; Han/Kana may adjoin without spaces.

    This is lexical matching, not morphological or semantic analysis. Hangul
    deliberately retains the existing word boundaries.
    """
    if not needle.strip():
        return False
    folded, want = text.casefold(), needle.casefold()
    for match in re.finditer(re.escape(want), folded):
        before = folded[match.start() - 1] if match.start() else ""
        after = folded[match.end()] if match.end() < len(folded) else ""
        left_ok = not before or not (before.isalnum() or before == "_") or _han_or_kana(before) or _han_or_kana(want[0])
        right_ok = not after or not (after.isalnum() or after == "_") or _han_or_kana(after) or _han_or_kana(want[-1])
        if left_ok and right_ok:
            return True
    return False
