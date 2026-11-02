"""Normalize only run identity, elapsed timing and generated database identifiers."""

import json


def project(board):
    result = json.loads(json.dumps(board))
    for key in ("generated_at", "source_sha"):
        result["header"].pop(key, None)
    result["header"]["fixture_set"].pop("path", None)
    for fixture in result["fixtures"]:
        fixture.pop("duration_s", None)
        for key in ("prohibited_hits", "unexpected"):
            for hit in fixture[key]:
                if "trace_id" in hit:
                    hit["trace_id"] = "<generated-trace-id>"
        for miss in fixture["attribution"]["misses"]:
            if "subject_id" in miss:
                miss["subject_id"] = "<generated-subject-id>"
    return result


def differences(before, after, path=()):
    """Retain every changed field, including ranks; never filter a difference."""
    if type(before) is type(after) and isinstance(before, dict) and before.keys() == after.keys():
        return [row for key in before for row in differences(before[key], after[key], (*path, key))]
    if type(before) is type(after) and isinstance(before, list) and len(before) == len(after):
        return [row for index, (left, right) in enumerate(zip(before, after))
                for row in differences(left, right, (*path, index))]
    return [] if before == after else [{"path": list(path), "before": before, "after": after}]


def compare_isolated(board, historical, contract):
    actual = project(board)
    expected = json.loads(json.dumps(historical["scoreboard"]))
    for change in contract["differences"]:
        target = expected
        for key in change["path"][:-1]:
            target = target[key]
        key = change["path"][-1]
        if target[key] != change["before"]:
            raise ValueError("historical baseline does not match declared instrument change")
        target[key] = change["after"]
    return {
        "instrument": contract["format"],
        "isolated_baseline_passed": actual == expected,
        "historical_exact_parity": actual == historical["scoreboard"],
        "historical_differences": differences(historical["scoreboard"], actual),
    }
