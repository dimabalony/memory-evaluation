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
