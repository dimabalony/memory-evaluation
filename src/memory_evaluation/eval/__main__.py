"""Recorded evaluation, independent of any application registry or paid provider."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from memory_evaluation.eval import DisposableDB, build_scoreboard, load_fixture_set, run_write_readback
from memory_evaluation.eval.standalone import HISTORY, recorded_host
from memory_evaluation.eval.fixtures import select_fixture_subset


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--db-url", help="Explicit empty loopback PostgreSQL database")
    parser.add_argument("--create-schema", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--subset", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source-sha", default="unknown")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report-out", type=Path)
    return parser


def fixture_end(fixture):
    return datetime.fromisoformat(fixture["start_at"]) + timedelta(
        minutes=max(message["minute"] for message in fixture["messages"]) + 1
    )


async def run(args):
    if args.subset is not None and args.subset < 1:
        raise ValueError("--subset must be at least 1")
    fixtures = load_fixture_set(args.fixtures, ids=args.ids)
    if args.subset is not None:
        chosen = select_fixture_subset(fixtures.fixtures, args.subset, args.seed)
        fixtures = load_fixture_set(args.fixtures, ids=[f["id"] for f in chosen])
    if args.validate_only:
        print(json.dumps({"ids": [f["id"] for f in fixtures.fixtures], "sha256": fixtures.sha256}))
        return 0
    if not args.db_url:
        raise ValueError("--db-url is required unless --validate-only")
    db = DisposableDB(args.db_url, production_url=os.environ.get("DATABASE_URL"))
    with recorded_host(max(map(fixture_end, fixtures.fixtures))):
        if args.create_schema:
            await db.create_schema(HISTORY)
        report = await run_write_readback(fixtures, db, fixture_clock=fixture_end)
        board = build_scoreboard(report, fixtures, source_sha=args.source_sha)
    for path, data in ((args.out, board), (args.report_out, report)):
        if path:
            path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(json.dumps(board, indent=2, default=str))
    mutation = board["mutation_check"]
    return int(
        bool(
            board["aggregate"]["n_invalid"]
            or board["aggregate"].get("n_invalid_readback", 0)
            or mutation["residual_rows"]
            or mutation["changed_tables"]
        )
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
