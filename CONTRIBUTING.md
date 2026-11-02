# Contributing

Report reproducible bugs and propose changes in
[GitHub issues](https://github.com/dimabalony/memory-evaluation/issues).
Bootstrap tracking: [issue #1](https://github.com/dimabalony/memory-evaluation/issues/1).

Run `python -m pytest tests/neutral` without the SDK; run the full suite with the
pinned SDK artifact installed. PostgreSQL integration tests require a fresh local
pgvector database through `EVAL_DATABASE_URL`; they fail when explicitly invoked
without that URL. Never use production storage or an existing populated database.

Keep backend-neutral imports free of `kai_memory`, `app`, and SQLAlchemy.
SDK white-box lifecycle measurements belong in the optional adapter, not in a
fake generic schema. Preserve historical score/prompt/fixture versions when
refactoring. Changes to measurement semantics require explicit instrument versions
and before/after evidence. Do not silently reinterpret unsupported measurements.

Fixture additions require explicit provenance, license, and privacy review.
Generate new synthetic cases; never relabel personal or production content as
synthetic. Keep fixture hashes and `FIXTURE_LICENSES.json` synchronized.
Contributions are provided under GPL-3.0-or-later.
