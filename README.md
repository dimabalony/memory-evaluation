# memory-evaluation

One runner compares memory backends using identical questions and answer/judge
functions. The core and independent no-memory backend have no runtime dependency
on a memory SDK, database, application, or provider account.

```python
from memory_evaluation.backends.no_memory import NoMemoryBackend
from memory_evaluation.contracts import RunConfig
from memory_evaluation.eval.fixtures import load_fixture_set
from memory_evaluation.runner import evaluate

fixtures = load_fixture_set("tests/fixtures/memory_lifecycle/v1")

async def answer(request):
    # Supply your model adapter here; this example makes no provider calls.
    return "I don't know."

report = await evaluate(
    fixtures, NoMemoryBackend(), answer_fn=answer,
    config=RunConfig(
        source_revision="your-exact-source-revision",
        clock="2026-01-01T00:00:00+00:00", seed=42,
        answerer={"model": "recorded-abstention", "prompt_version": "example-v1"},
    ),
)
```

Run asynchronous examples inside your application's event loop. Fixture files
are included in the source distribution and repository. An abstaining example is
an executable demonstration, not evidence of model quality.

## Backends and instruments

Implement `Backend.describe()` and `Backend.open_fixture()` from
`memory_evaluation.contracts`. A fixture session supplies `retrieve()` and a
stable `state_receipt()`. Payloads are plain Python values. The runner checks
that retrieval and answering leave backend state unchanged, records common input
hashes, and uses the canonical lexical answer instrument. See the independent
dictionary backend in `tests/neutral/test_runner.py` for a complete extension.

`NoMemoryBackend` does not create a store, extract memories, or generate
embeddings. Unsupported write/readback measurements are explicitly unmeasured.
The report `memory-backend-comparison/v1` retains valid, invalid, unrun, and
unmeasured denominators; empty/failing answers are never counted as correct.
Provider usage and costs remain unknown unless your adapter separately records
them. A fixture schema and deterministic lexical score do not establish semantic
learning efficacy or clinical/personal-advice quality.

The optional `sdk` extra pins `recoleva==0.3.0` (import namespace `kai_memory`).
`memory_evaluation.backends.sdk.SDKBackend` uses a disposable local PostgreSQL
with pgvector and requires the installed SDK wheel SHA-256. Use only a dedicated,
empty database. Its fixture session reuses the canonical SDK replay and retrieval
instrument. The original Layer A/B APIs remain at `memory_evaluation.eval`:
`DisposableDB`, `run_write_readback`, `build_scoreboard`, `run_read_to_answer`,
`build_answer_scoreboard`, and fixture loaders. Existing scoreboard versions,
fixture identities, canonical prompts, and denominator rules remain intact.
The historical matched `no_memory` arm still replays the same SDK store and omits
retrieval; it is distinct from the independent `NoMemoryBackend`.

## Behavioral scoring

`memory_evaluation.elephant_scorer` is the single canonical ELEPHANT/GRACE prompt
and parser implementation. `score_response(..., judge=async_judge)` receives an
explicit async callable accepting the rendered prompt and returning text.
Applications own provider setup, authentication, telemetry, and spending limits.
Failed or malformed verdicts carry `ok=False` and must be excluded from means.

Included ELEPHANT stimuli are authored representative examples, not the paper's
dataset. The five-axis rubric alone does not make a model run paper-comparable.
No personal GRACE fixtures, user persona, production transcripts, or baseline
results are distributed. An externally supplied GRACE fixture requires its own
provenance and rights review. Prompt versions are preserved, not reset on release.

## Install and verification

Install the wheel from [GitHub Releases](https://github.com/dimabalony/memory-evaluation/releases)
and verify it against the release checksum. Installing from source:

```sh
python -m pip install '.[test]'
python -m pytest tests/neutral
# Install the pinned recoleva wheel before requesting the optional SDK extra.
python -m pip install '.[sdk,test]'
python -m pytest
```

`SOURCE_PROVENANCE.json` records the reviewed seed hashes and transformations;
`FIXTURE_LICENSES.json` pins every included fixture's bytes and license. Release
checksums identify built archives. A source receipt is not a test or publication
attestation. No provider is called by the test suite. Optional pilot tooling
requires an explicit approval contract and never grants spending authority by
being installed.

Licensed under GPL-3.0-or-later. See [NOTICE](NOTICE), [CONTRIBUTING.md](CONTRIBUTING.md),
and [SECURITY.md](SECURITY.md).
