import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("scoreboard_projection", ROOT / "scripts/scoreboard_projection.py")
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)
BASELINE = json.loads((ROOT / "tests/baselines/lifecycle-v1.json").read_text())
CONTRACT = json.loads((ROOT / "tests/baselines/lifecycle-isolated-v2.json").read_text())


def isolated_board():
    board = copy.deepcopy(BASELINE["scoreboard"])
    board["fixtures"][6]["readback"]["probes"][0]["expect_ranks"]["stable_preference"] = 1
    return board


def test_isolated_comparison_reports_the_historical_rank_change():
    result = projection.compare_isolated(isolated_board(), BASELINE, CONTRACT)
    assert result["isolated_baseline_passed"]
    assert result["historical_exact_parity"] is False
    assert result["historical_differences"] == CONTRACT["differences"]


def test_shared_index_baseline_is_not_accepted_as_isolated_proof():
    result = projection.compare_isolated(BASELINE["scoreboard"], BASELINE, CONTRACT)
    assert not result["isolated_baseline_passed"]
    assert result["historical_exact_parity"]


@pytest.mark.parametrize("change", ["rank", "omission", "denominator", "label"])
def test_additional_scoreboard_change_fails(change):
    board = isolated_board()
    if change == "rank":
        board["fixtures"][6]["readback"]["probes"][0]["expect_ranks"]["stable_preference"] = 3
    elif change == "omission":
        board["fixtures"][0]["omitted"].append({"unexpected": True})
    elif change == "denominator":
        board["aggregate"]["n_invalid"] += 1
    else:
        board["fixtures"][6]["id"] = "wrong-fixture"
    result = projection.compare_isolated(board, BASELINE, CONTRACT)
    assert not result["isolated_baseline_passed"]
    assert result["historical_differences"] != CONTRACT["differences"]


def test_changed_historical_baseline_is_rejected():
    old = copy.deepcopy(BASELINE)
    old["scoreboard"]["fixtures"][6]["readback"]["probes"][0]["expect_ranks"]["stable_preference"] = 9
    with pytest.raises(ValueError, match="historical baseline"):
        projection.compare_isolated(isolated_board(), old, CONTRACT)
