"""Historical SDK lifecycle instruments; install the ``sdk`` extra to execute them."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .answering import build_answer_scoreboard as build_answer_scoreboard
    from .answering import run_read_to_answer as run_read_to_answer
    from .disposable_db import DisposableDB as DisposableDB
    from .fixtures import load_fixture_set as load_fixture_set
    from .fixtures import select_fixture_subset as select_fixture_subset
    from .runner import run_write_readback as run_write_readback
    from .scoring import build_scoreboard as build_scoreboard

_EXPORTS = {
    "DisposableDB": "disposable_db",
    "build_answer_scoreboard": "answering",
    "run_read_to_answer": "answering",
    "build_scoreboard": "scoring",
    "run_write_readback": "runner",
    "load_fixture_set": "fixtures",
    "select_fixture_subset": "fixtures",
}
__all__ = [
    "DisposableDB",
    "build_answer_scoreboard",
    "run_read_to_answer",
    "build_scoreboard",
    "run_write_readback",
    "load_fixture_set",
    "select_fixture_subset",
]

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
