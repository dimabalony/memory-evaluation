"""Historical SDK lifecycle instruments; install the ``sdk`` extra to execute them."""

from importlib import import_module

_EXPORTS = {
    "DisposableDB": "disposable_db",
    "build_answer_scoreboard": "answering",
    "run_read_to_answer": "answering",
    "build_scoreboard": "scoring",
    "run_write_readback": "runner",
    "load_fixture_set": "fixtures",
    "select_fixture_subset": "fixtures",
}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
