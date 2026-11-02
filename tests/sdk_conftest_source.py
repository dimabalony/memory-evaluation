from datetime import UTC, datetime
import pytest
from kai_memory import ports
from memory_evaluation.eval.standalone import recorded_host

@pytest.fixture(autouse=True)
def synthetic_host():
    for name in ports._HOST_SYMBOLS:
        ports.__dict__.pop(name, None)
    ports._BOUND.clear()
    with recorded_host(datetime.now(UTC)) as names:
        pass
    previous = ports.set_default_resolver(lambda name: names.get(name, ports.UNBOUND))
    try:
        yield
    finally:
        ports.set_default_resolver(previous)
        ports._BOUND.clear()
        for name in ports._HOST_SYMBOLS:
            ports.__dict__.pop(name, None)
