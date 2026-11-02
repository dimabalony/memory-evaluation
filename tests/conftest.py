import pytest

@pytest.fixture(autouse=True)
def synthetic_host(request):
    if '/neutral/' in str(request.node.path):
        yield
        return
    from tests.sdk_conftest_source import synthetic_host as original
    yield from original.__wrapped__()
