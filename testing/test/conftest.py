import pytest

@pytest.fixture(scope="module")
def my_root_fixture():
    return "root_fixture_token"
