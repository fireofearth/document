import pytest

@pytest.fixture(scope="module")
def my_b_fixture():
    return "b_fixture_token"

