import pytest
from .. import MY_STATIC_TOKEN

@pytest.fixture(scope="module")
def my_a_fixture():
    return "a_fixture_token"

@pytest.fixture(scope="module")
def fixture_2():
    """
    can't auto-import variables, etc from parent conftest.py
    use __init__.py instead
    """
    return MY_STATIC_TOKEN

@pytest.fixture(scope="module")
def fixture_3(my_root_fixture):
    """
    It's possible to import fixtures from the parent conftest.py
    """
    return my_root_fixture

