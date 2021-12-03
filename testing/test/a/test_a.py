import logging
import pytest

def test_a1(my_root_fixture, my_a_fixture):
    """Fixtures are automatically imported from `conftest.py`.
    Tests can use fixtures declared from the directory or
    parent directory the test is in."""
    logging.info("in test_a1()")
    logging.info(f"I have {my_root_fixture}")
    logging.info(f"I have {my_a_fixture}")

def test_a2(fixture_2):
    logging.info("in test_a2()")
    logging.info("fixture_2 evaluates to: %s", fixture_2)

def test_a3(fixture_3):
    logging.info("in test_a3()")
    logging.info("fixture_3 evaluates to: %s", fixture_3)

"""
Tests can't use fixtures declared in a directory that is not the current or parent.

```
def test_a2(my_b_fixture):
    logging.info("in test_a2()")
    logging.info(f"I have {my_b_fixture}")
```
"""

