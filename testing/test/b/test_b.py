import logging
import pytest

def test_b(my_root_fixture, my_b_fixture):
    logging.info("in test_b()")
    logging.info(f"I have {my_root_fixture}")
    logging.info(f"I have {my_b_fixture}")
