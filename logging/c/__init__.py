import logging

def run():
    """We did not get a logger associated with this module
    so the warning will be associated with the root logger."""
    logging.warning("in module c")

