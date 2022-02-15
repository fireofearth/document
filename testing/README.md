
## Introduction

Guide on using PyTest.

Currently describes the use of subdirectories and fixtures in-heritance.

## Quick-start

Install the `pytest` package and call `pytest`.

## Logging

Running `pytest --capture=tee-sys` allows capturing and passing of STDERR and STDOUT,
which is ideal for debugging and using `print()` statements.
However, there is no configuration option for it.
Instead, put this in `pytest.ini`

```
[pytest]
log_cli = True
log_file_level = DEBUG
```

and using `import logging; logging.debug("...")` in the tests.

