"""Automatic execution-logger bootstrap.

Python imports ``sitecustomize`` at interpreter start-up if this directory is on
``sys.path``. When the process is running inside this project we initialise the
run logger immediately, so *every* invocation (``python -m src.train``,
``python test_model.py``, ``python local_train.py``, ...) is logged with its own
execution id — no code changes required in the entry point.

To guarantee it is picked up, run from the project root with the root on
``PYTHONPATH`` (``set PYTHONPATH=.`` / ``export PYTHONPATH=.``) or use the
``python -m src.run_logger`` launcher. It is a no-op anywhere else and never
raises.
"""

import os
import sys


def _looks_like_project(root):
    return (os.path.isfile(os.path.join(root, "src", "run_logger.py"))
            and os.path.isfile(os.path.join(root, "src", "config.py")))


def _bootstrap():
    if os.environ.get("RUN_LOG_DISABLE") == "1":
        return
    # Candidate roots: cwd, this file's dir, and anything already on sys.path.
    candidates = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    candidates += [p for p in sys.path if p]
    root = next((p for p in candidates if p and _looks_like_project(p)), None)
    if root is None:
        return
    if root not in sys.path:
        sys.path.insert(0, root)
    # Skip trivial/REPL invocations that would just create noise folders.
    argv = sys.argv
    trivial = (len(argv) < 2) or (argv[1] in ("-c", "-") )
    if trivial:
        return
    try:
        from src.run_logger import init_run_logger
        init_run_logger()
    except Exception:
        pass


try:
    _bootstrap()
except Exception:
    pass
