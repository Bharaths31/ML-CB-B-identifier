"""Unified per-execution logging for every python entry point.

Each process gets one execution folder::

    logs/<exec_id>/
        manifest.json     argv, cwd, python, git commit, env, timing, exit code
        config.json       snapshot of every constant in src.config
        run.log           human-readable consolidated log (incl. tee'd stdio)
        events.jsonl      every structured event (one JSON object per line)
        training.jsonl    per-epoch loss + validation metrics
        data.jsonl        dataset download / split / inventory events
        test.jsonl        per-prediction records
        export.jsonl      exported artifact paths + parity verdicts

``exec_id`` is ``YYYYmmdd-HHMMSS-<4 hex>`` unless supplied (``--exec-id`` or the
``RUN_EXEC_ID`` env var). The logger is a process-wide singleton; calling
``init_run_logger`` more than once is a no-op so a ``sitecustomize`` bootstrap
and an explicit call in ``main()`` coexist safely.

Usage from any module::

    from .run_logger import init_run_logger, log_event, log_metrics
    init_run_logger(module="src.train")
    log_event("epoch_done", category="training", epoch=3, top1=0.61)
    log_metrics(metrics, epoch=3, tag="raw")
"""

import atexit
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid

from . import config as _config

_LOGGER = None
_STDIO_INSTALLED = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    try:
        return obj.item()          # torch/numpy scalar
    except Exception:
        pass
    try:
        return list(obj)
    except Exception:
        pass
    return str(obj)


def _dumps(record):
    return json.dumps(record, default=_json_default, ensure_ascii=False)


def make_exec_id(tag=None):
    """``YYYYmmdd-HHMMSS-<4 hex>`` — unique per process, filesystem-safe."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:4]
    if tag:
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in str(tag))
        return f"{ts}-{safe}-{short}"
    return f"{ts}-{short}"


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=_config.PROJECT_ROOT)
        return out.stdout.strip() or None
    except Exception:
        return None


class _Tee:
    """Write to the original stream AND a log file, collapsing tqdm \\r updates."""

    def __init__(self, stream, fileobj):
        self._stream = stream
        self._file = fileobj
        self._buf = ""

    def write(self, data):
        try:
            self._stream.write(data)
        except Exception:
            pass
        if self._file is None or self._file.closed:
            return
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.replace("\r", "").rstrip()
            if line:
                try:
                    self._file.write(line + "\n")
                except ValueError:
                    return      # file closed mid-write (shutdown race)
        # progress bars emit many \r without \n — keep only the last frame
        if len(self._buf) > 4000:
            self._buf = self._buf.rsplit("\r", 1)[-1][-1000:]
        try:
            self._file.flush()
        except Exception:
            pass

    def flush(self):
        for s in (self._stream, self._file):
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------

class RunLogger:
    def __init__(self, exec_id, log_root=None, module=None, argv=None):
        self.exec_id = exec_id
        self.log_root = log_root or _config.LOG_ROOT
        self.dir = os.path.join(self.log_root, exec_id)
        self.module = module or self._module_from_argv(argv)
        self.argv = list(argv if argv is not None else sys.argv)
        self.start_time = time.time()
        self.exit_code = None
        self._category_files = {}
        self._category_fhs = {}
        self._closed = False

        os.makedirs(self.dir, exist_ok=True)
        self._run_log_path = os.path.join(self.dir, "run.log")
        self._events_path = os.path.join(self.dir, "events.jsonl")
        self._events_fh = open(self._events_path, "a", encoding="utf-8")
        self._run_fh = open(self._run_log_path, "a", encoding="utf-8")

        self._setup_python_logging()
        self._write_manifest()
        self._write_config_snapshot()
        self.event("run_start", category="actions", module=self.module,
                   argv=self.argv, cwd=os.getcwd())

    # -- construction helpers ------------------------------------------------
    @staticmethod
    def _module_from_argv(argv):
        # sys.orig_argv preserves the real command line: ['python','-m','src.train',...]
        orig = list(getattr(sys, "orig_argv", None) or [])
        tokens = orig or list(argv if argv is not None else sys.argv)
        if "-m" in tokens:
            i = tokens.index("-m")
            if i + 1 < len(tokens):
                return tokens[i + 1]
        for tok in tokens:
            if tok.endswith(".py"):
                return os.path.basename(tok)
        return "python"

    def _setup_python_logging(self):
        self.log = logging.getLogger("run_logger")
        self.log.setLevel(getattr(logging, str(_config.LOG_LEVEL).upper(), logging.INFO))
        self.log.propagate = False
        if not self.log.handlers:
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S")
            fh = logging.FileHandler(self._run_log_path, encoding="utf-8")
            fh.setFormatter(fmt)
            self.log.addHandler(fh)
            sh = logging.StreamHandler(sys.__stdout__)
            sh.setFormatter(fmt)
            self.log.addHandler(sh)

    def _write_manifest(self):
        env_keys = ("CUDA_VISIBLE_DEVICES", "KAGGLE_USERNAME", "PYTHONHASHSEED",
                    "RUN_EXEC_ID", "PYTHONPATH")
        manifest = {
            "exec_id": self.exec_id,
            "module": self.module,
            "argv": self.argv,
            "cwd": os.getcwd(),
            "project_root": _config.PROJECT_ROOT,
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "git_commit": _git_commit(),
            "env": {k: os.environ.get(k) for k in env_keys if os.environ.get(k)},
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.start_time)),
            "hostname": platform.node(),
        }
        self._write_json("manifest.json", manifest)

    def _write_config_snapshot(self):
        snapshot = {}
        for name in dir(_config):
            if name.startswith("_") or not name.isupper():
                continue
            val = getattr(_config, name)
            if callable(val) or isinstance(val, type(os)):
                continue
            try:
                json.dumps(val, default=_json_default)
                snapshot[name] = val
            except Exception:
                snapshot[name] = str(val)
        self._write_json("config.json", snapshot)

    def _write_json(self, filename, obj):
        try:
            with open(os.path.join(self.dir, filename), "w", encoding="utf-8") as f:
                json.dump(obj, f, default=_json_default, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # -- event API -----------------------------------------------------------
    def event(self, event, category="general", level="info", **fields):
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "t": round(time.time() - self.start_time, 3),
            "exec_id": self.exec_id,
            "module": self.module,
            "category": category,
            "event": event,
        }
        record.update(fields)
        line = _dumps(record)
        try:
            self._events_fh.write(line + "\n")
            self._events_fh.flush()
        except Exception:
            pass
        # category-specific stream
        if category != "general":
            try:
                fh = self._category_fhs.get(category)
                if fh is None:
                    path = os.path.join(self.dir, f"{category}.jsonl")
                    fh = open(path, "a", encoding="utf-8")
                    self._category_fhs[category] = fh
                fh.write(line + "\n")
                fh.flush()
            except Exception:
                pass
        getattr(self.log, level if level in
                ("debug", "info", "warning", "error", "critical") else "info",
                self.log.info)(f"[{category}] {event}"
                               + (f" {fields}" if fields else ""))
        return record

    def metrics(self, metrics, epoch=None, tag="", category="training", **extra):
        fields = {"epoch": epoch, "tag": tag}
        fields.update({k: v for k, v in metrics.items()})
        fields.update(extra)
        return self.event("metrics", category=category, **fields)

    def log(self, level, message, **fields):  # convenience
        return self.event("log", category="actions", level=level, message=message, **fields)

    def section(self, title, **fields):
        return self.event("section", category="actions", title=title, **fields)

    # -- lifecycle -----------------------------------------------------------
    def close(self, exit_code=0):
        if self._closed:
            return
        self._closed = True
        self.exit_code = exit_code
        duration = round(time.time() - self.start_time, 3)
        self.event("run_end", category="actions", exit_code=exit_code,
                   duration_s=duration)
        # update manifest with end info
        try:
            mpath = os.path.join(self.dir, "manifest.json")
            with open(mpath, encoding="utf-8") as f:
                m = json.load(f)
            m.update({"ended_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "duration_s": duration, "exit_code": exit_code})
            self._write_json("manifest.json", m)
        except Exception:
            pass
        # Print summary BEFORE closing file handles (sys.stdout may be a
        # _Tee writing to self._run_fh — closing first triggers ValueError).
        print(f"[logger] exec {self.exec_id} -> {self.dir} "
              f"({duration:.1f}s, exit={exit_code})")
        # Restore original stdio so later prints don't hit the closed _Tee
        global _STDIO_INSTALLED
        if _STDIO_INSTALLED:
            try:
                if isinstance(sys.stdout, _Tee):
                    sys.stdout = sys.stdout._stream
                if isinstance(sys.stderr, _Tee):
                    sys.stderr = sys.stderr._stream
            except Exception:
                pass
            _STDIO_INSTALLED = False
        for fh in [self._events_fh, self._run_fh] + list(self._category_fhs.values()):
            try:
                fh.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# module-level API
# ---------------------------------------------------------------------------

def init_run_logger(exec_id=None, module=None, argv=None, log_root=None,
                    capture_stdio=None):
    """Initialise (once) and return the process RunLogger.

    Idempotent: a second call returns the existing logger (updating module if
    given). Disabled by ``LOG_TO_FILE=False`` or ``RUN_LOG_DISABLE=1``.
    """
    global _LOGGER, _STDIO_INSTALLED
    if _LOGGER is not None:
        if module:
            _LOGGER.module = module
        return _LOGGER
    if not _config.LOG_TO_FILE or os.environ.get("RUN_LOG_DISABLE") == "1":
        return None
    exec_id = exec_id or os.environ.get(_config.LOG_EXEC_ID_ENV) or make_exec_id()
    try:
        _LOGGER = RunLogger(exec_id, log_root=log_root, module=module, argv=argv)
    except Exception as exc:  # never let logging break a run
        print(f"[logger] disabled (init failed: {exc})")
        _LOGGER = None
        return None
    atexit.register(lambda: _LOGGER and _LOGGER.close(0))
    _install_excepthook()
    if capture_stdio is None:
        capture_stdio = _config.LOG_CAPTURE_STDIO
    if capture_stdio and not _STDIO_INSTALLED:
        try:
            sys.stdout = _Tee(sys.stdout, _LOGGER._run_fh)
            sys.stderr = _Tee(sys.stderr, _LOGGER._run_fh)
            _STDIO_INSTALLED = True
        except Exception:
            pass
    return _LOGGER


def get_logger():
    return _LOGGER if _LOGGER is not None else init_run_logger()


def log_event(event, category="general", level="info", **fields):
    logger = get_logger()
    if logger is None:
        return None
    return logger.event(event, category=category, level=level, **fields)


def log_metrics(metrics, epoch=None, tag="", category="training", **extra):
    logger = get_logger()
    if logger is None:
        return None
    return logger.metrics(metrics, epoch=epoch, tag=tag, category=category, **extra)


def finish_run(exit_code=0):
    if _LOGGER is not None:
        _LOGGER.close(exit_code)


def _install_excepthook():
    if getattr(_install_excepthook, "_done", False):
        return
    _install_excepthook._done = True
    original = sys.excepthook

    def _hook(exc_type, exc, tb):
        if _LOGGER is not None:
            _LOGGER.event("uncaught_exception", category="actions",
                          level="error", type=exc_type.__name__, message=str(exc),
                          traceback="".join(traceback.format_exception(
                              exc_type, exc, tb)))
            _LOGGER.close(1)
        original(exc_type, exc, tb)

    sys.excepthook = _hook
