"""Run infrastructure: run ids / timestamped paths / checkpoint discovery + logging.

Consolidates the former ``src/run_utils.py`` (naming + discovery) and
``src/run_logger.py`` (per-execution logging) into one runtime-support module.

SECTION MAP
  1. Run ids & timestamped paths ... make_run_id, timestamped, unique_path,
                                     find_latest_checkpoint, resolve_checkpoint
  2. Execution logger .............. RunLogger, init_run_logger, log_event,
                                     log_metrics, finish_run

REFERENCES
  - ``src/config.py`` = LOG_ROOT / LOG_* and RUN_ID_FORMAT / TIMESTAMP_OUTPUTS.
  - Imported by src/train.py, src/export.py, src/evaluate.py, src/parity_check.py,
    src/verify.py, local_train.py, test_model.py, server/.
  - ``sitecustomize.py`` auto-initialises the logger for any python process run
    from the project root (with PYTHONPATH=.).
"""

import glob
import os
import re
import time
import uuid

from .config import RUN_ID_FORMAT, TIMESTAMP_OUTPUTS

# Characters illegal (or dangerous) in Windows/POSIX filenames.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_run_id(run_id):
    """Make a run id safe to embed in a filename.

    Replaces illegal characters with ``-``, collapses repeats, and strips
    leading/trailing dots and spaces (Windows rejects those on directories).
    """
    cleaned = _UNSAFE.sub("-", str(run_id))
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip(" .-")
    return cleaned or "run"


def make_run_id(run_tag=None, fmt=None):
    """Return a sanitized explicit run tag, or an auto-incremented version tag (v5, v6...)."""
    if run_tag:
        return sanitize_run_id(run_tag)
    if not TIMESTAMP_OUTPUTS:
        return "latest"
    
    # Auto-increment version mechanic
    version_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "VERSION.txt")
    current_version = 4
    if os.path.exists(version_file):
        try:
            with open(version_file, "r") as f:
                content = f.read().strip().lower().replace("v", "")
                current_version = int(content)
        except ValueError:
            pass
    
    next_version = current_version + 1
    with open(version_file, "w") as f:
        f.write(f"v{next_version}")

    # Unique modifier: version + MMDD-HHMM + 4-char random, so two runs with the
    # same version/time still never collide. Example: v7_1006-1530_a3f2
    stamp = time.strftime("%m%d-%H%M")
    uniq = uuid.uuid4().hex[:4]
    return sanitize_run_id(f"v{next_version}_{stamp}_{uniq}")


def timestamped(path, run_id):
    """Insert ``_<run_id>`` before the extension.

    ``outputs/checkpoints/lite2_phase2_best.pt`` ->
    ``outputs/checkpoints/lite2_phase2_best_23-09-2026-22-15.pt``
    """
    root, ext = os.path.splitext(path)
    return f"{root}_{run_id}{ext}"


def timestamped_dir(path, run_id):
    """Append ``_<run_id>`` to a directory path (no extension expected)."""
    return f"{path}_{run_id}"


def unique_path(path):
    """Return ``path`` if free, else ``path`` with ``_2``, ``_3`` ... appended.

    Guards explicit ``--run-tag`` reuse so results are never overwritten.
    """
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{root}_{i}{ext}"):
        i += 1
    return f"{root}_{i}{ext}"


def find_latest(directory, pattern):
    """Newest file matching ``pattern`` in ``directory`` (by mtime), or None."""
    if not os.path.isdir(directory):
        return None
    candidates = glob.glob(os.path.join(directory, pattern))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def find_latest_checkpoint(checkpoint_dir, backbone, phase="phase2"):
    """Find the newest ``<backbone>_...<phase>_best*.pt`` checkpoint.

    Matches both the timestamped names produced by current training
    (``lite2_phase2_best_V3.pt`` / ``..._23-09-2026-22-15.pt``) and the legacy
    untimestamped ``lite2_phase2_best.pt``.
    """
    return find_latest(checkpoint_dir, f"{backbone}_*{phase}_best*.pt")


def resolve_checkpoint(spec, backbone, checkpoint_dir, phase="phase2"):
    """Resolve a ``--checkpoint`` value (path, run-tag, or None) to a file.

    * an existing path           -> returned unchanged
    * a run-tag (e.g. ``V3``)    -> newest ``<backbone>_*<phase>_best*V3*.pt``
    * any filename fragment      -> newest ``*<spec>*.pt`` in the dir
    * ``None``                   -> newest ``<backbone>_*<phase>_best.pt``
    Returns None when nothing matches.
    """
    if spec:
        if os.path.exists(spec):
            return spec
        tag = sanitize_run_id(spec)
        for pattern in (f"{backbone}_*{phase}_best*{tag}*.pt",
                        f"*{tag}*.pt"):
            hit = find_latest(checkpoint_dir, pattern)
            if hit:
                return hit
        return None
    return find_latest_checkpoint(checkpoint_dir, backbone, phase)

# =======================================================================
# SECTION 2 — Execution logger
# RunLogger writes logs/<exec_id>/{manifest,config,run.log,events,*.jsonl}; init_run_logger() is idempotent; log_event/log_metrics are module-level.
# Used by: Every entry point; any module via log_event.
# Formerly src/run_logger.py. Disable with RUN_LOG_DISABLE=1 / LOG_TO_FILE=False.
# =======================================================================
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
