"""Run-scoped output helpers.

Every training/export/evaluation run writes to a timestamped name so previous
results are never overwritten. The run id defaults to the config
``RUN_ID_FORMAT`` (DD-MM-YYYY-HH-MM) and can be overridden with ``--run-tag``.
"""

import glob
import os
import re
import time

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
    """Return a sanitized explicit run tag, or a fresh timestamp.

    Uses config ``RUN_ID_FORMAT`` (DD-MM-YYYY-HH-MM). When
    ``TIMESTAMP_OUTPUTS`` is False the literal ``"latest"`` is returned, which
    makes outputs overwrite (opt-out only).
    """
    if run_tag:
        return sanitize_run_id(run_tag)
    if not TIMESTAMP_OUTPUTS:
        return "latest"
    return sanitize_run_id(time.strftime(fmt or RUN_ID_FORMAT))


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
