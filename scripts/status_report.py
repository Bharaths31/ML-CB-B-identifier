#!/usr/bin/env python3
"""Compact, dependency-free project status report.

Run directly with the venv Python:
    python scripts/status_report.py
    python scripts/status_report.py --json

Sections: git working tree, VERSION.txt, active/newest model, virtualenv and
requirement health. Safe on a partially configured machine; never raises for a
missing tool or file.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from setup_venv import normalize_name, read_requirements, satisfies, version_key
except Exception:  # setup_venv must never break the status report
    read_requirements = None
    normalize_name = lambda value: re.sub(r"[-_.]+", "-", str(value).lower())
    version_key = lambda value: tuple(int(x) for x in re.findall(r"\d+", str(value).split("+")[0])) or (0,)
    satisfies = lambda *_args: True


def run(command: list[str], timeout: int = 15) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def git_report() -> dict:
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = run(["git", "status", "--porcelain=v1"])
    commit = run(["git", "log", "-1", "--oneline"])
    upstream = run(["git", "rev-list", "--left-right", "--count", "@{upstream}...HEAD"])

    lines = [line for line in (status.stdout or "").splitlines() if line.strip()] if status else []
    counts = {"modified": 0, "staged": 0, "untracked": 0, "deleted": 0}
    for line in lines:
        code = line[:2]
        if code == "??":
            counts["untracked"] += 1
            continue
        if code[0] not in (" ", "?"):
            counts["staged"] += 1
        if code[1] not in (" ", "?"):
            counts["modified"] += 1
        if "D" in code:
            counts["deleted"] += 1
    ahead = behind = None
    if upstream and upstream.returncode == 0 and upstream.stdout.strip():
        parts = upstream.stdout.strip().split()
        if len(parts) == 2:
            behind, ahead = int(parts[0]), int(parts[1])
    return {
        "branch": (branch.stdout.strip() if branch and branch.returncode == 0 else "unknown"),
        "commit": (commit.stdout.strip() if commit and commit.returncode == 0 else "unknown"),
        "dirty": bool(lines),
        "counts": counts,
        "ahead": ahead,
        "behind": behind,
    }


def version_report() -> dict:
    path = os.path.join(ROOT, "VERSION.txt")
    if not os.path.exists(path):
        return {"present": False, "value": None, "newline": False}
    try:
        raw = open(path, encoding="utf-8").read()
        return {"present": True, "value": raw.strip(), "newline": raw.endswith("\n")}
    except OSError as exc:
        return {"present": True, "value": None, "error": str(exc), "newline": False}


def newest_file(pattern_root: str, suffix: str = "") -> str | None:
    if not os.path.isdir(pattern_root):
        return None
    candidates = []
    for name in os.listdir(pattern_root):
        path = os.path.join(pattern_root, name)
        if not os.path.exists(path):
            continue
        if suffix and not name.endswith(suffix):
            continue
        candidates.append(path)
    return max(candidates, key=os.path.getmtime) if candidates else None


def model_report() -> dict:
    config_path = os.path.join(ROOT, "outputs", "app_config.json")
    active = None
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as handle:
                active = json.load(handle).get("active_model")
        except (OSError, ValueError):
            active = None

    active_path = None
    if active:
        clean_active = active.replace(" (ONNX)", "")
        if active.startswith("portable/"):
            active_path = os.path.join(ROOT, "outputs", "export", "portable", active.split("/", 1)[1], "model.pt")
        else:
            for candidate in (
                os.path.join(ROOT, "outputs", "checkpoints", f"{clean_active}.pt"),
                os.path.join(ROOT, "outputs", "export", f"{clean_active}.onnx"),
                os.path.join(ROOT, "outputs", "checkpoints", f"{active}.pt"),
            ):
                if os.path.exists(candidate):
                    active_path = candidate
                    break

    newest_portable = newest_file(os.path.join(ROOT, "outputs", "export", "portable"))
    newest_checkpoint = newest_file(os.path.join(ROOT, "outputs", "checkpoints"), ".pt")
    return {
        "configured": active,
        "configured_exists": bool(active_path and os.path.exists(active_path)),
        "configured_path": active_path,
        "newest_portable": newest_portable,
        "newest_checkpoint": newest_checkpoint,
    }


def venv_report() -> dict:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    pip = run([sys.executable, "-m", "pip", "--version"], timeout=20)
    packages = {}
    notes = []
    try:
        for dist in importlib.metadata.distributions():
            name = dist.metadata.get("Name")
            if name:
                packages[normalize_name(name)] = dist.version
    except Exception as exc:
        notes.append(f"could not enumerate packages: {exc}")

    torch_version = packages.get("torch")
    torch_cuda = None
    if torch_version:
        try:
            import torch  # only when installed
            torch_cuda = bool(torch.cuda.is_available())
        except Exception as exc:
            notes.append(f"torch import failed: {exc}")

    missing = []
    outdated = []
    if read_requirements:
        try:
            for requirement in read_requirements():
                current = packages.get(requirement.normalized)
                if current is None:
                    missing.append(requirement.raw)
                elif requirement.operator and requirement.version and not satisfies(
                    current, requirement.operator, requirement.version
                ):
                    outdated.append(f"{requirement.name} {current} < {requirement.version}")
        except Exception as exc:
            notes.append(f"requirement check failed: {exc}")

    return {
        "active": in_venv,
        "path": sys.prefix,
        "python": platform_python(),
        "pip": pip.stdout.strip() if pip and pip.returncode == 0 else "missing",
        "torch": torch_version,
        "torch_cuda": torch_cuda,
        "missing": missing,
        "outdated": outdated,
        "notes": notes,
    }


def platform_python() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def path_label(path: str | None) -> str:
    if not path:
        return "—"
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def print_report(report: dict) -> None:
    git = report["git"]
    version = report["version"]
    model = report["model"]
    venv = report["venv"]

    print("=" * 72)
    print("  BreedID — Project Status")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 72)

    dirty = "dirty" if git["dirty"] else "clean"
    counts = git["counts"]
    detail = f"M:{counts['modified']} A:{counts['staged']} D:{counts['deleted']} ??:{counts['untracked']}"
    sync = ""
    if git["ahead"] is not None or git["behind"] is not None:
        sync = f" | ahead {git['ahead']}, behind {git['behind']}"
    print(f"Git        {git['branch']} @ {git['commit']}")
    print(f"           {dirty} ({detail}){sync}")

    version_text = version["value"] if version.get("present") else "missing"
    newline_note = "" if version.get("newline", True) else " (no trailing newline)"
    print(f"Version    VERSION.txt: {version_text}{newline_note}")

    configured = model["configured"] or "not configured (default resolution)"
    exists = "found" if model["configured_exists"] else "MISSING" if model["configured"] else "—"
    print(f"Model      active: {configured} [{exists}]")
    print(f"           newest checkpoint: {path_label(model['newest_checkpoint'])}")
    print(f"           newest portable:   {path_label(model['newest_portable'])}")

    active = "active" if venv["active"] else "NOT ACTIVE (system Python)"
    print(f"Venv       {active}: {venv['path']}")
    print(f"           Python {venv['python']} | {venv['pip']}")
    cuda = "unknown" if venv["torch_cuda"] is None else str(venv["torch_cuda"])
    print(f"           torch: {venv['torch'] or 'missing'} (CUDA={cuda})")

    if venv["missing"]:
        print(f"Packages   missing: {', '.join(venv['missing'][:10])}"
              + (" …" if len(venv["missing"]) > 10 else ""))
    if venv["outdated"]:
        print(f"           outdated: {', '.join(venv['outdated'][:10])}"
              + (" …" if len(venv["outdated"]) > 10 else ""))
    if not venv["missing"] and not venv["outdated"]:
        print("Packages   requirements satisfied")
    for note in venv["notes"]:
        print(f"           note: {note}")

    if not version.get("present"):
        print("Warnings   VERSION.txt is missing")
    if model["configured"] and not model["configured_exists"]:
        print("Warnings   configured model file is missing")
    if not venv["active"]:
        print("Warnings   run with the venv Python (or activate .venv first)")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Project git/version/model/venv status")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": git_report(),
        "version": version_report(),
        "model": model_report(),
        "venv": venv_report(),
    }
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
