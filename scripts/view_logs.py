#!/usr/bin/env python
"""Inspect execution logs written by src/run_logger.py. Read-only.

    python scripts/view_logs.py list
    python scripts/view_logs.py summary [<exec_id>|latest]
    python scripts/view_logs.py events  [<exec_id>|latest] [--category training]
    python scripts/view_logs.py metrics [<exec_id>|latest] [--tag raw|ema]
    python scripts/view_logs.py diff <exec_id_a> <exec_id_b>
"""

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.config import LOG_ROOT  # noqa: E402


def all_execs():
    if not os.path.isdir(LOG_ROOT):
        return []
    return sorted((d for d in os.listdir(LOG_ROOT)
                   if os.path.isdir(os.path.join(LOG_ROOT, d))), reverse=True)


def resolve(exec_id):
    if exec_id in (None, "latest"):
        ex = all_execs()
        return ex[0] if ex else None
    return exec_id


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def cmd_list(_args):
    ex = all_execs()
    if not ex:
        print(f"(no executions under {LOG_ROOT})")
        return
    print(f"{'exec_id':<26} {'module':<20} {'started':<20} {'dur':>8} {'exit':>5}")
    for e in ex:
        m = load_json(os.path.join(LOG_ROOT, e, "manifest.json"), {}) or {}
        print(f"{e:<26} {str(m.get('module',''))[:20]:<20} "
              f"{str(m.get('started_at','')):<20} "
              f"{str(m.get('duration_s','')):>8} {str(m.get('exit_code','')):>5}")


def cmd_summary(args):
    e = resolve(args.exec_id)
    if not e:
        print("no executions"); return
    d = os.path.join(LOG_ROOT, e)
    m = load_json(os.path.join(d, "manifest.json"), {})
    print(f"exec_id : {e}\nmodule  : {m.get('module')}\nargv    : {' '.join(m.get('argv', []))}")
    print(f"cwd     : {m.get('cwd')}\ngit     : {m.get('git_commit')}")
    print(f"started : {m.get('started_at')}  ended: {m.get('ended_at')}  "
          f"dur: {m.get('duration_s')}s  exit: {m.get('exit_code')}")
    train = load_jsonl(os.path.join(d, "training.jsonl"))
    if train:
        final = {}
        for r in train:
            final[(r.get("phase"), r.get("tag"))] = r
        print("\nfinal training metrics:")
        for (phase, tag), r in sorted(final.items(), key=lambda kv: str(kv[0])):
            print(f"  phase{phase} {tag:>3}: blend={r.get('blended_score',0):.4f} "
                  f"top1s={r.get('combined_top1_soft',0):.4f} "
                  f"mF1={0.5*(r.get('cattle_macro_f1',0)+r.get('buffalo_macro_f1',0)):.4f} "
                  f"few/med/many={r.get('acc_fewshot',0):.2f}/{r.get('acc_mediumshot',0):.2f}/"
                  f"{r.get('acc_manyshot',0):.2f} ent={r.get('pred_hist_entropy',0):.3f}")
    errs = [r for r in load_jsonl(os.path.join(d, "events.jsonl"))
            if r.get("event") in ("uncaught_exception", "command_failed", "artifact_failed")]
    if errs:
        print("\nerrors:")
        for r in errs:
            print(f"  {r.get('event')}: {r.get('message') or r.get('error') or r.get('cmd')}")


def cmd_events(args):
    e = resolve(args.exec_id)
    if not e:
        print("no executions"); return
    path = os.path.join(LOG_ROOT, e, f"{args.category}.jsonl" if args.category
                        else "events.jsonl")
    for r in load_jsonl(path):
        print(json.dumps(r, ensure_ascii=False))


def cmd_metrics(args):
    e = resolve(args.exec_id)
    if not e:
        print("no executions"); return
    for r in load_jsonl(os.path.join(LOG_ROOT, e, "training.jsonl")):
        if args.tag and r.get("tag") != args.tag:
            continue
        print(f"phase{r.get('phase')} e{r.get('epoch'):>3} {r.get('tag',''):>3} "
              f"blend={r.get('blended_score',0):.4f} top1s={r.get('combined_top1_soft',0):.4f} "
              f"bin={r.get('binary_acc',0):.3f} ent={r.get('pred_hist_entropy',0):.3f}")


def _final(train, tag="raw"):
    best = None
    for r in train:
        if r.get("tag") == tag:
            best = r
    return best or {}


def cmd_diff(args):
    a, b = resolve(args.exec_a), resolve(args.exec_b)
    if not a or not b:
        print("need two exec ids"); return
    ra = _final(load_jsonl(os.path.join(LOG_ROOT, a, "training.jsonl")))
    rb = _final(load_jsonl(os.path.join(LOG_ROOT, b, "training.jsonl")))
    keys = ["blended_score", "combined_top1_soft", "binary_acc",
            "cattle_acc", "buffalo_acc", "acc_fewshot", "acc_mediumshot",
            "acc_manyshot", "pred_hist_entropy", "val_min_per_breed"]
    print(f"{'metric':<22} {a[:20]:>22} {b[:20]:>22} {'delta':>10}")
    for k in keys:
        va, vb = ra.get(k, 0.0), rb.get(k, 0.0)
        print(f"{k:<22} {va:>22.4f} {vb:>22.4f} {vb - va:>+10.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("summary"); s.add_argument("exec_id", nargs="?")
    s = sub.add_parser("events"); s.add_argument("exec_id", nargs="?")
    s.add_argument("--category", default=None)
    s = sub.add_parser("metrics"); s.add_argument("exec_id", nargs="?")
    s.add_argument("--tag", default=None)
    s = sub.add_parser("diff"); s.add_argument("exec_a"); s.add_argument("exec_b")
    args = ap.parse_args()
    {"list": cmd_list, "summary": cmd_summary, "events": cmd_events,
     "metrics": cmd_metrics, "diff": cmd_diff}[args.cmd](args)


if __name__ == "__main__":
    main()
