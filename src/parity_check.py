"""Parity gate: compare fp32 PyTorch vs mobile artifacts (TFLite INT8 /
ONNX Runtime INT8) on the same data.

Two modes:

1. Accuracy mode (default, needs data/splits):
   python -m src.parity_check --checkpoint outputs/checkpoints/lite2_phase2_best.pt \
       --tflite outputs/export/lite2_int8.tflite --onnx-int8 outputs/export/lite2_mobile_int8.onnx \
       --onnx outputs/export/lite2_fp32.onnx --split val

   Reports the same combined metrics as training validation for every runtime
   plus deltas vs the fp32 reference. ACCEPTANCE: any INT8 artifact must stay
   within 1.0 pt combined_top1 of fp32, otherwise fall back to FP16 TFLite or
   revisit quantization.

2. Synthetic mode (no dataset needed — verifies artifact wiring):
   python -m src.parity_check --checkpoint <pt> --synthetic 8

   Runs random [0,1] tensors through every runtime and reports max |logit|
   differences vs the fp32 PyTorch reference under each artifact's input
   convention (fp32 ONNX expects caller-normalized input; mobile artifacts
   have normalization baked in).
"""

import argparse
import json
import os

import numpy as np
import torch

from .config import (CHECKPOINT_DIR, IMAGENET_MEAN, IMAGENET_STD,
                     METRICS_DIR, SPLIT_DIR)
from .data_pipeline import get_dataloaders
from .export import _load_model  # reuse the guarded checkpoint loader
from .run_utils import make_run_id, timestamped

MOBILE_CONVENTION = "input RGB in [0,1], normalization baked into the artifact"
RAW_CONVENTION = "input ImageNet-normalized by the caller"


# ---------------------------------------------------------------------------
# Runtime runners — each takes a batch of [0,1] float tensors (B,3,H,W)
# and returns (binary, cattle, buffalo) CPU logits.
# ---------------------------------------------------------------------------

def make_torch_runner(model, device):
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def run(x01):
        x = (x01.to(device) - mean) / std
        out = model(x)
        return (out["binary"].cpu(), out["cattle"].cpu(), out["buffalo"].cpu())

    return run, RAW_CONVENTION


def make_onnx_runner(path, mobile=False):
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)

    def run(x01):
        arr = x01.numpy().astype(np.float32)
        if not mobile:
            arr = (arr - mean) / std
        outs = sess.run(None, {input_name: arr})
        return (torch.from_numpy(outs[0]), torch.from_numpy(outs[1]),
                torch.from_numpy(outs[2]))

    return run, MOBILE_CONVENTION if mobile else RAW_CONVENTION


def make_tflite_runner(path):
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=path)
    except ImportError:
        from tflite_runtime.interpreter import Interpreter
        interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out_details = interp.get_output_details()

    def run(x01):
        arr = x01.numpy().astype(inp["dtype"])
        interp.set_tensor(inp["index"], arr)
        interp.invoke()
        # Prefer explicit name matching (onnx2tf copies ONNX output names
        # into the TFLite signature); fall back to declaration order
        # (binary, cattle, buffalo) when names are absent.
        if all(any(w in (d.get("name") or "").lower() for d in out_details)
               for w in ("binary", "cattle", "buffalo")):
            ordered = [next(d for d in out_details
                            if w in (d.get("name") or "").lower())
                       for w in ("binary", "cattle", "buffalo")]
        else:
            ordered = out_details
        outs = [interp.get_tensor(d["index"]) for d in ordered]
        return (torch.from_numpy(outs[0]), torch.from_numpy(outs[1]),
                torch.from_numpy(outs[2]))

    return run, MOBILE_CONVENTION


# ---------------------------------------------------------------------------
# Metrics (same definitions as src/metrics.evaluate_epoch)
# ---------------------------------------------------------------------------

def accumulate(metrics, logits, labels):
    binary, cattle, buffalo = logits
    bin_pred = binary.argmax(1)
    bin_true = labels["binary"].argmax(1)
    cattle_pred = cattle.argmax(1)
    cattle_true = labels["cattle"].argmax(1)
    buffalo_pred = buffalo.argmax(1)
    buffalo_true = labels["buffalo"].argmax(1)
    cmask = labels["cattle_mask"] > 0.5
    bmask = labels["buffalo_mask"] > 0.5
    bs = binary.size(0)

    m = metrics
    m["n"] += bs
    m["binary_correct"] += (bin_pred == bin_true).sum().item()
    m["n_cattle"] += cmask.sum().item()
    m["cattle_correct"] += (cattle_pred[cmask] == cattle_true[cmask]).sum().item()
    m["n_buffalo"] += bmask.sum().item()
    m["buffalo_correct"] += (buffalo_pred[bmask] == buffalo_true[bmask]).sum().item()

    cattle_hit = (bin_pred == 0) & (bin_true == 0) & (cattle_pred == cattle_true)
    buffalo_hit = (bin_pred == 1) & (bin_true == 1) & (buffalo_pred == buffalo_true)
    m["top1_correct"] += (cattle_hit | buffalo_hit).sum().item()

    cattle_top3 = cattle.topk(3, dim=1).indices
    buffalo_top3 = buffalo.topk(3, dim=1).indices
    cattle_in3 = (cattle_top3 == cattle_true.unsqueeze(1)).any(dim=1)
    buffalo_in3 = (buffalo_top3 == buffalo_true.unsqueeze(1)).any(dim=1)
    m["top3_correct"] += torch.where(bin_true == 0, cattle_in3,
                                     buffalo_in3).sum().item()


def finalize(metrics):
    n = max(1, metrics["n"])

    def acc(c, d):
        return c / d if d else 0.0

    return {
        "binary_acc": acc(metrics["binary_correct"], n),
        "cattle_acc": acc(metrics["cattle_correct"], metrics["n_cattle"]),
        "buffalo_acc": acc(metrics["buffalo_correct"], metrics["n_buffalo"]),
        "combined_top1": acc(metrics["top1_correct"], n),
        "combined_top3": acc(metrics["top3_correct"], n),
        "images": metrics["n"],
    }


def new_metrics():
    return {"n": 0, "binary_correct": 0, "n_cattle": 0, "cattle_correct": 0,
            "n_buffalo": 0, "buffalo_correct": 0, "top1_correct": 0,
            "top3_correct": 0}


# ---------------------------------------------------------------------------
# Evaluation drivers
# ---------------------------------------------------------------------------

def evaluate_on_split(name, runner, loader, device, limit):
    metrics = new_metrics()
    seen = 0
    for images, labels in loader:
        if limit and seen >= limit:
            break
        if limit:
            take = min(images.size(0), limit - seen)
            images = images[:take]
            labels = {k: v[:take] for k, v in labels.items()}
        seen += images.size(0)
        logits = runner(images)
        accumulate(metrics, logits, labels)
    return finalize(metrics)


def synthetic_parity(runnings, n, batch_size=1):
    """Max |logit diff| vs the torch reference on random [0,1] inputs.

    Batch size 1 by default: mobile artifacts are exported with a static
    batch of 1.
    """
    ref_run = runnings[0][1]
    report = {}
    gen = torch.Generator().manual_seed(1234)
    for name, run, convention in runnings[1:]:
        diffs = [0.0, 0.0, 0.0]
        done = 0
        while done < n:
            b = min(batch_size, n - done)
            x = torch.rand(b, 3, 260, 260, generator=gen)
            r_bin, r_cat, r_buf = ref_run(x)
            b_bin, b_cat, b_buf = run(x)
            for i, (r, t) in enumerate(((r_bin, b_bin), (r_cat, b_cat),
                                        (r_buf, b_buf))):
                diffs[i] = max(diffs[i],
                               (r - t).abs().max().item())
            done += b
        report[name] = {
            "convention": convention,
            "max_abs_diff": {"binary": round(diffs[0], 5),
                             "cattle": round(diffs[1], 5),
                             "buffalo": round(diffs[2], 5)},
        }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parity gate: fp32 PyTorch vs TFLite INT8 vs ONNX INT8")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--backbone", choices=["lite2", "lite4"], default="lite2")
    parser.add_argument("--attention", choices=["cbam", "se"], default="cbam")
    parser.add_argument("--onnx", default=None,
                        help=f"fp32 ONNX ({RAW_CONVENTION})")
    parser.add_argument("--onnx-int8", default=None,
                        help=f"quantized ONNX for ORT Mobile ({MOBILE_CONVENTION})")
    parser.add_argument("--tflite", default=None,
                        help=f"TFLite fp32 or INT8 ({MOBILE_CONVENTION})")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--split-dir", default=SPLIT_DIR)
    parser.add_argument("--limit", type=int, default=0,
                        help="max images per runtime (0 = all)")
    parser.add_argument("--synthetic", type=int, default=0,
                        help="skip the dataset; compare N random inputs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="kept for compatibility; evaluation always feeds "
                             "one image at a time because mobile artifacts are "
                             "exported with a static batch of 1")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="allowed combined_top1 drop vs fp32 (default 1pt)")
    parser.add_argument("--out", default=None, help="JSON report path")
    parser.add_argument("--run-tag", default=None,
                        help="run id for timestamped report filenames")
    args = parser.parse_args()

    run_id = make_run_id(args.run_tag)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        from .run_utils import find_latest_checkpoint
        checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR, args.backbone)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"[parity] checkpoint not found under {CHECKPOINT_DIR} "
              f"(looked for {args.backbone}_*_phase2_best.pt); pass --checkpoint")
        return 1

    model = _load_model(checkpoint_path, args.backbone, args.attention)
    model.to(device)
    print(f"[parity] fp32 reference: {checkpoint_path} ({args.backbone}+{args.attention})")

    runnings = [("pytorch_fp32", *make_torch_runner(model, device))]
    for name, path, mobile in (("onnx_fp32", args.onnx, False),
                               ("onnx_int8", args.onnx_int8, True),
                               ("tflite", args.tflite, False)):
        if not path:
            continue
        if not os.path.exists(path):
            print(f"[parity] WARNING: {name} not found at {path} — skipped")
            continue
        if name == "tflite":
            try:
                runnings.append((name, *make_tflite_runner(path)))
            except ImportError:
                print("[parity] WARNING: tensorflow / tflite_runtime not "
                      "installed — TFLite skipped")
                continue
        else:
            runnings.append((name, *make_onnx_runner(path, mobile=mobile)))

    if args.synthetic:
        if len(runnings) < 2:
            print("[parity] no artifact runtimes given — nothing to compare")
            return 1
        print(f"[parity] synthetic mode: {args.synthetic} random inputs")
        detail = synthetic_parity(runnings, args.synthetic)
        print()
        for name, info in detail.items():
            diffs = info["max_abs_diff"]
            print(f"  {name:14s} max|Δlogits| binary={diffs['binary']:.4f} "
                  f"cattle={diffs['cattle']:.4f} buffalo={diffs['buffalo']:.4f} "
                  f"({info['convention']})")
        worst = max(v["max_abs_diff"][h]
                    for v in detail.values() for h in v["max_abs_diff"])
        verdict = "PASS" if worst < 0.05 else "REVIEW"
        print(f"\n[parity] verdict: {verdict} "
              f"(worst max|Δlogit|={worst:.4f}, guidance < 0.05)")
        report = {"mode": "synthetic", "verdict": verdict, "detail": detail,
                  "run_id": run_id}
        out = args.out or timestamped(
            os.path.join(METRICS_DIR, f"{args.backbone}_parity"), run_id) + ".json"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[parity] report -> {out}")
        return 0 if verdict == "PASS" else 2

    loaders = get_dataloaders(split_dir=args.split_dir or SPLIT_DIR,
                              batch_size=1,  # static-batch-1 mobile artifacts
                              num_workers=args.num_workers)
    if loaders is None:
        print("[parity] no splits found (run prepare_splits) — or use "
              "--synthetic for artifact-only parity")
        return 1
    loader = loaders[1] if args.split == "val" else loaders[2]

    results = {}
    for name, run, convention in runnings:
        print(f"[parity] evaluating {name} ({convention}) on {args.split}...")
        results[name] = evaluate_on_split(name, run, loader, device, args.limit)

    keys = ["binary_acc", "cattle_acc", "buffalo_acc",
            "combined_top1", "combined_top3"]
    header = f"  {'runtime':14s}" + "".join(f"{k:>15s}" for k in keys)
    print("\n" + header)
    for name, m in results.items():
        print(f"  {name:14s}" + "".join(f"{m[k]:15.4f}" for k in keys))

    ref = results["pytorch_fp32"]
    print("\n  deltas vs pytorch_fp32 (combined_top1):")
    verdict = "PASS"
    for name, m in results.items():
        if name == "pytorch_fp32":
            continue
        delta = m["combined_top1"] - ref["combined_top1"]
        print(f"  {name:14s} {delta:+.4f}")
        if name.endswith("int8") or name == "tflite":
            if delta < -args.tolerance:
                verdict = "FAIL"

    print(f"\n[parity] verdict: {verdict} (tolerance {args.tolerance:.2f} on "
          f"combined_top1)")
    report = {"mode": args.split, "verdict": verdict, "results": results,
              "tolerance": args.tolerance, "run_id": run_id}
    out = args.out or timestamped(
        os.path.join(METRICS_DIR, f"{args.backbone}_parity_{args.split}"),
        run_id) + ".json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[parity] report -> {out}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
