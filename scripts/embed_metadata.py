#!/usr/bin/env python
"""Make existing ``.pt`` / ``.onnx`` models self-describing.

The GUI and the Flask API no longer need ``cattle_classes.json``,
``buffalo_classes.json``, ``model_info.json`` or ``outputs/ood/*.json`` beside a
model: this tool writes class maps, preprocessing parameters, input resolution
and OOD thresholds **inside the model file**.

Usage
-----
    # Embed metadata into every model under outputs/
    python scripts/embed_metadata.py --all

    # One file (format detected from the extension)
    python scripts/embed_metadata.py --file outputs/checkpoints/lite2_phase2_best_v26.pt

    # Show what a model contains without loading it for inference
    python scripts/embed_metadata.py --inspect outputs/export/lite2_v27_1020-1201_0547_fp32.onnx

    # Scan another directory
    python scripts/embed_metadata.py --all --outputs-dir path/to/models

    # Override the auto-detected contract
    python scripts/embed_metadata.py --file model.onnx \
        --cattle-classes data/splits/cattle_classes.json \
        --buffalo-classes data/splits/buffalo_classes.json \
        --ood-json outputs/ood/my_model.json --no-pad-to-square

The operation is idempotent: re-running replaces the metadata block and keeps
the weights/optimizer state untouched.  Model files are written atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import OUTPUT_DIR  # noqa: E402
from src.model_introspect import (  # noqa: E402
    build_model_metadata,
    describe_model_file,
    embed_model_metadata,
    find_model_files,
    find_ood_json,
    inspect_and_load_onnx,
    read_onnx_metadata,
    read_pt_metadata,
)


def _load_json(path: Optional[str]) -> Any:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _ood_overrides(args) -> Optional[Dict[str, Any]]:
    if args.no_ood:
        return {"enabled": False}
    data = _load_json(args.ood_json) if args.ood_json else None
    if not isinstance(data, dict):
        return None
    return {
        "enabled": True,
        "energy_temperature": data.get("energy_temperature", data.get("temperature")),
        "temperature": data.get("temperature", data.get("energy_temperature")),
        "energy_threshold": data.get("energy_threshold"),
        "msp_threshold": data.get("msp_threshold"),
        "binary_confidence_floor": data.get("binary_confidence_floor"),
        "calibrated_at": data.get("calibrated_at"),
        "percentile": data.get("percentile"),
        "calibrated_on_samples": data.get("calibrated_on_samples"),
        "source_file": args.ood_json,
    }


def _embed_kwargs(args) -> Dict[str, Any]:
    pad = None
    if args.pad_to_square:
        pad = True
    elif args.no_pad_to_square:
        pad = False
    normalized = None
    if args.baked_normalization:
        normalized = True
    elif args.caller_normalization:
        normalized = False
    return {
        "backbone": args.backbone,
        "attention": args.attention,
        "image_size": args.image_size,
        "eval_resize": args.eval_resize,
        "pad_to_square": pad,
        "normalization_baked": normalized,
        "cattle_classes": _load_json(args.cattle_classes),
        "buffalo_classes": _load_json(args.buffalo_classes),
        "split_dir": args.split_dir,
        "ood": _ood_overrides(args),
    }


def _author_metadata(path: str, args) -> Dict[str, Any]:
    """Build metadata, preferring the model's own OOD JSON when present."""
    kwargs = _embed_kwargs(args)
    kwargs["search_dirs"] = [os.path.dirname(os.path.abspath(path))]
    ext = os.path.splitext(path)[1].lower()
    if ext == ".onnx":
        # Read the graph to detect baked mean/std and the true input size.
        try:
            bundle = inspect_and_load_onnx(path, use_cuda=False)
            if kwargs.get("normalization_baked") is None:
                kwargs["normalization_baked"] = bundle.normalization_baked
            if kwargs.get("image_size") is None:
                kwargs["image_size"] = bundle.image_size
            if kwargs.get("eval_resize") is None:
                kwargs["eval_resize"] = bundle.image_size
            kwargs["input_dtype"] = bundle.input_dtype
            kwargs["spec"] = {
                "num_cattle": bundle.num_cattle,
                "num_buffalo": bundle.num_buffalo,
                "backbone": bundle.spec.get("backbone"),
                "attention": bundle.spec.get("attention"),
            }
        except Exception:
            pass
    else:
        # Derive architecture + class counts from the checkpoint tensors.
        try:
            import torch
            from src.model_introspect import extract_state_dict
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, torch.nn.Module):
                first = next(ckpt.parameters(), None)
                if first is not None and first.dtype == torch.float16:
                    kwargs["input_dtype"] = "float16"
            state = extract_state_dict(ckpt)
            if state:
                kwargs["state_dict"] = state
            elif hasattr(ckpt, "state_dict"):
                kwargs["state_dict"] = ckpt.state_dict()
        except Exception:
            pass
        if kwargs.get("normalization_baked") is None:
            # Checkpoints use caller-side normalization unless forced otherwise.
            kwargs["normalization_baked"] = False
    if kwargs.get("ood") is None:
        ood_path = find_ood_json(path)
        if ood_path:
            data = _load_json(ood_path) or {}
            kwargs["ood"] = {
                "enabled": True,
                "energy_temperature": data.get("energy_temperature", data.get("temperature")),
                "temperature": data.get("temperature", data.get("energy_temperature")),
                "energy_threshold": data.get("energy_threshold"),
                "msp_threshold": data.get("msp_threshold"),
                "binary_confidence_floor": data.get("binary_confidence_floor"),
                "calibrated_at": data.get("calibrated_at"),
                "percentile": data.get("percentile"),
                "calibrated_on_samples": data.get("calibrated_on_samples"),
                "source_file": ood_path,
            }
    return build_model_metadata(source_checkpoint=path, **kwargs)


def _embed_one(path: str, args) -> Dict[str, Any]:
    meta = _author_metadata(path, args)
    return embed_model_metadata(path, meta=meta, overwrite=True)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=True, default=str))


def _inspect(path: str, args) -> int:
    ext = os.path.splitext(path)[1].lower()
    metadata = read_onnx_metadata(path) if ext == ".onnx" else read_pt_metadata(path)
    print("=" * 78)
    print(f"Model: {path}")
    print("=" * 78)
    print("\nEmbedded metadata:")
    if metadata:
        _print_json(metadata)
    else:
        print("  (none - run this script without --inspect to embed it)")
    print("\nRuntime introspection (derived from tensors / graph):")
    try:
        _print_json(describe_model_file(path))
    except Exception as exc:
        print(f"  introspection failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


def _scan(args) -> int:
    root = args.outputs_dir or OUTPUT_DIR
    if not os.path.isdir(root):
        print(f"[embed] directory not found: {root}")
        return 1
    files = find_model_files(root)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"[embed] no .pt/.pth/.onnx files under {root}")
        return 0

    print(f"[embed] {len(files)} model file(s) under {root}")
    ok = failed = 0
    for index, path in enumerate(files, 1):
        rel = os.path.relpath(path, root)
        try:
            meta = _embed_one(path, args)
            ok += 1
            print(f"  [{index:>3}/{len(files)}] OK   {rel}  "
                  f"(cattle={meta.get('num_cattle')}, buffalo={meta.get('num_buffalo')}, "
                  f"size={meta.get('image_size')}, baked={meta.get('normalization_baked')})")
        except Exception as exc:
            failed += 1
            print(f"  [{index:>3}/{len(files)}] FAIL {rel}: {type(exc).__name__}: {exc}")
            if args.debug:
                traceback.print_exc()
    print(f"\n[embed] done: {ok} embedded, {failed} failed")
    return 0 if failed == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed class maps, preprocessing and OOD thresholds into "
                    ".pt/.onnx model files (zero external JSON at runtime).")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true",
                        help="scan --outputs-dir and embed into every model")
    target.add_argument("--file", action="append", metavar="PATH",
                        help="embed into one file (repeatable)")
    target.add_argument("--inspect", metavar="PATH",
                        help="print embedded metadata + introspected spec")

    parser.add_argument("--outputs-dir", default=None,
                        help=f"root for --all (default: {OUTPUT_DIR})")
    parser.add_argument("--split-dir", default=None,
                        help="directory holding cattle_classes.json/buffalo_classes.json")
    parser.add_argument("--cattle-classes", default=None,
                        help="JSON file mapping cattle breed -> index")
    parser.add_argument("--buffalo-classes", default=None,
                        help="JSON file mapping buffalo breed -> index")
    parser.add_argument("--ood-json", default=None,
                        help="OOD threshold JSON to embed (default: auto-discover)")
    parser.add_argument("--no-ood", action="store_true",
                        help="embed OOD as disabled (config defaults kept)")
    parser.add_argument("--backbone", choices=["lite2", "lite4"], default=None)
    parser.add_argument("--attention", choices=["cbam", "se"], default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--eval-resize", type=int, default=None)
    parser.add_argument("--pad-to-square", action="store_true", default=None,
                        help="force PadToSquare preprocessing")
    parser.add_argument("--no-pad-to-square", action="store_true", default=None,
                        help="force Resize+CenterCrop preprocessing")
    parser.add_argument("--baked-normalization", action="store_true", default=False,
                        help="ONNX: mean/std are inside the graph (feed [0,1])")
    parser.add_argument("--caller-normalization", action="store_true", default=False,
                        help="ONNX: caller must apply ImageNet Normalize")
    parser.add_argument("--limit", type=int, default=None,
                        help="max files processed by --all (debugging)")
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    # argparse can leave both flags True with default=None; normalise.
    if args.pad_to_square and args.no_pad_to_square:
        args.no_pad_to_square = False

    if args.inspect:
        return _inspect(args.inspect, args)

    if args.file:
        ok = failed = 0
        for path in args.file:
            if not os.path.exists(path):
                print(f"[embed] missing: {path}")
                failed += 1
                continue
            try:
                meta = _embed_one(path, args)
                ok += 1
                print(f"[embed] OK {path}")
                print(f"        cattle={meta.get('num_cattle')} "
                      f"buffalo={meta.get('num_buffalo')} "
                      f"image_size={meta.get('image_size')} "
                      f"baked={meta.get('normalization_baked')} "
                      f"ood_energy={meta.get('ood', {}).get('energy_threshold')}")
            except Exception as exc:
                failed += 1
                print(f"[embed] FAIL {path}: {type(exc).__name__}: {exc}")
                if args.debug:
                    traceback.print_exc()
        return 0 if failed == 0 else 2

    return _scan(args)


if __name__ == "__main__":
    raise SystemExit(main())
