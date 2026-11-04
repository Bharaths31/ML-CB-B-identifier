#!/usr/bin/env python
"""Create OOD-embedded / non-embedded mutant copies of a PT and ONNX model.

Produces ``outputs/mutated_models/{pt,onnx}/`` with two variants per format:

* ``*__ood.*``     — the model's calibrated OOD thresholds are embedded.
* ``*__no_ood.*``  — the same weights with **no** ``ood`` metadata key.

Weights/graphs are never modified.  A ``manifest.json`` records hashes and a
verification section, and ``README.md`` explains how to compare the variants.

Usage
-----
    python scripts/make_ood_variants.py
    python scripts/make_ood_variants.py \
        --pt outputs/checkpoints/my_model.pt \
        --onnx outputs/export/my_model.onnx \
        --ood-json outputs/ood/my_model.json \
        --out-dir outputs/mutated_models
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import shutil
import sys
from typing import Any, Dict, Optional

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_introspect import (  # noqa: E402
    build_model_metadata,
    embed_model_metadata,
    extract_state_dict,
    inspect_and_load_onnx,
    inspect_and_load_pt,
    read_onnx_metadata,
    read_pt_metadata,
)

DEFAULT_PT = os.path.join(
    PROJECT_ROOT, "outputs", "checkpoints",
    "lite2_phase2_best_v26_1012-2332_f9ce.pt")
DEFAULT_ONNX = os.path.join(
    PROJECT_ROOT, "outputs", "export",
    "lite2_v27_1020-1201_0547_fp32.onnx")
DEFAULT_OOD = os.path.join(
    PROJECT_ROOT, "outputs", "ood",
    "lite2_phase2_best_v26_1012-2332_f9ce.json")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "outputs", "mutated_models")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state(state: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_ood(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "enabled": True,
        "energy_temperature": data.get("energy_temperature", data.get("temperature", 1.0)),
        "temperature": data.get("temperature", data.get("energy_temperature", 1.0)),
        "energy_threshold": data.get("energy_threshold"),
        "msp_threshold": data.get("msp_threshold"),
        "binary_confidence_floor": data.get("binary_confidence_floor"),
        "calibrated_at": data.get("calibrated_at"),
        "percentile": data.get("percentile"),
        "calibrated_on_samples": data.get("calibrated_on_samples"),
        "source_file": os.path.basename(path),
    }


def _variant_meta(base: Dict[str, Any], variant: str) -> Dict[str, Any]:
    meta = copy.deepcopy(base)
    meta["variant"] = variant
    if variant == "no_ood":
        meta.pop("ood", None)
        meta["ood_embedded"] = False
        meta["ood_note"] = "No OOD thresholds embedded; runtime config defaults apply."
    else:
        meta["ood_embedded"] = True
    return meta


def make_pt(source: str, out_dir: str, ood: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    stem = os.path.splitext(os.path.basename(source))[0]
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    state = extract_state_dict(ckpt)
    if not state:
        raise RuntimeError(f"{source} is not a state_dict checkpoint")
    source_state_hash = sha256_state(state)

    base = build_model_metadata(
        state_dict=state,
        ood=ood,
        source_checkpoint=source,
        extra={"source_model": os.path.basename(source)},
    )

    results = {}
    for variant, suffix in (("with_ood", "__ood"), ("no_ood", "__no_ood")):
        dest = os.path.join(out_dir, "pt", f"{stem}{suffix}.pt")
        shutil.copy2(source, dest)
        meta = _variant_meta(base, variant)
        embed_model_metadata(dest, meta=meta, overwrite=True)
        out_state = extract_state_dict(
            torch.load(dest, map_location="cpu", weights_only=False))
        results[variant] = {
            "file": os.path.relpath(dest, out_dir),
            "path": dest,
            "file_sha256": sha256_file(dest),
            "state_sha256": sha256_state(out_state),
            "weights_unchanged": sha256_state(out_state) == source_state_hash,
            "ood_embedded": variant == "with_ood",
            "num_cattle": meta.get("num_cattle"),
            "num_buffalo": meta.get("num_buffalo"),
            "image_size": meta.get("image_size"),
            "normalization_baked": meta.get("normalization_baked"),
        }
    return {
        "source": source,
        "source_file_sha256": sha256_file(source),
        "source_state_sha256": source_state_hash,
        "variants": results,
    }


def _onnx_diff(reference_path: str, candidate_path: str) -> float:
    ref = inspect_and_load_onnx(reference_path, use_cuda=False)
    cand = inspect_and_load_onnx(candidate_path, use_cuda=False)
    x = torch.randn(1, 3, ref.image_size, ref.image_size)
    arr = x.numpy()
    if "float16" in str(ref.input_dtype).lower():
        arr = arr.astype("float16")
    ref_out = ref.model_or_session.run(None, {ref.onnx_input_name: arr})
    cand_out = cand.model_or_session.run(None, {cand.onnx_input_name: arr})
    import numpy as np
    return float(max(np.max(np.abs(a - b)) for a, b in zip(ref_out, cand_out)))


def make_onnx(source: str, out_dir: str, ood: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    stem = os.path.splitext(os.path.basename(source))[0]
    bundle = inspect_and_load_onnx(source, use_cuda=False)

    base = build_model_metadata(
        spec=bundle.spec,
        image_size=bundle.image_size,
        normalization_baked=bundle.normalization_baked,
        input_dtype=bundle.input_dtype,
        cattle_classes=bundle.cattle_map,
        buffalo_classes=bundle.buffalo_map,
        ood=ood,
        source_checkpoint=source,
        extra={"source_model": os.path.basename(source)},
    )

    results = {}
    for variant, suffix in (("with_ood", "__ood"), ("no_ood", "__no_ood")):
        dest = os.path.join(out_dir, "onnx", f"{stem}{suffix}.onnx")
        shutil.copy2(source, dest)
        meta = _variant_meta(base, variant)
        embed_model_metadata(dest, meta=meta, overwrite=True)
        results[variant] = {
            "file": os.path.relpath(dest, out_dir),
            "path": dest,
            "file_sha256": sha256_file(dest),
            "weights_unchanged": _onnx_diff(source, dest) < 1e-4,
            "max_abs_diff_vs_source": _onnx_diff(source, dest),
            "ood_embedded": variant == "with_ood",
            "num_cattle": meta.get("num_cattle"),
            "num_buffalo": meta.get("num_buffalo"),
            "image_size": meta.get("image_size"),
            "normalization_baked": meta.get("normalization_baked"),
        }
    return {
        "source": source,
        "source_file_sha256": sha256_file(source),
        "variants": results,
    }


def verify(pt_result: Dict[str, Any], onnx_result: Dict[str, Any],
           ood: Dict[str, Any]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    for variant in ("with_ood", "no_ood"):
        pt_path = pt_result["variants"][variant]["path"]
        onnx_path = onnx_result["variants"][variant]["path"]

        pt_meta = read_pt_metadata(pt_path)
        onnx_meta = read_onnx_metadata(onnx_path)
        pt_bundle = inspect_and_load_pt(pt_path, device="cpu")
        onnx_bundle = inspect_and_load_onnx(onnx_path, use_cuda=False)

        expected_ood = variant == "with_ood"
        checks[f"pt_{variant}"] = {
            "ood_present": isinstance(pt_meta.get("ood"), dict),
            "ood_matches_source": (
                pt_meta.get("ood", {}).get("energy_threshold") == ood["energy_threshold"]
                if expected_ood else "ood" not in pt_meta
            ),
            "runtime_ood_source": pt_bundle.ood_config.get("source_file"),
            "energy_threshold": pt_bundle.ood_config.get("energy_threshold"),
            "msp_threshold": pt_bundle.ood_config.get("msp_threshold"),
            "cattle": pt_bundle.num_cattle,
            "buffalo": pt_bundle.num_buffalo,
        }
        checks[f"onnx_{variant}"] = {
            "ood_present": isinstance(onnx_meta.get("ood"), dict),
            "ood_matches_source": (
                onnx_meta.get("ood", {}).get("energy_threshold") == ood["energy_threshold"]
                if expected_ood else "ood" not in onnx_meta
            ),
            "runtime_ood_source": onnx_bundle.ood_config.get("source_file"),
            "energy_threshold": onnx_bundle.ood_config.get("energy_threshold"),
            "msp_threshold": onnx_bundle.ood_config.get("msp_threshold"),
            "cattle": onnx_bundle.num_cattle,
            "buffalo": onnx_bundle.num_buffalo,
        }
    return checks


def write_readme(out_dir: str, pt_result: Dict[str, Any],
                 onnx_result: Dict[str, Any], ood: Dict[str, Any]) -> None:
    body = f"""# Mutated model variants — OOD embedded vs. absent

Two weight-identical variants of the reference PT checkpoint and its ONNX
export. The only difference is the metadata block:

| Variant | PT | ONNX | `ood` metadata |
|---|---|---|---|
| With OOD | `pt/*__ood.pt` | `onnx/*__ood.onnx` | embedded (calibrated) |
| Without OOD | `pt/*__no_ood.pt` | `onnx/*__no_ood.onnx` | **absent** |

## Embedded thresholds

- energy temperature: `{ood.get('energy_temperature')}`
- energy threshold: `{ood.get('energy_threshold')}`
- MSP threshold: `{ood.get('msp_threshold')}`
- binary confidence floor: `{ood.get('binary_confidence_floor')}`
- calibrated at: `{ood.get('calibrated_at')}` on `{ood.get('calibrated_on_samples')}` samples

## Verify

```python
from src.model_introspect import (inspect_and_load_pt,
                                  inspect_and_load_onnx,
                                  read_pt_metadata, read_onnx_metadata)

with_ood = inspect_and_load_pt("pt/...__ood.pt")
print(with_ood.metadata.get("ood"))        # dict
print(with_ood.ood_config["source_file"])  # embedded:<file>

no_ood = inspect_and_load_pt("pt/...__no_ood.pt")
print(no_ood.metadata.get("ood"))          # None
print(no_ood.ood_config["source_file"])    # config:defaults

onnx = inspect_and_load_onnx("onnx/...__ood.onnx", use_cuda=False)
print(onnx.metadata.get("ood"))
```

`read_pt_metadata()` / `read_onnx_metadata()` show the raw embedded block.
Weights and graphs are unchanged — see `manifest.json` for state/file hashes and
the source-vs-mutant ONNX max-abs-diff.

> Note: `test_model.py` / the Flask API deliberately fall back to
> `outputs/ood/<model>.json` when a model has **no** embedded OOD block, so the
> `no_ood` variant may still pick up a legacy JSON by filename. Use
> `inspect_and_load_*` (as above) to observe the file's own metadata.
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt", default=DEFAULT_PT)
    parser.add_argument("--onnx", default=DEFAULT_ONNX)
    parser.add_argument("--ood-json", default=DEFAULT_OOD)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--skip-pt", action="store_true")
    parser.add_argument("--skip-onnx", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not os.path.exists(args.ood_json):
        print(f"[variants] OOD JSON not found: {args.ood_json}")
        return 1
    ood = load_ood(args.ood_json)

    os.makedirs(os.path.join(args.out_dir, "pt"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "onnx"), exist_ok=True)

    pt_result = None
    onnx_result = None
    if not args.skip_pt:
        if not os.path.exists(args.pt):
            print(f"[variants] PT source not found: {args.pt}")
            return 1
        print(f"[variants] PT source: {args.pt}")
        pt_result = make_pt(args.pt, args.out_dir, ood)
    if not args.skip_onnx:
        if not os.path.exists(args.onnx):
            print(f"[variants] ONNX source not found: {args.onnx}")
            return 1
        print(f"[variants] ONNX source: {args.onnx}")
        onnx_result = make_onnx(args.onnx, args.out_dir, ood)

    checks = verify(pt_result, onnx_result, ood) if pt_result and onnx_result else {}

    manifest = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "ood_source": args.ood_json,
        "ood": ood,
        "pt": pt_result,
        "onnx": onnx_result,
        "verification": checks,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    if pt_result and onnx_result:
        write_readme(args.out_dir, pt_result, onnx_result, ood)

    print(f"[variants] output: {args.out_dir}")
    for kind, result in (("pt", pt_result), ("onnx", onnx_result)):
        if not result:
            continue
        for variant, info in result["variants"].items():
            print(f"  {kind:4} {variant:8} -> {os.path.relpath(info['path'], args.out_dir)} "
                  f"(weights_unchanged={info['weights_unchanged']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
