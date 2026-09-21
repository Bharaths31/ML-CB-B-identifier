import argparse
import glob
import json
import os
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn

from .config import (CHECKPOINT_DIR, EXPORT_DIR, IMAGENET_MEAN, IMAGENET_STD,
                     IMAGE_SIZE, PORTABLE_EXPORT_DIR, SPLIT_DIR,
                     SPECIES_LABELS, TFLITE_APP_ASSETS_DIR)
from .model import BreedClassifier

MOBILE_INPUT_RANGE = "[0, 1] RGB float32 (NCHW), normalization baked into the graph"


# ---------------------------------------------------------------------------
# Export wrappers
# ---------------------------------------------------------------------------

class _RawOutputs(nn.Module):
    """Legacy convention: the CALLER normalizes with ImageNet mean/std.

    Used for `--mode onnx` / `--mode float16` so test_model.py keeps working.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["binary"], out["cattle"], out["buffalo"]


class _MobileOutputs(nn.Module):
    """Mobile convention: input is RGB in [0, 1]; ImageNet normalization is
    baked into the graph as constant buffers.

    This makes the on-device preprocessing (pixel / 255.0, as done by the
    Flutter engine) exactly match training-time normalization with zero app
    changes, and it quantizes cleanly because mean/std are graph constants.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        out = self.model((x - self.mean) / self.std)
        return out["binary"], out["cattle"], out["buffalo"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _size(path):
    return os.path.getsize(path) / (1024 * 1024)


def _sanitize_state_dict(state):
    """Make a checkpoint loadable into a plain float BreedClassifier.

    Phase-3 QAT checkpoints (and torch.compile checkpoints) carry keys that a
    non-quantized model does not have:
      * `_orig_mod.` / `module.` prefixes from torch.compile / DataParallel
      * `.weight_fake_quant.*`, `.activation_post_process.*` observers added by
        prepare_qat
      * fused `_bn0/_bn1/_bn2` BatchNorm keys after conv-BN fusion
    Previously these made every ONNX/FP16/INT8 export raise a RuntimeError.
    We strip the extra keys and drop fused BN entries so the remaining float
    weights load; the caller uses strict=False to tolerate any leftovers.
    """
    if not isinstance(state, dict):
        return state
    cleaned = {}
    for key, value in state.items():
        k = key
        for prefix in ("module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        if ".weight_fake_quant" in k or ".activation_post_process" in k:
            continue
        if k.endswith((".fake_quant_enabled", ".observer_enabled",
                       ".scale", ".zero_point", ".min_val", ".max_val",
                       ".eps")):
            continue
        cleaned[k] = value
    return cleaned


def _load_model(checkpoint_path, backbone, attention):
    model = BreedClassifier(backbone=backbone, attention=attention)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = _sanitize_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The projection head is training-only; ignore it. Anything else missing is
    # a real problem worth surfacing.
    missing = [k for k in missing if not k.startswith("projection_head.")]
    if missing:
        raise RuntimeError(
            f"checkpoint {checkpoint_path} does not match BreedClassifier"
            f"({backbone}+{attention}); missing keys: {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[export] ignored {len(unexpected)} unexpected keys "
              f"(e.g. {unexpected[:3]})")
    model.eval()
    return model


def _write_label_files(split_dir, out_dir):
    """Emit labels_binary.txt / labels_cattle.txt / labels_buffalo.txt.

    Lines are ordered by class index (line i == class i) so the mobile app
    can map output indices to breed names directly.
    """
    written = []
    with open(os.path.join(out_dir, "labels_binary.txt"), "w") as f:
        f.write("\n".join([s for s, _ in sorted(SPECIES_LABELS.items(),
                                                key=lambda kv: kv[1])]) + "\n")
    written.append("labels_binary.txt")
    for map_name, out_name in (("cattle_classes.json", "labels_cattle.txt"),
                               ("buffalo_classes.json", "labels_buffalo.txt")):
        src = os.path.join(split_dir, map_name)
        if not os.path.exists(src):
            continue
        with open(src) as f:
            classes = json.load(f)
        ordered = [name for name, _ in sorted(classes.items(), key=lambda kv: kv[1])]
        with open(os.path.join(out_dir, out_name), "w") as f:
            f.write("\n".join(ordered) + "\n")
        written.append(out_name)
    return written


def _copy_to_app_assets(paths_by_name):
    """Copy mobile artifacts into flutter_app/assets/models/ when present.

    paths_by_name maps source file path -> destination filename inside the
    app assets dir (model.tflite / labels_*.txt).
    """
    if not os.path.isdir(TFLITE_APP_ASSETS_DIR):
        return
    for src, dst_name in paths_by_name.items():
        if os.path.exists(src):
            dst = os.path.join(TFLITE_APP_ASSETS_DIR, dst_name)
            shutil.copy2(src, dst)
            print(f"[export] app assets: {dst}")


def _calibration_images(split_dir, limit=500):
    """Yield (1, 3, H, W) float32 arrays in [0, 1] from the train split."""
    import io

    from PIL import Image
    from torchvision import transforms

    tfm = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    csv_path = os.path.join(split_dir, "train.csv")
    if not os.path.exists(csv_path):
        return
    import pandas as pd
    df = pd.read_csv(csv_path)
    if len(df) > limit:
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)
    for path in df["path"]:
        if not os.path.exists(path):
            continue
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                yield tfm(img).unsqueeze(0).numpy().astype(np.float32)
        except Exception:
            continue


def _onnx_input_name(path):
    import onnx
    model = onnx.load(path)
    return model.graph.input[0].name


# ---------------------------------------------------------------------------
# ONNX Runtime Mobile INT8 (QDQ static quantization)
# ---------------------------------------------------------------------------

def export_onnx_int8(model, onnx_fp32_path, out_path, split_dir,
                     calibration_images=500):
    """Static INT8 QDQ quantization for ONNX Runtime Mobile.

    Per-channel weights + per-tensor activations, calibrated on real training
    images through the mobile wrapper (input [0,1], normalization inside).
    """
    from onnxruntime.quantization import (CalibrationDataReader,
                                          CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)

    class _Reader(CalibrationDataReader):
        def __init__(self, split_dir, input_name, limit):
            self._gen = _calibration_images(split_dir, limit=limit)
            self._input_name = input_name

        def get_next(self):
            arr = next(self._gen, None)
            return None if arr is None else {self._input_name: arr}

    print("[export] ONNX INT8 (QDQ) calibration + quantization...")
    reader = _Reader(split_dir, _onnx_input_name(onnx_fp32_path),
                     calibration_images)
    quantize_kwargs = dict(
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )
    try:
        quantize_static(onnx_fp32_path, out_path, reader,
                        calibrate_method=CalibrationMethod.MinMax,
                        **quantize_kwargs)
    except TypeError:
        # onnxruntime < 1.18 used the old kwarg name
        quantize_static(onnx_fp32_path, out_path, reader,
                        calibration_method=CalibrationMethod.MinMax,
                        **quantize_kwargs)
    print(f"[export] ONNX INT8 -> {out_path} ({_size(out_path):.2f} MB)")


# ---------------------------------------------------------------------------
# TFLite (fp32 + full-integer INT8 PTQ)
# ---------------------------------------------------------------------------

def export_tflite(model, backbone, out_dir, split_dir, static_batch=True,
                  calibration_images=500):
    """PyTorch -> ONNX -> (onnx2tf) -> TFLite fp32 -> full-integer INT8 PTQ.

    Returns (int8_path, fp32_path). Requires `onnx2tf` + `tensorflow`.
    """
    try:
        import tensorflow as tf  # noqa: F401  (dependency guard)
    except ImportError as exc:
        raise RuntimeError(
            "TFLite export needs TensorFlow: pip install tensorflow\n"
            "plus the ONNX conversion toolchain: pip install onnx2tf "
            "tf-keras onnx-graphsurgeon sng4onnx onnxsim") from exc

    onnx_dir = os.path.join(out_dir, "onnx_tmp")
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, f"{backbone}_mobile_fp32.onnx")
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    wrapper = _MobileOutputs(model).eval()
    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["input"],
        output_names=["binary", "cattle", "buffalo"],
        opset_version=17,
        dynamic_axes=None if static_batch else
        {"input": {0: "batch"}, "binary": {0: "batch"},
         "cattle": {0: "batch"}, "buffalo": {0: "batch"}},
        dynamo=False,
    )
    print(f"[export] mobile ONNX (normalization baked) -> {onnx_path}")

    tf_dir = os.path.join(out_dir, "tf_tmp")
    if os.path.isdir(tf_dir):
        shutil.rmtree(tf_dir)
    cmd = ["onnx2tf", "-i", onnx_path, "-o", tf_dir,
           "-osd", "-otf", "--non_verbose",
           "--copy_onnx_input_output_names_to_tflite"]
    print(f"[export] running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        raise RuntimeError(
            "onnx2tf not found. Install the TFLite conversion toolchain:\n"
            "  pip install onnx2tf tensorflow tf-keras onnx-graphsurgeon "
            "sng4onnx onnxsim")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"onnx2tf conversion failed (exit {exc.returncode}). "
                           f"Re-run with verbose output to debug.")

    tflites = sorted(glob.glob(os.path.join(tf_dir, "**", "*.tflite"),
                               recursive=True))
    if not tflites:
        raise RuntimeError("onnx2tf produced no .tflite file")
    fp32_src = next((p for p in tflites
                     if "float32" in os.path.basename(p)), tflites[0])
    fp32_path = os.path.join(out_dir, f"{backbone}_fp32.tflite")
    shutil.copy2(fp32_src, fp32_path)
    print(f"[export] TFLite FP32 -> {fp32_path} ({_size(fp32_path):.2f} MB)")

    # --- Full-integer INT8 PTQ on the SavedModel ---
    saved_model_dir = os.path.join(tf_dir, "saved_model")
    if not os.path.isdir(saved_model_dir):
        raise RuntimeError(f"onnx2tf did not emit a SavedModel at {saved_model_dir}")

    def representative_dataset():
        for arr in _calibration_images(split_dir, limit=calibration_images):
            yield [arr]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # INT8 compute with float32 [0,1] I/O: the converter inserts a Quantize op
    # after the input and Dequantize before the outputs, so the Flutter engine
    # keeps feeding Float32List pixel/255 values with zero app changes while
    # every conv/BN/activation inside runs as INT8.
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    int8_bytes = converter.convert()

    int8_path = os.path.join(out_dir, f"{backbone}_int8.tflite")
    with open(int8_path, "wb") as f:
        f.write(int8_bytes)
    print(f"[export] TFLite INT8 -> {int8_path} ({_size(int8_path):.2f} MB)")
    return int8_path, fp32_path


# ---------------------------------------------------------------------------
# Portable export — self-contained model folder
# ---------------------------------------------------------------------------

def create_portable_export(checkpoint_path, backbone, split_dir=SPLIT_DIR,
                           export_dir=PORTABLE_EXPORT_DIR):
    """Bundle the trained model + labels + metadata into a portable folder.

    The resulting folder is self-contained and can be copied/zipped for use
    anywhere — no project dependency needed.
    """
    tag = os.path.splitext(os.path.basename(checkpoint_path))[0]
    out_dir = os.path.join(export_dir, f"{backbone}_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[export] creating portable bundle -> {out_dir}")

    # Copy checkpoint
    dst_ckpt = os.path.join(out_dir, "model.pt")
    shutil.copy2(checkpoint_path, dst_ckpt)
    print(f"[export]   model.pt ({_size(dst_ckpt):.2f} MB)")

    # Copy class maps
    for name in ("cattle_classes.json", "buffalo_classes.json"):
        src = os.path.join(split_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, name))
            print(f"[export]   {name}")

    # Write model info
    info = {
        "backbone": backbone,
        "checkpoint_source": os.path.basename(checkpoint_path),
        "image_size": 260,
        "num_cattle_breeds": 57,
        "num_buffalo_breeds": 18,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "usage": {
            "load": (
                "model = BreedClassifier(backbone='{backbone}'); "
                "ckpt = torch.load('model.pt', map_location='cpu'); "
                "model.load_state_dict(ckpt['state_dict']); model.eval()"
            ).format(backbone=backbone),
            "predict": (
                "out = model(tensor)  # tensor: [B, 3, 260, 260]\n"
                "species = out['binary'].argmax(1)  # 0=cattle, 1=buffalo\n"
                "breed = out['cattle' if species==0 else 'buffalo'].argmax(1)"
            ),
        },
    }
    with open(os.path.join(out_dir, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[export]   model_info.json")

    print(f"[export] portable bundle complete: {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export the trained model for desktop testing or mobile "
                    "deployment (TFLite INT8 / ONNX Runtime Mobile INT8)")
    parser.add_argument("--backbone", choices=["lite2", "lite4"], default="lite2")
    parser.add_argument("--attention", choices=["cbam", "se"], default="cbam")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", choices=["int8", "onnx", "onnx-int8",
                                           "float16", "portable", "tflite"],
                        default="onnx")
    parser.add_argument("--onnx-static-batch", action="store_true",
                        help="export ONNX with fixed batch size 1")
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="TFLite path: keep dynamic batch in the ONNX step "
                             "(default is static batch 1 for mobile)")
    parser.add_argument("--calibration-images", type=int, default=500,
                        help="train-split images used for INT8 calibration")
    parser.add_argument("--data", default=None, help="(unused, kept for compat)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="(unused, kept for compat)")
    parser.add_argument("--out-dir", default=EXPORT_DIR)
    parser.add_argument("--split-dir", default=SPLIT_DIR)
    parser.add_argument("--skip-app-assets", action="store_true",
                        help="do not copy artifacts into flutter_app/assets/models")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or os.path.join(
        CHECKPOINT_DIR, f"{args.backbone}_phase2_best.pt")
    if not os.path.exists(checkpoint_path):
        print(f"[export] checkpoint not found: {checkpoint_path}")
        return 1

    model = _load_model(checkpoint_path, args.backbone, args.attention)
    print(f"[export] loaded {checkpoint_path}")

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"{args.backbone}")

    if args.mode == "portable":
        create_portable_export(checkpoint_path, args.backbone,
                               args.split_dir, PORTABLE_EXPORT_DIR)
        return 0

    if args.mode == "float16":
        print("[export] converting to float16...")
        model.half()
        try:
            traced = torch.jit.trace(_RawOutputs(model), torch.randn(1, 3, 260, 260).half())
            path = base + "_float16.pt"
            traced.save(path)
            print(f"[export] float16 TorchScript -> {path} "
                  f"({_size(path):.2f} MB)")
        except Exception as exc:
            print(f"[export] float16 export failed ({exc})")
        return 0

    if args.mode == "onnx":
        # Legacy fp32 ONNX for test_model.py: caller normalizes the input.
        print("[export] exporting to ONNX (fp32, caller-normalized input)...")
        wrapper = _RawOutputs(model).eval()
        path = base + "_fp32.onnx"
        dynamic_axes = None if args.onnx_static_batch else {
            "input": {0: "batch"},
            "binary": {0: "batch"},
            "cattle": {0: "batch"},
            "buffalo": {0: "batch"}}
        try:
            torch.onnx.export(wrapper, torch.randn(1, 3, 260, 260), path,
                              input_names=["input"],
                              output_names=["binary", "cattle", "buffalo"],
                              opset_version=13, dynamic_axes=dynamic_axes,
                              dynamo=False)
            print(f"[export] ONNX -> {path} ({_size(path):.2f} MB)")
        except Exception as exc:
            print(f"[export] ONNX export failed ({exc})")
            return 1
        return 0

    if args.mode == "onnx-int8":
        # Mobile wrapper: [0,1] input, normalization baked in.
        onnx_fp32 = base + "_mobile_fp32.onnx"
        wrapper = _MobileOutputs(model).eval()
        try:
            torch.onnx.export(wrapper, torch.randn(1, 3, 260, 260), onnx_fp32,
                              input_names=["input"],
                              output_names=["binary", "cattle", "buffalo"],
                              opset_version=17, dynamic_axes=None,
                              dynamo=False)
            print(f"[export] mobile ONNX -> {onnx_fp32} "
                  f"({_size(onnx_fp32):.2f} MB)")
        except Exception as exc:
            print(f"[export] ONNX export failed ({exc})")
            return 1
        int8_path = base + "_mobile_int8.onnx"
        try:
            export_onnx_int8(model, onnx_fp32, int8_path, args.split_dir,
                             calibration_images=args.calibration_images)
            labels = _write_label_files(args.split_dir, args.out_dir)
            print(f"[export] labels written: {', '.join(labels)}")
            print(f"[export] input convention: {MOBILE_INPUT_RANGE}")
        except Exception as exc:
            print(f"[export] ONNX INT8 quantization failed ({exc})")
            return 1
        return 0

    if args.mode == "tflite":
        try:
            int8_path, fp32_path = export_tflite(
                model, args.backbone, args.out_dir, args.split_dir,
                static_batch=not args.dynamic_batch,
                calibration_images=args.calibration_images)
        except Exception as exc:
            print(f"[export] TFLite export failed: {exc}")
            return 1
        labels = _write_label_files(args.split_dir, args.out_dir)
        print(f"[export] labels written: {', '.join(labels)}")
        print(f"[export] input convention: {MOBILE_INPUT_RANGE}")

        if not args.skip_app_assets and os.path.isdir(TFLITE_APP_ASSETS_DIR):
            _copy_to_app_assets({
                int8_path: "model.tflite",
                fp32_path: f"{args.backbone}_fp32.tflite",
                os.path.join(args.out_dir, "labels_binary.txt"): "labels_binary.txt",
                os.path.join(args.out_dir, "labels_cattle.txt"): "labels_cattle.txt",
                os.path.join(args.out_dir, "labels_buffalo.txt"): "labels_buffalo.txt",
            })
        # Clean the large intermediate TF/ONNX dirs, keep the fp32 ONNX.
        for junk in ("tf_tmp", "onnx_tmp"):
            p = os.path.join(args.out_dir, junk)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
        return 0

    if args.mode == "int8":
        print("[export] The x86 PTQ INT8 path was removed: it produced a "
              "'Unsupported qscheme: per_channel_affine' failure and its "
              "artifacts were not usable on Android. Use --mode tflite "
              "(TFLite INT8) or --mode onnx-int8 (ONNX Runtime Mobile).")
        return 1

    print(f"[export] unknown mode {args.mode}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
