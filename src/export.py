import argparse
import json
import os
import shutil
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from .config import (BACKBONE_WEIGHTS, CHECKPOINT_DIR, EXPORT_DIR,
                     PORTABLE_EXPORT_DIR, RAW_DATA_DIR, SPLIT_DIR)
from .data_pipeline import get_dataloaders
from .model import BreedClassifier


class _ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["binary"], out["cattle"], out["buffalo"]


def _size(path):
    return os.path.getsize(path) / (1024 * 1024)


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


def main():
    parser = argparse.ArgumentParser(description="Quantize and export the model")
    parser.add_argument("--backbone", choices=["lite2", "lite4"], default="lite2")
    parser.add_argument("--attention", choices=["cbam", "se"], default="cbam")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", choices=["int8", "onnx", "float16", "portable"],
                        default="onnx")
    parser.add_argument("--onnx-static-batch", action="store_true",
                        help="export ONNX with fixed batch size 1 "
                             "(required for TFLite conversion)")
    parser.add_argument("--data", default=RAW_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=EXPORT_DIR)
    parser.add_argument("--split-dir", default=SPLIT_DIR)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or os.path.join(
        CHECKPOINT_DIR, f"{args.backbone}_phase2_best.pt")
    if not os.path.exists(checkpoint_path):
        print(f"[export] checkpoint not found: {checkpoint_path}")
        return 1

    model = BreedClassifier(backbone=args.backbone, attention=args.attention)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)
    print(f"[export] loaded {checkpoint_path}")

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"{args.backbone}")
    dummy = torch.randn(1, 3, 260, 260, device=device)

    if args.mode == "portable":
        create_portable_export(checkpoint_path, args.backbone,
                               args.split_dir, PORTABLE_EXPORT_DIR)
        return 0

    if args.mode == "float16":
        print("[export] converting to float16...")
        model.half()
        try:
            traced = torch.jit.trace(model, dummy.half())
            path = base + "_float16.pt"
            traced.save(path)
            print(f"[export] float16 TorchScript -> {path} "
                  f"({_size(path):.2f} MB)")
        except Exception as exc:
            print(f"[export] float16 export failed ({exc})")
        return 0

    if args.mode == "onnx":
        print("[export] exporting to ONNX...")
        wrapper = _ExportWrapper(model).to(device).eval()
        path = base + "_fp32.onnx"
        dynamic_axes = None if args.onnx_static_batch else {
            "input": {0: "batch"},
            "binary": {0: "batch"},
            "cattle": {0: "batch"},
            "buffalo": {0: "batch"}}
        try:
            torch.onnx.export(wrapper, dummy, path, input_names=["input"],
                              output_names=["binary", "cattle", "buffalo"],
                              opset_version=13, dynamic_axes=dynamic_axes,
                              dynamo=False)
            print(f"[export] ONNX -> {path} ({_size(path):.2f} MB)")
        except Exception as exc:
            print(f"[export] ONNX export failed ({exc})")
        return 0

    # INT8 quantization
    print("[export] classic torch INT8 quantization runs on CPU")
    device = "cpu"
    model = model.to("cpu")
    try:
        import torch.ao.quantization as qat
        loaders = get_dataloaders(batch_size=args.batch_size)
        if loaders is None:
            print("[export] no calibration data found; using random calibration")
            calib = None
        else:
            calib = loaders[0]
        model.eval()
        model.qconfig = qat.get_default_qconfig("x86")
        qat.prepare(model, inplace=True)
        with torch.no_grad():
            steps = 0
            if calib is not None:
                for images, _ in tqdm(calib, desc="calibrating INT8",
                                      total=32, leave=False,
                                      bar_format="{l_bar}{bar:30}{r_bar}"):
                    model(images.to("cpu"))
                    steps += 1
                    if steps >= 32:
                        break
            else:
                for _ in tqdm(range(8), desc="calibrating (random)",
                              leave=False):
                    model(dummy.to("cpu"))
        qat.convert(model, inplace=True)
        path = base + "_int8.pt"
        torch.save({"state_dict": model.state_dict()}, path)
        print(f"[export] INT8 model -> {path} ({_size(path):.2f} MB)")
        try:
            traced = torch.jit.trace(model, dummy.to("cpu"))
            traced_path = base + "_int8_traced.pt"
            traced.save(traced_path)
            print(f"[export] INT8 TorchScript -> {traced_path} "
                  f"({_size(traced_path):.2f} MB)")
        except Exception as exc:
            print(f"[export] INT8 TorchScript export failed ({exc})")
    except Exception as exc:
        print(f"[export] INT8 quantization failed ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())