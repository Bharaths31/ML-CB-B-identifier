import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
APP_DIR = os.path.join(PROJECT_ROOT, "flutter_app")
MODELS_DIR = os.path.join(APP_DIR, "assets", "models")


def load_index(path):
    with open(path) as f:
        mapping = json.load(f)
    ordered = [None] * len(mapping)
    for name, idx in mapping.items():
        ordered[idx] = name
    if any(v is None for v in ordered):
        raise ValueError(f"gaps in index mapping: {path}")
    return ordered


def write_labels(name, items):
    out = os.path.join(MODELS_DIR, name)
    with open(out, "w") as f:
        f.write("\n".join(items) + "\n")
    print(f"wrote {len(items)} lines -> {out}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    cattle = load_index(os.path.join(SPLIT_DIR, "cattle_classes.json"))
    buffalo = load_index(os.path.join(SPLIT_DIR, "buffalo_classes.json"))
    write_labels("labels_binary.txt", ["cattle", "buffalo"])
    write_labels("labels_cattle.txt", cattle)
    write_labels("labels_buffalo.txt", buffalo)

    backbone = sys.argv[1] if len(sys.argv) > 1 else "lite2"
    ckpt = os.path.join(PROJECT_ROOT, "outputs", "checkpoints",
                        f"{backbone}_phase2_best.pt")
    if os.path.exists(ckpt):
        print(f"\nfound checkpoint {ckpt}; exporting static-batch ONNX ...")
        subprocess.run([sys.executable, "-m", "src.export", "--backbone",
                        backbone, "--mode", "onnx", "--onnx-static-batch"],
                       cwd=PROJECT_ROOT, check=True)
    else:
        print(f"\nno checkpoint at {ckpt}; skipping ONNX export")
        print("train first (python -m src.train) then re-run this script")

    print("""
Next: convert the ONNX export for each target platform.

  pip install onnx2tf onnx tf2onnx  (uses the onnx runtime)

  1) Android/iOS TFLite  (ai-edge-litert, formerly TFLite):
       ai-edge-litert --output_path=flutter_app/assets/models/model.tflite \\
           outputs/export/lite2_fp32.onnx
       # verify input tensor is [1,3,260,260] float32 NCHW (no transpose)

  2) Web TF.js graph model:
       pip install onnx-tf tensorflow tfjs
       python -m onnx_tf.backend.convert -o model_tf \
           outputs/export/lite2_fp32.onnx
       tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model \\
           model_tf flutter_app/assets/models/model_web
       # produces model_web/model.json + weights bin; move .bin next to model.json
""")


if __name__ == "__main__":
    main()
