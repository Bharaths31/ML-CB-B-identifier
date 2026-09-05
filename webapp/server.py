import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (BACKBONE_WEIGHTS, CHECKPOINT_DIR, EXPORT_DIR,
                        IMAGE_SIZE, METRICS_DIR, NUM_BUFFALO_BREEDS,
                        NUM_CATTLE_BREEDS, PORTABLE_EXPORT_DIR, SPLIT_DIR)
from src.model import BreedClassifier
from memory import Mem0Layer

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
])

app = FastAPI(title="Cattle & Buffalo Breed API", version="1.1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _softmax(logits):
    exps = torch.exp(logits - logits.max())
    return exps / exps.sum()


class LabelMap:
    def __init__(self):
        self.binary = ["cattle", "buffalo"]
        self.cattle = self._load("cattle_classes.json")
        self.buffalo = self._load("buffalo_classes.json")

    def _load(self, name):
        path = os.path.join(SPLIT_DIR, name)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            mapping = json.load(f)
        ordered = [None] * len(mapping)
        for k, v in mapping.items():
            ordered[v] = k
        return ordered


LABELS = LabelMap()


class ModelBox:
    def __init__(self):
        self.model = None
        self.backbone = None
        self.lock = threading.Lock()
        self._ckpt_mtime = None

    def _find_checkpoint(self, backbone):
        for name in (f"{backbone}_phase2_best.pt",
                     f"{backbone}_phase1_best.pt"):
            path = os.path.join(CHECKPOINT_DIR, name)
            if os.path.exists(path):
                return path
        return None

    def invalidate(self):
        """Force model reload on next prediction (after retraining)."""
        with self.lock:
            self.model = None
            self.backbone = None
            self._ckpt_mtime = None

    def ensure_loaded(self, backbone="lite2"):
        with self.lock:
            path = self._find_checkpoint(backbone)
            if path is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"No trained checkpoint for '{backbone}'. "
                           "Run training first.")
            # Reload if model changed on disk (after retraining)
            current_mtime = os.path.getmtime(path)
            if (self.model is not None and self.backbone == backbone
                    and self._ckpt_mtime == current_mtime):
                return self.model
            model = BreedClassifier(backbone=backbone)
            state = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["state_dict"])
            model.eval()
            self.model = model
            self.backbone = backbone
            self._ckpt_mtime = current_mtime
            return model

    def predict(self, image_bytes, backbone="lite2"):
        model = self.ensure_loaded(backbone)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = EVAL_TRANSFORM(image).unsqueeze(0)
        sw = time.time()
        with torch.no_grad():
            out = model(tensor)
        latency_ms = int((time.time() - sw) * 1000)

        binary_probs = _softmax(out["binary"][0])
        species_index = int(binary_probs.argmax())
        is_cattle = species_index == 0
        logits = out["cattle"][0] if is_cattle else out["buffalo"][0]
        labels = LABELS.cattle if is_cattle else LABELS.buffalo
        probs = _softmax(logits)

        order = torch.argsort(probs, descending=True)
        top3 = []
        for idx in order[:3].tolist():
            top3.append({
                "index": idx,
                "label": labels[idx] if idx < len(labels) else f"class_{idx}",
                "confidence": round(float(probs[idx]), 6),
            })
        top = top3[0]
        return {
            "species": "cattle" if is_cattle else "buffalo",
            "speciesIndex": species_index,
            "speciesConfidence": round(float(binary_probs[species_index]), 6),
            "breed": top["label"],
            "breedIndex": top["index"],
            "breedConfidence": top["confidence"],
            "top3": top3,
            "latencyMs": latency_ms,
        }


MODEL_BOX = ModelBox()


def _fold_cr(text):
    lines = []
    buf = ""
    for ch in text:
        if ch == "\r":
            buf = ""
        elif ch == "\n":
            if buf:
                lines.append(buf)
            buf = ""
        else:
            buf += ch
    if buf:
        lines.append(buf)
    return lines


class JobRunner:
    def __init__(self):
        self.proc = None
        self.lines = []
        self.status = {"running": False}
        self.lock = threading.Lock()
        self.max_lines = 3000

    def start(self, args, kind, phases=None):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                raise HTTPException(status_code=409,
                                    detail="A job is already running")
            self.lines = []
            self.status = {"running": True, "kind": kind,
                           "phase": None, "phases": phases or 3,
                           "epoch": None, "epochs": None, "batch": None,
                           "eta_s": None, "bar": None, "metrics": {},
                           "best": None, "exit": None, "progress": 0.0}
            cmd = [sys.executable, "-u", "-m"] + args
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            self.proc = subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        assert self.proc is not None
        for line in self.proc.stdout:
            for folded in _fold_cr(line):
                if not folded.strip():
                    continue
                with self.lock:
                    self.lines.append({"t": time.time(), "text": folded})
                    if len(self.lines) > self.max_lines:
                        del self.lines[:len(self.lines) - self.max_lines]
                    self._parse(folded)
        exit_code = self.proc.wait()
        with self.lock:
            self.status["running"] = False
            self.status["exit"] = exit_code
            # Invalidate model cache when training finishes so next
            # prediction picks up the newly trained weights
            if self.status.get("kind") == "train":
                MODEL_BOX.invalidate()

    def _parse(self, line):
        m = re.search(r"\[train\] phase (\d) epoch (\d+)/(\d+):"
                      r" loss=([\d.]+) ce_b=([\d.]+) ce_c=([\d.]+)"
                      r" ce_buf=([\d.]+) \| val binary=([\d.]+)"
                      r" cattle=([\d.]+) buffalo=([\d.]+) top1=([\d.]+)", line)
        if m:
            self.status["phase"] = int(m.group(1))
            self.status["epoch"] = int(m.group(2))
            self.status["epochs"] = int(m.group(3))
            self.status["metrics"] = {
                "loss": float(m.group(4)), "ce_b": float(m.group(5)),
                "ce_c": float(m.group(6)), "ce_buf": float(m.group(7)),
                "binary_acc": float(m.group(8)), "cattle_acc": float(m.group(9)),
                "buffalo_acc": float(m.group(10)), "combined_top1": float(m.group(11)),
            }
            self._update_progress()
            return
        m = re.match(r"^phase(\d+) e(\d+)/(\d+):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+)"
                     r"(?:\s+\[(\d+):(\d+)<(\d+):(\d+))?", line)
        if m:
            self.status["phase"] = int(m.group(1))
            self.status["epoch"] = int(m.group(2))
            self.status["epochs"] = int(m.group(3))
            done, total = int(m.group(5)), int(m.group(6))
            self.status["batch"] = {"done": done, "total": total,
                                    "pct": round(done * 100.0 / max(1, total), 1)}
            if m.group(9):
                self.status["eta_s"] = int(m.group(9)) * 60 + int(m.group(10))
            self._update_progress()
            return
        # Generic tqdm bar: "label: XX%|...|  N/M [...]"
        m = re.match(r"^([A-Za-z_][\w ]*?):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+)"
                     r"(?:\s+\[(\d+):(\d+)<(\d+):(\d+))?", line)
        if m:
            done, total = int(m.group(3)), int(m.group(4))
            pct = round(done * 100.0 / max(1, total), 1)
            self.status["bar"] = {"label": m.group(1).strip(), "done": done,
                                  "total": total, "pct": pct}
            if m.group(7):
                self.status["eta_s"] = int(m.group(7)) * 60 + int(m.group(8))
            # Update progress for non-training jobs too
            if self.status.get("kind") != "train":
                self.status["progress"] = pct
            return
        m = re.search(r"\[train\] phase(\d+) best val top1: ([\d.]+)", line)
        if m:
            self.status["best"] = {"phase": int(m.group(1)),
                                   "top1": float(m.group(2))}
            self.status["batch"] = None
            return
        m = re.search(r"\[train\] phase (\d): ", line)
        if m and (self.status["phase"] is None
                  or int(m.group(1)) > self.status["phase"]):
            self.status["phase"] = int(m.group(1))
            self.status["epoch"] = None
            self.status["epochs"] = None
            self.status["batch"] = None
        # Detect training completion for progress
        if "TRAINING COMPLETE" in line:
            self.status["progress"] = 100.0
        # Detect export progress
        if "[export]" in line and "portable bundle complete" in line:
            self.status["progress"] = 100.0

    def _update_progress(self):
        st = self.status
        if st.get("kind") != "train":
            return
        phase = st.get("phase")
        if not phase:
            st["progress"] = 0.0
            return
        frac = 0.0
        if st.get("epochs") and st.get("epoch"):
            batch_pct = (st.get("batch") or {}).get("pct", 0.0)
            frac = (st["epoch"] - 1 + batch_pct / 100.0) / st["epochs"]
        st["progress"] = round(min(100.0, ((phase - 1) + min(max(frac, 0.0), 1.0))
                                   / max(1, st.get("phases") or 3) * 100), 1)

    def stop(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                return True
        return False

    def snapshot(self, tail=600):
        with self.lock:
            return {
                "status": dict(self.status),
                "log": self.lines[-tail:],
            }


JOB = JobRunner()

MEMORY = Mem0Layer()


def _dataset_summary():
    counts = {"cattle": {}, "buffalo": {}}
    for species in ("cattle", "buffalo"):
        sdir = os.path.join(ROOT, "data", "raw", species)
        if not os.path.isdir(sdir):
            continue
        for breed in sorted(os.listdir(sdir)):
            bdir = os.path.join(sdir, breed)
            if not os.path.isdir(bdir):
                continue
            n = sum(1 for f in os.listdir(bdir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))
            counts[species][breed] = n
    return {
        "cattle": counts["cattle"],
        "buffalo": counts["buffalo"],
        "cattle_total": sum(counts["cattle"].values()),
        "buffalo_total": sum(counts["buffalo"].values()),
        "cattle_breeds": len(counts["cattle"]),
        "buffalo_breeds": len(counts["buffalo"]),
        "expected_cattle": NUM_CATTLE_BREEDS,
        "expected_buffalo": NUM_BUFFALO_BREEDS,
    }


def _checkpoints():
    out = {}
    if os.path.isdir(CHECKPOINT_DIR):
        for f in sorted(os.listdir(CHECKPOINT_DIR)):
            out[f] = round(os.path.getsize(os.path.join(CHECKPOINT_DIR, f)) / 1e6, 2)
    return out


def _exports():
    out = {}
    if os.path.isdir(EXPORT_DIR):
        for f in sorted(os.listdir(EXPORT_DIR)):
            p = os.path.join(EXPORT_DIR, f)
            if os.path.isfile(p):
                out[f] = round(os.path.getsize(p) / 1e6, 2)
    # Also list portable exports
    if os.path.isdir(PORTABLE_EXPORT_DIR):
        for d in sorted(os.listdir(PORTABLE_EXPORT_DIR)):
            dp = os.path.join(PORTABLE_EXPORT_DIR, d)
            if os.path.isdir(dp):
                total_mb = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for f in os.listdir(dp)
                    if os.path.isfile(os.path.join(dp, f))
                ) / 1e6
                out[f"portable/{d}"] = round(total_mb, 2)
    return out


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/status")
def api_status():
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision_version(),
        "image_size": IMAGE_SIZE,
        "num_cattle": NUM_CATTLE_BREEDS,
        "num_buffalo": NUM_BUFFALO_BREEDS,
        "model_loaded": MODEL_BOX.model is not None,
        "loaded_backbone": MODEL_BOX.backbone,
        "checkpoints": _checkpoints(),
        "exports": _exports(),
        "job_running": JOB.status.get("running", False),
        "job_kind": JOB.status.get("kind"),
    }


def torchvision_version():
    try:
        import torchvision
        return torchvision.__version__
    except ImportError:
        return "missing"


@app.get("/api/dataset")
def api_dataset():
    return _dataset_summary()


@app.get("/api/sample-image")
def api_sample_image(species: str = "cattle"):
    import random
    counts = _dataset_summary()
    breeds = counts.get(species) or {}
    candidates = [b for b, n in breeds.items() if n > 0]
    if not candidates:
        raise HTTPException(status_code=404, detail=f"no {species} images found")
    breed = random.choice(candidates)
    bdir = os.path.join(ROOT, "data", "raw", species, breed)
    files = [f for f in os.listdir(bdir)
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
    if not files:
        raise HTTPException(status_code=404, detail="no files in breed dir")
    return FileResponse(os.path.join(bdir, random.choice(files)))


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...), backbone: str = "lite2"):
    if backbone not in ("lite2", "lite4"):
        raise HTTPException(status_code=400, detail="backbone must be lite2/lite4")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    return MODEL_BOX.predict(data, backbone)


@app.post("/api/train")
def api_train(payload: dict):
    backbone = payload.get("backbone", "lite2")
    if backbone not in ("lite2", "lite4"):
        raise HTTPException(status_code=400, detail="backbone must be lite2/lite4")
    args = ["src.train", "--backbone", backbone]
    if payload.get("smoke_test"):
        args.append("--smoke-test")
    if payload.get("skip_qat"):
        args.append("--skip-qat")
    # FIXED: use hyphens not underscores for argparse args
    for key, flag in [("phase1_epochs", "--phase1-epochs"),
                      ("phase2_epochs", "--phase2-epochs"),
                      ("phase3_epochs", "--phase3-epochs")]:
        val = payload.get(key)
        if val:
            args += [flag, str(int(val))]
    if payload.get("num_workers"):
        args += ["--num-workers", str(int(payload["num_workers"]))]
    JOB.start(args, "train",
              phases=2 if payload.get("skip_qat") else 3)
    return {"started": True, "command": " ".join(args)}


@app.get("/api/job")
def api_job():
    return JOB.snapshot()


@app.post("/api/job/stop")
def api_job_stop():
    stopped = JOB.stop()
    return {"stopped": stopped}


@app.post("/api/evaluate")
def api_evaluate(payload: dict):
    backbone = payload.get("backbone", "lite2")
    if backbone not in ("lite2", "lite4"):
        raise HTTPException(status_code=400, detail="backbone must be lite2/lite4")
    JOB.start(["src.evaluate", "--backbone", backbone], "evaluate")
    return {"started": True}


@app.get("/api/metrics")
def api_metrics():
    out = {}
    if os.path.isdir(METRICS_DIR):
        for f in sorted(os.listdir(METRICS_DIR)):
            if f.endswith(".json"):
                with open(os.path.join(METRICS_DIR, f)) as fh:
                    out[f] = json.load(fh)
    return out


@app.post("/api/verify")
def api_verify():
    JOB.start(["src.verify"], "verify")
    return {"started": True}


@app.post("/api/export")
def api_export(payload: dict):
    backbone = payload.get("backbone", "lite2")
    mode = payload.get("mode", "onnx")
    if backbone not in ("lite2", "lite4"):
        raise HTTPException(status_code=400, detail="backbone must be lite2/lite4")
    if mode not in ("onnx", "int8", "float16", "portable"):
        raise HTTPException(status_code=400, detail="bad mode")
    args = ["src.export", "--backbone", backbone, "--mode", mode]
    if mode == "onnx":
        args.append("--onnx-static-batch")
    JOB.start(args, "export")
    return {"started": True}


def _memory_scope(payload):
    return {
        "user_id": (payload.get("user_id") or None),
        "agent_id": (payload.get("agent_id") or None),
        "run_id": (payload.get("run_id") or None),
    }


@app.get("/api/memory/status")
def api_memory_status(user_id: str = "", agent_id: str = "",
                      run_id: str = ""):
    try:
        return MEMORY.status(user_id=user_id or None,
                             agent_id=agent_id or None,
                             run_id=run_id or None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/memory/add")
def api_memory_add(payload: dict):
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    try:
        results = MEMORY.remember(
            text, **_memory_scope(payload),
            metadata=payload.get("metadata"),
            infer=payload.get("infer"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"results": results,
            "llm_used": MEMORY.llm_configured}


@app.post("/api/memory/recall")
def api_memory_recall(payload: dict):
    query = (payload.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    try:
        stats = MEMORY.build_context(
            query, **_memory_scope(payload),
            top_k=int(payload.get("top_k") or 5),
            max_tokens=int(payload.get("max_tokens") or 600))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return stats


@app.post("/api/memory/chat")
def api_memory_chat(payload: dict):
    message = (payload.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    try:
        result = MEMORY.chat(
            message, **_memory_scope(payload),
            system_hint=payload.get("system_hint"),
            top_k=int(payload.get("top_k") or 5),
            persist=payload.get("persist", True))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@app.get("/api/memory/all")
def api_memory_all(user_id: str = "", agent_id: str = "",
                   run_id: str = ""):
    try:
        return {"results": MEMORY.list_memories(
            user_id=user_id or None, agent_id=agent_id or None,
            run_id=run_id or None)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/memory/{memory_id}")
def api_memory_delete(memory_id: str):
    try:
        MEMORY.delete(memory_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"deleted": memory_id}


@app.post("/api/memory/clear")
def api_memory_clear(payload: dict):
    try:
        MEMORY.clear(**_memory_scope(payload))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"cleared": _memory_scope(payload)}


@app.post("/api/memory/reset")
def api_memory_reset():
    try:
        MEMORY.reset()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"reset": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
