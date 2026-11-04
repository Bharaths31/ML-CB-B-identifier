"""Unit tests for the self-describing model layer.

Run with::

    python -m unittest tests.test_model_introspect -v

Artifact-dependent tests skip cleanly when ``outputs/`` is empty (CI).
"""

import hashlib
import importlib.util
import os
import shutil
import tempfile
import unittest

import torch

from src.config import (EXPECTED_BUFFALO_BREEDS, EXPECTED_CATTLE_BREEDS,
                        IMAGE_SIZE)
from src.model_introspect import (
    build_transform,
    embed_onnx_metadata,
    embed_pt_metadata,
    extract_state_dict,
    inspect_and_load_onnx,
    inspect_and_load_pt,
    inspect_state_dict,
    read_onnx_metadata,
    read_pt_metadata,
    resolve_class_maps,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V26_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "outputs", "checkpoints",
    "lite2_phase2_best_v26_1012-2332_f9ce.pt")
V27_ONNX = os.path.join(
    PROJECT_ROOT, "outputs", "export",
    "lite2_v27_1020-1201_0547_fp32.onnx")


def _synthetic_state(num_cattle=50, num_buffalo=18, cosine=False):
    state = {
        "backbone.blocks.0.0._project_conv.weight": torch.zeros(16, 3, 1, 1),
        "attention.spatial_attention.conv.weight": torch.zeros(1, 2, 7, 7),
        "attention.channel_attention.mlp.0.weight": torch.zeros(5, 88, 1, 1),
        "binary_head.0.weight": torch.zeros(512, 1280),
        "cattle_head.0.weight": torch.zeros(384, 1280),
        "cattle_head.8.weight": torch.zeros(num_cattle, 192),
        "buffalo_head.8.weight": torch.zeros(num_buffalo, 192),
    }
    if not cosine:
        state["cattle_head.8.bias"] = torch.zeros(num_cattle)
        state["buffalo_head.8.bias"] = torch.zeros(num_buffalo)
    return state


class TensorIntrospectionTests(unittest.TestCase):
    def test_architecture_from_shapes(self):
        spec = inspect_state_dict(_synthetic_state())
        self.assertEqual(spec["backbone"], "lite2")
        self.assertEqual(spec["attention"], "cbam")
        self.assertEqual(spec["feature_dim"], 1280)
        self.assertEqual(spec["binary_dim"], 512)
        self.assertEqual(spec["breed_dim"], 384)
        self.assertEqual(spec["num_cattle"], 50)
        self.assertEqual(spec["num_buffalo"], 18)
        self.assertFalse(spec["cosine_head"])

    def test_lite4_and_cosine(self):
        state = _synthetic_state(cosine=True)
        state["backbone.blocks.0.0._project_conv.weight"] = torch.zeros(24, 3, 1, 1)
        spec = inspect_state_dict(state)
        self.assertEqual(spec["backbone"], "lite4")
        self.assertTrue(spec["cosine_head"])

    def test_class_map_priority(self):
        cattle, buffalo, source = resolve_class_maps(50, 18)
        self.assertEqual(source, "config:EXPECTED_*_BREEDS")
        self.assertEqual(list(cattle)[:2], list(EXPECTED_CATTLE_BREEDS)[:2])
        self.assertEqual(len(buffalo), len(EXPECTED_BUFFALO_BREEDS))

        cattle, buffalo, source = resolve_class_maps(
            7, 3, embedded={"cattle_classes": {f"c{i}": i for i in range(7)},
                            "buffalo_classes": {f"b{i}": i for i in range(3)}})
        self.assertEqual(source, "embedded")
        self.assertEqual(cattle["c6"], 6)

        cattle, _, source = resolve_class_maps(7, 3)
        self.assertEqual(source, "synthetic")
        self.assertEqual(len(cattle), 7)

    def test_embedded_map_count_mismatch_is_rejected(self):
        cattle, buffalo, source = resolve_class_maps(
            50, 18, embedded={"cattle_classes": {"only": 0},
                              "buffalo_classes": {"only": 0}})
        self.assertEqual(source, "config:EXPECTED_*_BREEDS")
        self.assertEqual(len(cattle), 50)

    def test_transform_normalization_modes(self):
        caller = build_transform(IMAGE_SIZE, normalization_baked=False)
        baked = build_transform(IMAGE_SIZE, normalization_baked=True)
        self.assertIn("Normalize", repr(caller))
        self.assertNotIn("Normalize", repr(baked))
        self.assertEqual(caller.transforms[0].size, IMAGE_SIZE)


@unittest.skipUnless(os.path.exists(V26_CHECKPOINT), "no v26 checkpoint")
class CheckpointRoundTripTests(unittest.TestCase):
    def test_load_introspects_without_json(self):
        bundle = inspect_and_load_pt(V26_CHECKPOINT, device="cpu")
        self.assertEqual(bundle.engine, "pt")
        self.assertEqual(bundle.num_cattle, 50)
        self.assertEqual(bundle.num_buffalo, 18)
        self.assertEqual(bundle.image_size, IMAGE_SIZE)
        self.assertFalse(bundle.normalization_baked)
        self.assertEqual(len(bundle.cattle_inv_map), 50)
        self.assertEqual(bundle.cattle_inv_map[0], EXPECTED_CATTLE_BREEDS[0])

    def test_embed_preserves_weights_and_is_idempotent(self):
        def state_hash(path):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            state = extract_state_dict(ckpt)
            digest = hashlib.sha256()
            for key in sorted(state):
                digest.update(key.encode())
                digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        before = state_hash(V26_CHECKPOINT)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pt")
            shutil.copy2(V26_CHECKPOINT, path)
            meta = embed_pt_metadata(path, overwrite=True)
            self.assertEqual(meta["num_cattle"], 50)
            self.assertEqual(meta["num_buffalo"], 18)
            self.assertEqual(state_hash(path), before)
            first = read_pt_metadata(path)
            second = embed_pt_metadata(path, overwrite=True)
            self.assertEqual(first["num_cattle"], second["num_cattle"])
            self.assertEqual(read_pt_metadata(path)["backbone"], "lite2")


@unittest.skipUnless(os.path.exists(V27_ONNX), "no v27 ONNX export")
class OnnxRoundTripTests(unittest.TestCase):
    def test_normalization_detection_and_embedding(self):
        if not (importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime")):
            self.skipTest("onnx/onnxruntime not installed")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.onnx")
            shutil.copy2(V27_ONNX, path)
            bundle = inspect_and_load_onnx(path, use_cuda=False)
            self.assertEqual(bundle.num_cattle, 50)
            self.assertEqual(bundle.num_buffalo, 18)
            self.assertFalse(bundle.normalization_baked)

            meta = embed_onnx_metadata(path, overwrite=True)
            self.assertFalse(meta["normalization_baked"])
            stored = read_onnx_metadata(path)
            self.assertEqual(stored["model_meta_version"], "1.0")
            self.assertEqual(stored["num_cattle"], 50)
            self.assertEqual(len(stored["cattle_classes"]), 50)

            # Re-run must not duplicate metadata keys.
            embed_onnx_metadata(path, overwrite=True)
            reread = read_onnx_metadata(path)
            self.assertEqual(reread["model_meta_version"], "1.0")


if __name__ == "__main__":
    unittest.main()
