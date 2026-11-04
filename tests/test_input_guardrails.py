"""Unit tests for the 4-layer input guardrails.

Run with::

    python -m unittest tests.test_input_guardrails -v

No CLIP/OpenCV download is required; those stages are optional and gracefully
skip when unavailable.
"""

import glob
import os
import unittest

import torch
from PIL import Image, ImageDraw

from src.input_guardrails import (InputGuardrailEngine,
                                  verdict_to_ood_explanation)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(PROJECT_ROOT, "data", "external",
                        "atharvadarpude_buffalo", "buffalo")


def _sketch():
    image = Image.new("RGB", (400, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((80, 80, 320, 320), outline="black", width=6)
    draw.line((200, 80, 200, 320), fill="black", width=6)
    return image


def _cartoon():
    image = Image.new("RGB", (400, 400), (30, 160, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 100, 350, 300), fill=(240, 200, 60))
    draw.ellipse((150, 150, 250, 250), fill=(20, 20, 20))
    return image


class VisualGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.engine = InputGuardrailEngine(clip_enabled=False)

    def test_sketch_is_rejected_with_localized_copy(self):
        verdict = self.engine.validate_visual(_sketch())
        self.assertTrue(verdict["rejected"])
        self.assertEqual(verdict["reason_code"], "sketch_or_line_drawing")
        info = verdict["rejection_info"]
        for lang in ("en", "ta", "hi"):
            self.assertIn(lang, info["title"])
            self.assertIn(lang, info["message"])
            self.assertIn(lang, info["instruction"])
        explanation = verdict_to_ood_explanation(verdict)
        self.assertEqual(explanation["code"], "sketch_or_line_drawing")
        self.assertTrue(explanation["tips_by_language"]["ta"])

    def test_cartoon_is_rejected(self):
        verdict = self.engine.validate_visual(_cartoon())
        self.assertTrue(verdict["rejected"])
        self.assertEqual(verdict["reason_code"], "cartoon_or_digital_graphic")

    def test_solid_colour_is_not_a_camera_photo(self):
        verdict = self.engine.validate_visual(Image.new("RGB", (400, 400), (120, 90, 60)))
        self.assertTrue(verdict["rejected"])

    @unittest.skipUnless(os.path.isdir(REAL_DIR), "external buffalo dataset missing")
    def test_real_photo_passes_visual_stage(self):
        paths = glob.glob(os.path.join(REAL_DIR, "*", "*.jpg"))
        if not paths:
            self.skipTest("no real buffalo images")
        verdict = self.engine.validate_visual(Image.open(paths[0]).convert("RGB"))
        self.assertFalse(verdict["rejected"], verdict)


class ModelOutputGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.engine = InputGuardrailEngine(clip_enabled=False)

    def test_confident_distribution_is_accepted(self):
        binary = torch.tensor([[9.0, -9.0]])
        cattle = torch.full((1, 50), -9.0)
        cattle[0, 7] = 12.0
        buffalo = torch.full((1, 18), -9.0)
        features = torch.full((1, 1280), 0.2)
        verdict = self.engine.validate_model_outputs(
            {"binary": binary, "cattle": cattle, "buffalo": buffalo,
             "features": features})
        self.assertFalse(verdict["rejected"], verdict)
        self.assertLess(verdict["details"]["entropy"], 0.5)
        self.assertGreater(verdict["details"]["top1"], 0.5)

    def test_uniform_distribution_is_rejected(self):
        verdict = self.engine.validate_model_outputs({
            "binary": torch.zeros(1, 2),
            "cattle": torch.zeros(1, 50),
            "buffalo": torch.zeros(1, 18),
            "features": torch.ones(1, 1280),
        })
        self.assertTrue(verdict["rejected"])
        self.assertEqual(verdict["reason_code"], "uncertain_animal")

    def test_logit_ood_stage_rejects(self):
        class _Detector:
            def is_ood(self, binary, cattle, buffalo):
                return True, 1.23, {"energy": 1.23, "ood_by_energy": True}

        binary = torch.tensor([[9.0, -9.0]])
        cattle = torch.full((1, 50), -9.0)
        cattle[0, 7] = 12.0
        buffalo = torch.full((1, 18), -9.0)
        verdict = self.engine.validate_model_outputs(
            {"binary": binary, "cattle": cattle, "buffalo": buffalo},
            _Detector())
        self.assertTrue(verdict["rejected"])
        self.assertEqual(verdict["reason_code"], "out_of_distribution")

    def test_feature_norm_outlier_rejected(self):
        binary = torch.tensor([[9.0, -9.0]])
        cattle = torch.full((1, 50), -9.0)
        cattle[0, 7] = 12.0
        buffalo = torch.full((1, 18), -9.0)
        verdict = self.engine.validate_model_outputs({
            "binary": binary, "cattle": cattle, "buffalo": buffalo,
            "features": torch.full((1, 1280), 10.0),  # norm ~357
        })
        self.assertTrue(verdict["rejected"])
        self.assertEqual(verdict["details"].get("trigger"), "feature_norm")


if __name__ == "__main__":
    unittest.main()
