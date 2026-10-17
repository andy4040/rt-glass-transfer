"""Coverage-aware detection regressions; no GPU or held-out images are used."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "evaluate_rt_material", Path(__file__).resolve().parents[1] / "scripts/evaluate_rt.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MaterialLocalizationTests(unittest.TestCase):
    def test_empty_scene_cannot_receive_perfect_glass_iou(self):
        zeros = np.zeros((2, 2))
        glass = valid = np.ones((2, 2), dtype=bool)
        raw = MODULE.localization_metrics(1 - zeros, glass, valid)
        result, maps = MODULE.material_localization_diagnostic(zeros, zeros, glass, valid)
        self.assertEqual(raw["iou"], 1)
        self.assertEqual(result["iou"], 0)
        self.assertEqual(result["fn"], 4)
        self.assertEqual(result["scored_pixels"], 4)
        self.assertEqual(result["no_hit_fraction"], 1)
        self.assertEqual(result["low_coverage_fraction"], 1)
        self.assertFalse(maps["prediction"].any())

    def test_partial_coverage_opaque_surface_is_not_glass(self):
        alpha = np.array([[0.4, 0.8]])
        opaque, valid = np.zeros_like(alpha, dtype=bool), np.ones_like(alpha, dtype=bool)
        result, maps = MODULE.material_localization_diagnostic(alpha, alpha, opaque, valid)
        self.assertTrue((1 - alpha[0, 0]) >= 0.5)  # Original proxy's false glass.
        np.testing.assert_array_equal(maps["conditional_transparency"], [[0, 0]])
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["no_hit_fraction"], 0)
        self.assertEqual(result["low_coverage_fraction"], 0.5)

    def test_missing_glass_stays_in_scoring_denominator(self):
        alpha = np.array([[1.0, 0.2, 0.0]])
        opacity = alpha * 0.1
        glass = valid = np.ones_like(alpha, dtype=bool)
        result, _ = MODULE.material_localization_diagnostic(opacity, alpha, glass, valid)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fn"], 2)
        self.assertAlmostEqual(result["iou"], 1 / 3)
        self.assertAlmostEqual(result["recall"], 1 / 3)
        self.assertEqual(result["scored_pixels"], 3)
        self.assertEqual(result["reference_glass_low_coverage_pixels"], 2)

    def test_coverage_statistics_use_only_fixed_annotation_mask(self):
        alpha = np.array([[1.0, 0.0]])
        valid = np.array([[True, False]])
        result, _ = MODULE.material_localization_diagnostic(alpha * 0.1, alpha, valid, valid)
        self.assertEqual(result["iou"], 1)
        self.assertEqual(result["scored_pixels"], 1)
        self.assertEqual(result["low_coverage_fraction"], 0)

    def test_reject_nonfinite_coverage(self):
        with self.assertRaises(ValueError):
            MODULE.material_localization_diagnostic(
                np.array([[0.0]]), np.array([[np.nan]]), np.array([[True]]), np.array([[True]]))


if __name__ == "__main__":
    unittest.main()
