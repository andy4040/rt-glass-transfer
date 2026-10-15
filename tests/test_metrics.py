"""CPU-only checks guarding ROI normalization and evaluation config parsing."""
import importlib.util
import math
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_rt.py"
SPEC = importlib.util.spec_from_file_location("evaluate_rt", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MetricsTests(unittest.TestCase):
    def test_roi_uses_only_selected_pixels(self):
        reference = np.zeros((10, 10, 3))
        prediction = reference.copy()
        prediction[:5, :5] = 1
        self.assertAlmostEqual(MODULE.roi_psnr(prediction, reference, (0, 0, 5, 5)), 0.0)
        self.assertAlmostEqual(MODULE.psnr(prediction, reference), -10 * math.log10(0.25))

    def test_outside_error_does_not_change_roi_score(self):
        reference = np.zeros((10, 10, 3))
        prediction = np.ones_like(reference)
        prediction[2:6, 3:8] = 0.1
        self.assertAlmostEqual(MODULE.roi_psnr(prediction, reference, (3, 2, 8, 6)), 20.0)

    def test_rejects_empty_roi_and_nonfinite(self):
        image = np.zeros((5, 5, 3))
        with self.assertRaises(ValueError):
            MODULE.roi_psnr(image, image, (0, 0, 0, 4))
        image[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            MODULE.psnr(image, image)

    def test_roi_reference_size_and_camera_guard(self):
        annotation = {"camera": "cam00", "frozen_before_training": True,
                      "annotation_reference_size": [1352, 1014], "window_xyxy": [170, 448, 813, 662]}
        self.assertEqual(MODULE.scaled_roi(annotation, 676, 507), (85, 224, 406, 331))
        annotation["camera"] = "cam01"
        with self.assertRaises(ValueError):
            MODULE.scaled_roi(annotation, 676, 507)

    def test_namespace_config_is_not_executed(self):
        path = Path("cfg_args")
        with patch.object(Path, "read_text", return_value="Namespace(sh_degree=3, env_scope_center=[0.0, 1.0, 2.0])"):
            self.assertEqual(MODULE.read_namespace(path)["sh_degree"], 3)
        with patch.object(Path, "read_text", return_value="Namespace(sh_degree=__import__('os').getcwd())"):
            with self.assertRaises(ValueError):
                MODULE.read_namespace(path)

    def test_detection_ignores_outside_roi(self):
        target = np.zeros((4, 4), dtype=bool)
        target[1, 1] = True
        valid = np.zeros_like(target)
        valid[1:3, 1:3] = True
        probability = np.ones((4, 4))
        probability[1:3, 1:3] = 0
        probability[1, 1] = 0.5
        result = MODULE.localization_metrics(probability, target, valid)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["iou"], 1)
        self.assertEqual(result["scored_pixels"], 4)

    def test_polygon_exclusion_and_boundary_ignore(self):
        annotation = {"camera": "cam00", "frozen_before_training": True,
                      "annotation_reference_size": [20, 20], "window_xyxy": [2, 2, 18, 18],
                      "glass_positive_polygons": [[[4, 4], [15, 4], [15, 15], [4, 15]]],
                      "opaque_exclusion_polygons": [[[8, 4], [9, 4], [9, 15], [8, 15]]],
                      "boundary_ignore_radius_at_eval_resolution": 1}
        target, valid = MODULE.glass_annotation(annotation, 20, 20)
        self.assertTrue(target[6, 6])
        self.assertFalse(target[6, 8])
        self.assertFalse(valid[4, 6])
        self.assertTrue(valid[6, 6])
        self.assertFalse(valid[0, 0])


if __name__ == "__main__":
    unittest.main()
