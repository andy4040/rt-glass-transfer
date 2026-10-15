"""CPU tests of pseudo-label semantics and train/test isolation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from export_masks import binary_mask, check_train_names, diffusion_labels, glint_validity
from prepare_data import validate_inputs


class Labels(unittest.TestCase):
    def test_binary_threshold_and_finite(self):
        result = binary_mask(np.array([[0.49, 0.5, 1, np.nan]]))
        np.testing.assert_array_equal(result, [[0, 255, 255, 0]])
        with self.assertRaises(ValueError):
            binary_mask(np.array([[1.1]]))

    def test_glint_unknown_is_not_negative(self):
        probability = np.array([[.05, .95, .5, .95, np.nan]])
        coverage = np.array([[1, 1, 1, .2, 1]])
        np.testing.assert_array_equal(glint_validity(probability, coverage), [[True, True, False, False, False]])
        with self.assertRaises(ValueError):
            glint_validity(probability, coverage[:, :2])

    def test_diffusion_rule_and_ambiguity(self):
        albedo = np.repeat(np.array([[[.01], [.3], [.01], [.075], [.01]]]), 3, axis=-1)
        basecolor = np.repeat(np.array([[[.7], [.7], [.1], [.7], [.225]]]), 3, axis=-1)
        probability, valid = diffusion_labels(albedo, basecolor)
        np.testing.assert_array_equal(probability, [[1, 0, 0, .5, .5]])
        np.testing.assert_array_equal(valid, [[True, True, True, False, False]])

    def test_camera_leakage_and_duplicates(self):
        for names in (["cam00", "cam01"], ["cam01", "cam01"], ["../cam01"], []):
            with self.assertRaises(ValueError):
                check_train_names(names)


class Packaging(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, self.masks = self.root / "train", self.root / "masks"
        for path in (self.source / "images", self.source / "sparse/0",
                     self.masks / "transparent_masks", self.masks / "mask_valid"):
            path.mkdir(parents=True)
        sparse = self.source / "sparse/0"
        (sparse / "cameras.txt").write_text("2 PINHOLE 4 2 2 2 2 1\n")
        (sparse / "images.txt").write_text("2 1 0 0 0 0 0 0 2 cam01.png\n\n")
        (sparse / "points3D.txt").write_text("1 0 0 2 127 127 127 0 2 0\n")
        (sparse / "points3D.ply").write_bytes(b"fixture; parser does not read PLY")
        Image.new("RGB", (4, 2)).save(self.source / "images/cam01.png")
        Image.new("L", (4, 2), 255).save(self.masks / "transparent_masks/cam01.png")
        Image.new("L", (4, 2), 255).save(self.masks / "mask_valid/cam01.png")
        (self.masks / "manifest.json").write_text(json.dumps({"train_cameras": ["cam01"], "held_out_used": False}))

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_train_inventory(self):
        cameras, images, _ = validate_inputs(self.source, self.masks)
        self.assertEqual(len(cameras), 1)
        self.assertEqual(images[0]["camera"], "cam01")

    def test_extra_cam00_rejected_before_packaging(self):
        Image.new("RGB", (4, 2)).save(self.source / "images/cam00.png")
        with self.assertRaisesRegex(ValueError, "inventory"):
            validate_inputs(self.source, self.masks)

    def test_shape_mismatch_rejected(self):
        Image.new("L", (2, 2), 255).save(self.masks / "transparent_masks/cam01.png")
        with self.assertRaisesRegex(ValueError, "geometry"):
            validate_inputs(self.source, self.masks)

    def test_soft_mask_rejected(self):
        Image.new("L", (4, 2), 128).save(self.masks / "transparent_masks/cam01.png")
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_inputs(self.source, self.masks)


if __name__ == "__main__":
    unittest.main()
