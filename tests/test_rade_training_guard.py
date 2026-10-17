"""Reject IGS scene crops and extra train-view withholding before CUDA imports."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from rade_training_guard import validate_rade_arguments


class RaDeTrainingGuard(unittest.TestCase):
    def test_generic_path(self):
        self.assertEqual(validate_rade_arguments(['--loader', 'colmap', '-s', 'data']), 'colmap')
        self.assertEqual(validate_rade_arguments(['-s', 'data']), '')

    def test_scene_crop_paths_rejected(self):
        for flag in ('--loader', '--load', '-l'):
            for loader in ('N3D', 'meet', 'enerf'):
                with self.assertRaisesRegex(ValueError, 'world-z'):
                    validate_rade_arguments([flag, loader])
        with self.assertRaisesRegex(ValueError, 'world-z'):
            validate_rade_arguments(['--loader=N3D'])

    def test_no_second_split(self):
        with self.assertRaisesRegex(ValueError, 'withhold'):
            validate_rade_arguments(['--loader', 'colmap', '--eval'])

    def test_crop_rejected_before_framework_load(self):
        result = subprocess.run([sys.executable, str(ROOT/'scripts/train_rade.py'),
                                 '--loader', 'N3D'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('absolute-world-z', result.stderr)
        self.assertNotIn('ModuleNotFoundError', result.stderr)


if __name__ == '__main__':
    unittest.main()
