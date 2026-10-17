"""No CUDA or PyTorch dependency: detect unsafe resume and source drift early."""
import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from rt_source_guard import patch_training_source, reject_checkpoint_resume, replace_exact


class RTSourceGuardTests(unittest.TestCase):
    def test_rejects_resume_forms_and_argparse_abbreviations(self):
        for args in (['--start_checkpoint', 'model.pth'],
                     ['--start_checkpoint=model.pth'],
                     ['--start_check=model.pth']):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, 'resume is unsupported'):
                reject_checkpoint_resume(args)
        reject_checkpoint_resume(['-s', '/data', '--save_iterations', '5000'])

    def test_executable_rejects_resume_before_framework_import(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/train_rt.py'),
                                 '--start_checkpoint=missing.pth'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('resume is unsupported', result.stderr)
        self.assertNotIn('No module named', result.stderr)

    def test_every_replacement_requires_one_anchor(self):
        self.assertEqual(replace_exact('before old after', 'old', 'new'), 'before new after')
        for source in ('absent', 'old old'):
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                replace_exact(source, 'old', 'new')

    def test_audited_source_is_patched_and_compiles(self):
        source = (ROOT / 'vendor/RT-Splatting/train.py').read_text(encoding='utf-8')
        patched = patch_training_source(source)
        tree = ast.parse(patched)
        training = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'training')
        calls = [node for node in ast.walk(training) if isinstance(node, ast.Call)]
        called_names = [node.func.id for node in calls if isinstance(node.func, ast.Name)]
        self.assertEqual(called_names.count('mask_terms'), 1)
        self.assertNotIn('training_report', called_names)
        self.assertEqual(called_names.count('record'), 1)
        self.assertEqual(called_names.count('initial_hash'), 1)
        self.assertIn('view_rng.integers', patched)
        self.assertNotIn('np.random.randint(len(viewpoint_stack))', patched)
        self.assertEqual(patched, patch_training_source(source.replace('\n', '\r\n')))
        compile(patched, 'audited_rt_train.py', 'exec')

    def test_changed_upstream_is_rejected_instead_of_silently_skipping_patch(self):
        source = (ROOT / 'vendor/RT-Splatting/train.py').read_text(encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Unaudited RT'):
            patch_training_source(source.replace('np.random.randint', 'np.random.choice'))


if __name__ == '__main__':
    unittest.main()
