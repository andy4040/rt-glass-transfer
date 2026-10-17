"""CPU-only regression tests for accidental stale or modified run reuse."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment_provenance import (completed, immutable_json, record_completion,
    require_copied_masks, require_free_disk, require_supported_config, require_teacher, signature)


class Provenance(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_protocol_cannot_be_relabelled(self):
        p=self.root/'protocol.json'
        immutable_json(p,{'teacher':'a','iterations':10})
        before=p.read_bytes()
        immutable_json(p,{'iterations':10,'teacher':'a'})
        with self.assertRaisesRegex(RuntimeError,'Immutable'):
            immutable_json(p,{'teacher':'b','iterations':10})
        self.assertEqual(p.read_bytes(),before)

    def test_mask_teacher_must_match(self):
        require_teacher({'method':'glint','checkpoint_sha256':'a'},'a')
        for manifest in ({'method':'glint','checkpoint_sha256':'old'},
                         {'method':'diffusion','checkpoint_sha256':'a'},{}):
            with self.assertRaisesRegex(ValueError,'teacher'):
                require_teacher(manifest,'a')

    def test_actual_training_mask_copy_checked(self):
        data,masks=self.root/'data',self.root/'masks'
        for root in (data,masks):
            for name in ('transparent_masks','mask_valid'):
                (root/name).mkdir(parents=True)
                (root/name/'cam01.png').write_bytes(b'label')
        require_copied_masks(data,masks)
        (data/'mask_valid/cam01.png').write_bytes(b'changed training label')
        with self.assertRaisesRegex(ValueError,'Packaged mask differs'):
            require_copied_masks(data,masks)

    def test_missing_training_mask_rejected(self):
        data,masks=self.root/'data',self.root/'masks'
        for name in ('transparent_masks','mask_valid'):
            (masks/name).mkdir(parents=True)
            (masks/name/'cam01.png').write_bytes(b'label')
        with self.assertRaisesRegex(ValueError,'inventory'):
            require_copied_masks(data,masks)

    def test_completion_tied_to_signature_and_output(self):
        output=self.root/'output.ply';output.write_bytes(b'complete')
        marker=self.root/'done.json'
        self.assertFalse(completed(marker,'run-a',[output]))
        record_completion(marker,'run-a',[output])
        self.assertTrue(completed(marker,'run-a',[output]))
        with self.assertRaisesRegex(RuntimeError,'different'):
            completed(marker,'run-b',[output])
        output.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'changed'):
            completed(marker,'run-a',[output])

    def test_legacy_and_empty_outputs_fail_closed(self):
        marker=self.root/'old.complete';marker.touch()
        with self.assertRaisesRegex(RuntimeError,'legacy'):
            completed(marker,'run',[])
        output=self.root/'empty.ply';output.touch()
        with self.assertRaisesRegex(RuntimeError,'empty'):
            record_completion(self.root/'new.complete','run',[output])

    def test_signature_does_not_depend_on_mapping_order(self):
        self.assertEqual(signature({'a':1,'b':2}),signature({'b':2,'a':1}))
        self.assertNotEqual(signature({'a':1}),signature({'a':2}))

    def test_disk_floor(self):
        with patch('experiment_provenance.shutil.disk_usage',return_value=SimpleNamespace(free=20)):
            self.assertEqual(require_free_disk(self.root,10),20)
            with self.assertRaisesRegex(RuntimeError,'Insufficient'):
                require_free_disk(self.root,30)
            with self.assertRaises(ValueError):
                require_free_disk(self.root,0)

    def test_unsupported_seeds_and_split_rejected(self):
        config={'seed':0,'view_sequence_seed':20261015,'held_out_camera':'cam00',
                'scene':'coffee_martini','frame':0,'resolution':[676,507]}
        require_supported_config(config)
        for key,changed in (('seed',1),('view_sequence_seed',9),('held_out_camera','cam07')):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'Unsupported'):
                require_supported_config({**config,key:changed})


if __name__=='__main__':
    unittest.main()
