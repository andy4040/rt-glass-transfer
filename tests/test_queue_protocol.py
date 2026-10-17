"""Exercise the real queue control flow with synthetic outputs and no CUDA work."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))


class QueueProtocol(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        base=self.root/'glint';run=self.root/'comparison'
        teacher=base/'results/glint/model/checkpoints/step_059999.pt'
        teacher.parent.mkdir(parents=True);teacher.write_bytes(b'synthetic teacher')
        self.teacher_sha=hashlib.sha256(teacher.read_bytes()).hexdigest()
        (base/'evaluation_regions.json').write_text('{"fixed_window": true}')
        self.enterContext(patch.dict(os.environ,{
            'GLINT_BASE':str(base),'COMPARISON_RUN':str(run),
            'TEACHER_SHA256':self.teacher_sha,'MIN_FREE_DISK_GIB':'3'}))
        # File-lock behavior is outside this control-flow regression; make this
        # suite runnable on Windows as well as the Linux training host.
        fcntl=types.ModuleType('fcntl')
        fcntl.LOCK_EX=2;fcntl.LOCK_NB=4;fcntl.flock=lambda stream,flags:stream.close()
        with patch.dict(sys.modules,{'fcntl':fcntl}):
            spec=importlib.util.spec_from_file_location('_test_comparison_queue',SCRIPTS/'queue_initialization_comparison.py')
            self.queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.queue)
        q=self.queue
        q.MASKS.mkdir(parents=True)
        (q.MASKS/'manifest.json').write_text(json.dumps({
            'method':'glint','checkpoint_sha256':self.teacher_sha}))
        for directory in ('images','transparent_masks','mask_valid','sparse/0'):
            (q.DATA/directory).mkdir(parents=True)
        for directory in ('images','transparent_masks','mask_valid'):
            (q.DATA/directory/'cam01.png').write_bytes(b'synthetic pixels')
        for name in ('cameras.txt','images.txt','points3D.txt','points3D.ply'):
            (q.DATA/'sparse/0'/name).write_bytes(b'synthetic geometry')
        (q.DATA/'manifest.json').write_text('{}')
        # Geometry/mask parser behavior has dedicated tests; do not load image
        # libraries or models here. Teacher matching remains the real guard.
        preparation=types.ModuleType('prepare_data')
        preparation.validate_inputs=lambda *args,**kwargs:None
        self.enterContext(patch.dict(sys.modules,{'prepare_data':preparation}))
        self.enterContext(patch.object(q,'source_hashes',return_value={'synthetic_source.py':'fixed-source'}))
        self.enterContext(patch.object(q,'require_free_disk',return_value=20*2**30))
        self.enterContext(patch.object(q.subprocess,'check_output',return_value='fixed-rt-commit\n'))
        self.calls=[]
        self.enterContext(patch.object(q,'run',side_effect=self.fake_run))

    def fake_run(self,command,name,extra=None):
        command=list(map(str,command));self.calls.append((name,command))
        if len(command)<2:return
        entry=Path(command[1]).name
        if entry=='train_rade.py':
            output=Path(command[command.index('-m')+1])/'point_cloud/iteration_30000/point_cloud.ply'
            output.parent.mkdir(parents=True);output.write_bytes(b'RaDe final geometry')
        elif entry=='train_rt.py':
            output=Path(command[command.index('-m')+1])
            steps=command[command.index('--save_iterations')+1:]
            for step in steps:
                for artifact in self.queue.model_artifacts(output,int(step)):
                    artifact.parent.mkdir(parents=True,exist_ok=True)
                    artifact.write_bytes(f'{output.name}/{step}/{artifact.name}'.encode())
            (output/'initial_sha256.txt').write_text('synthetic-initial-state')
            (output/'shared_modules_sha256.txt').write_text('shared-networks')
        elif entry=='evaluate_rt.py':
            output=Path(command[command.index('--output-dir')+1])
            output.mkdir(parents=True)
            (output/'metrics.json').write_text('{"psnr": 1.0}')

    def test_order_loader_and_completed_rerun(self):
        self.queue.main()
        training=[(name,command) for name,command in self.calls
                  if len(command)>1 and Path(command[1]).name in ('train_rade.py','train_rt.py')]
        self.assertEqual([name for name,_ in training],
                         ['rade_pretrain','rade_smoke','rade_train','sfm_smoke','sfm_train'])
        first=training[0][1]
        self.assertEqual(first[first.index('--loader')+1],'colmap')
        labels=[name for name,_ in self.calls]
        self.assertEqual(len(labels),len(set(labels)),'Stages must not overwrite command records or mix logs')
        self.calls.clear();self.queue.main()
        self.assertEqual([name for name,_ in self.calls],['tests'])
        self.assertEqual(json.loads((self.queue.RUN/'status.json').read_text())['phase'],'complete')

    def test_stale_teacher_mask_rejected(self):
        self.queue.main();self.calls.clear()
        (self.queue.MASKS/'manifest.json').write_text(json.dumps({'method':'glint','checkpoint_sha256':'old-teacher'}))
        with self.assertRaisesRegex(ValueError,'Mask teacher'):
            self.queue.main()
        self.assertEqual([name for name,_ in self.calls],['tests'])

    def test_changed_packaged_labels_rejected_on_restart(self):
        self.queue.main();self.calls.clear()
        (self.queue.DATA/'mask_valid/cam01.png').write_bytes(b'changed label')
        with self.assertRaisesRegex(RuntimeError,'Immutable experiment record'):
            self.queue.main()
        self.assertEqual([name for name,_ in self.calls],['tests'])

    def test_changed_config_rejected_without_relabeling(self):
        self.queue.main();self.calls.clear()
        old=(self.queue.RUN/'protocol.json').read_bytes()
        self.queue.CONFIG['init_until_iter']+=1
        with self.assertRaisesRegex(RuntimeError,'Immutable experiment record'):
            self.queue.main()
        self.assertEqual(self.calls,[])
        self.assertEqual((self.queue.RUN/'protocol.json').read_bytes(),old)

    def test_modified_completed_output_rejected(self):
        self.queue.main();self.calls.clear()
        output=self.queue.model_artifacts(self.queue.RUN/'rt_rade',self.queue.CONFIG['iterations'])[0]
        output.write_bytes(b'tampered completed model')
        with self.assertRaisesRegex(RuntimeError,'Completed output changed'):
            self.queue.main()
        self.assertEqual([name for name,_ in self.calls],['tests'])

    def test_partial_rade_before_first_status_preserved(self):
        self.queue.RADE.mkdir(parents=True)
        partial=self.queue.RADE/'cfg_args';partial.write_bytes(b'first startup output')
        with self.assertRaisesRegex(RuntimeError,'Partial RaDe'):
            self.queue.main()
        self.assertEqual(partial.read_bytes(),b'first startup output')
        self.assertEqual([name for name,_ in self.calls],['tests'])

    def test_partial_rt_before_initial_hash_preserved(self):
        output=self.queue.RUN/'rt_rade';output.mkdir()
        partial=output/'cfg_args';partial.write_bytes(b'first startup output')
        with self.assertRaisesRegex(RuntimeError,'Partial RT'):
            self.queue.main()
        self.assertEqual(partial.read_bytes(),b'first startup output')
        self.assertEqual([name for name,_ in self.calls],['tests','rade_pretrain','rade_smoke'])


if __name__=='__main__':
    unittest.main()
