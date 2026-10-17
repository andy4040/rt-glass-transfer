"""Sequential static experiment: RaDe initialization first, original SfM second."""
import fcntl, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
from experiment_provenance import (artifact_hashes, completed, immutable_json,
    record_completion, require_free_disk, require_supported_config, require_teacher, signature)
ROOT=Path(__file__).resolve().parents[1]
BASE=Path(os.environ.get('GLINT_BASE','/opt/experiments/glint_n3dv_static_20261014'))
RUN=Path(os.environ.get('COMPARISON_RUN',str(ROOT/'runs/rade_vs_sfm_20261015')))
RUN.mkdir(parents=True,exist_ok=True)
RP=Path(os.environ.get('RT_PYTHON',str(ROOT/'.venv/bin/python')))
GP=Path(os.environ.get('GLINT_PYTHON',str(BASE/'env-glint/bin/python')))
IP=Path(os.environ.get('RADE_PYTHON','/opt/igs-venvs/igs/bin/python'))
CONFIG=json.loads((ROOT/'config.json').read_text())
CONFIG={k:v for k,v in CONFIG.items() if k not in ('methods','diffusion_mask')}
TEACHER=BASE/'results/glint/model/checkpoints/step_059999.pt'
DATA=RUN/'data';MASKS=RUN/'masks';RADE=RUN/'rade'
MIN_FREE_DISK_BYTES=int(float(os.environ.get('MIN_FREE_DISK_GIB','3'))*2**30)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def status(phase,**kw):
    value=dict(phase=phase,updated_unix=time.time(),order=['rade','sfm'],**kw)
    temp=RUN/'status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(RUN/'status.json')

def run(command,name,extra=None):
    command=list(map(str,command));print('RUN',json.dumps(command),flush=True)
    (RUN/(name+'.command.json')).write_text(json.dumps(command,indent=2))
    env={**os.environ,'CUDA_HOME':os.environ.get('CUDA_HOME','/usr/local/cuda-12.1'),
         'MAX_JOBS':os.environ.get('MAX_JOBS','2'),
         'TORCH_CUDA_ARCH_LIST':os.environ.get('TORCH_CUDA_ARCH_LIST','8.6'),
         'LD_LIBRARY_PATH':str(BASE/'optix-libs')+':'+os.environ.get('LD_LIBRARY_PATH',''),**(extra or {})}
    with (RUN/(name+'.log')).open('a') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        (RUN/'child.pid').write_text(str(child.pid))
        rc=child.wait()
    if rc:raise RuntimeError(f'{name} exited {rc}; see {RUN/(name+".log")}')

def complete(out,step):
    return all(p.is_file() and p.stat().st_size > 0 for p in model_artifacts(out,step))

def model_artifacts(out,step):
    folder=out/'point_cloud'/f'iteration_{step}'
    return [folder/n for n in ('point_cloud.ply','light_mlp.pt','dir_encoding.pt')]

def source_hashes():
    files=[ROOT/'scripts'/name for name in (
        'queue_initialization_comparison.py','experiment_provenance.py','prepare_data.py',
        'export_masks.py','train_rade.py','train_rt.py','rade_initialization.py',
        'mask_loss.py','evaluate_rt.py','rt_source_guard.py','rade_training_guard.py')]
    rade_root=Path(os.environ.get('RADE_ROOT','/opt/igs-source/submodules/RaDe-GS'))
    for base in (ROOT/'vendor/RT-Splatting',rade_root):
        files.extend(base/name for name in ('train.py','scene/gaussian_model.py',
            'scene/__init__.py','scene/dataset_readers.py','arguments/__init__.py',
            'gaussian_renderer/__init__.py'))
    return artifact_hashes(files)

def main():
    lock=(RUN/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require_supported_config(CONFIG)
    (RUN/'queue.pid').write_text(str(os.getpid()))
    expected=os.environ.get('TEACHER_SHA256','450796e4fe03ea3699b733f9adb97d7f5594c8f68161d975307f46378b4421e7')
    assert sha(TEACHER)==expected, 'Teacher hash differs from the declared experiment'
    protocol=dict(order=['rade','sfm'],scene='coffee_martini',frame=0,resolution=[676,507],
        teacher=str(TEACHER),teacher_sha256=sha(TEACHER),shared_masks=True,test_camera='cam00',
        test_used_for_training_or_selection=False,rade_iterations=30000,rade_normal_start=10000,
        rade_source=os.environ.get('RADE_ROOT','/opt/igs-source/submodules/RaDe-GS'),
        rade_original_igs_commit='f4b1880c5230834e276c8f8793ffaac364210a37',
        rade_source_tree_patched=True,
        rade_loader='colmap',igs_absolute_z_pruning=False,
        min_free_disk_bytes=MIN_FREE_DISK_BYTES,
        evaluation_regions_sha256=sha(BASE/'evaluation_regions.json'),
        restart_note=os.environ.get('RESTART_NOTE','Original 20261015 teacher'),
        implementation_sha256=source_hashes(),
        rt_config=CONFIG,rt_commit=subprocess.check_output(['git','-C',str(ROOT/'vendor/RT-Splatting'),'rev-parse','HEAD'],text=True).strip(),
        transfer='RaDe xyz, tangent scales, orientation, filtered occupancy; nearest original SfM RGB; no learned SH',
        caveats=['3D ellipsoid to 2D surfel conversion is not rendering-equivalent.',
                 'Different Gaussian counts and extra RaDe pretraining are part of initialization treatment.',
                 'No guarantee that RaDe fixes glass depth; no depth ground truth.',
                 'Both arms share the same mask files, cameras, RT schedule and material defaults.'])
    immutable_json(RUN/'protocol.json',protocol)
    experiment_signature=signature(protocol)
    status('validating')
    run([RP,'-m','pytest','tests','-q'],'tests')
    if not (MASKS/'manifest.json').exists():
        status('exporting_shared_glint_masks')
        run([GP,ROOT/'scripts/export_masks.py','--method','glint','--glint-root',BASE/'GLINT',
             '--train-root',BASE/'data/train','--checkpoint',TEACHER,'--output',MASKS],'masks')
    require_teacher(json.loads((MASKS/'manifest.json').read_text()),expected)
    if not (DATA/'manifest.json').exists():
        run([GP,ROOT/'scripts/prepare_data.py','--source',BASE/'data/2dgs_train','--masks',MASKS,
             '--output',DATA,'--mode','symlink'],'prepare')
    from prepare_data import validate_inputs
    validate_inputs(DATA,MASKS,expected_teacher_sha256=expected,require_packaged_masks=True)
    inventory={p.relative_to(DATA).as_posix():sha(p) for d in ('images','transparent_masks','mask_valid','sparse/0') for p in (DATA/d).iterdir() if p.is_file()}
    immutable_json(RUN/'shared_input_hashes.json',inventory)
    # Bind every stage to both declared settings and the actual packaged inputs.
    experiment_signature=signature(dict(protocol=experiment_signature,inputs=inventory))
    final_rade=RADE/'point_cloud/iteration_30000/point_cloud.ply'
    if not completed(RUN/'rade.complete',experiment_signature,[final_rade]):
        if RADE.exists() and any(RADE.iterdir()):
            raise RuntimeError('Partial RaDe run exists; preserve and select explicit resume checkpoint.')
        require_free_disk(RUN,MIN_FREE_DISK_BYTES)
        status('rade_pretraining',target=30000)
        run([IP,ROOT/'scripts/train_rade.py','-s',DATA,'-m',RADE,'--loader','colmap','-r','1',
             '--iterations','30000','--regularization_from_iter','10000','--test_iterations','-1',
             '--save_iterations','5000','10000','20000','30000','--checkpoint_iterations','10000','20000','30000','--port','6019'],'rade_pretrain')
        assert final_rade.is_file()
        record_completion(RUN/'rade.complete',experiment_signature,[final_rade])
    for method in ('rade','sfm'):
        out=RUN/method if method=='sfm' else RUN/'rt_rade'
        method_signature=signature(dict(experiment=experiment_signature,method=method,
            initialization_sha256=sha(final_rade) if method=='rade' else inventory['sparse/0/points3D.ply']))
        saved_artifacts=[p for step in CONFIG['save_iterations'] for p in model_artifacts(out,step)]
        final_artifacts=[*saved_artifacts,*[out/'evaluation'/str(step)/'metrics.json' for step in CONFIG['save_iterations']]]
        if completed(out/'complete.json',method_signature,final_artifacts):continue
        env={'RADE_INIT_PLY':str(final_rade) if method=='rade' else '', 'INIT_REPORT_DIR':str(out)}
        smoke=RUN/(method+'_smoke')
        if not completed(RUN/(method+'_smoke.complete'),method_signature,model_artifacts(smoke,3)):
            if smoke.exists() and any(smoke.iterdir()):
                raise RuntimeError(f'Partial smoke outputs preserved: {smoke}; use an explicitly reviewed fresh output')
            require_free_disk(RUN,MIN_FREE_DISK_BYTES)
            status('rt_smoke',method=method)
            run([RP,ROOT/'scripts/train_rt.py','-s',DATA,'-m',smoke,'-r','1','--iterations','3',
                 '--init_until_iter','0','--norm_loss_from_iter','0','--save_iterations','3'],method+'_smoke',
                 {**env,'INIT_REPORT_DIR':str(smoke)})
            assert complete(smoke,3)
            record_completion(RUN/(method+'_smoke.complete'),method_signature,model_artifacts(smoke,3))
        train_completion=RUN/(method+'_train.complete.json')
        if not completed(train_completion,method_signature,saved_artifacts):
            if out.exists() and any(out.iterdir()):raise RuntimeError('Partial RT run preserved; exact resume unsupported')
            require_free_disk(RUN,MIN_FREE_DISK_BYTES)
            status('rt_training',method=method,target=CONFIG['iterations'])
            run([RP,ROOT/'scripts/train_rt.py','-s',DATA,'-m',out,'-r','1','--iterations',CONFIG['iterations'],
                 '--init_until_iter',CONFIG['init_until_iter'],'--norm_loss_from_iter',CONFIG['norm_loss_from_iter'],
                 '--save_iterations',*CONFIG['save_iterations']],method+'_train',env)
            record_completion(train_completion,method_signature,saved_artifacts)
        assert inventory=={k:sha(DATA/k) for k in inventory},'Shared inputs changed'
        if method=='sfm':
            assert (out/'shared_modules_sha256.txt').read_text()==(RUN/'rt_rade/shared_modules_sha256.txt').read_text(),'Shared appearance networks differ at initialization'
        for step in CONFIG['save_iterations']:
            evaluation=out/'evaluation'/str(step)
            eval_signature=signature(dict(method=method_signature,step=step,
                model=artifact_hashes(model_artifacts(out,step)),
                regions_sha256=sha(BASE/'evaluation_regions.json')))
            if completed(evaluation/'complete.json',eval_signature,[evaluation/'metrics.json']):continue
            if evaluation.exists():
                raise RuntimeError(f'Partial or unverified evaluation preserved: {evaluation}')
            status('evaluating',method=method,iteration=step)
            run([RP,ROOT/'scripts/evaluate_rt.py','--model-dir',out,'--iteration',step,
                 '--glint-eval-root',BASE/'data/eval','--evaluation-regions',BASE/'evaluation_regions.json',
                 '--output-dir',evaluation,'--init-until-iter',CONFIG['init_until_iter']],f'{method}_eval_{step}')
            record_completion(evaluation/'complete.json',eval_signature,[evaluation/'metrics.json'])
        record_completion(out/'complete.json',method_signature,final_artifacts)
    status('complete')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=str(exc));traceback.print_exc();sys.exit(1)
