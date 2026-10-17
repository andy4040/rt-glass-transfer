"""Rebuild lost inputs and run the paired experiment on the replacement server.

Run under Supervisor. Completed stages are immutable; partial training is never
silently restarted. This is a NEW teacher, not a resume of the lost checkpoint.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/workspace/experiments/glint_n3dv_5090')
STATE = ROOT / 'runs/rebuild_5090_20261017'
STATE.mkdir(parents=True, exist_ok=True)
PY = '/venv/main/bin/python'
ENV = dict(os.environ, CUDA_HOME='/usr/local/cuda-12.8',
           HF_HOME='/workspace/experiments/hf-cache',
           TORCH_CUDA_ARCH_LIST='12.0', MAX_JOBS='4', PIP_NO_CACHE_DIR='1',
           OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', QT_QPA_PLATFORM='offscreen',
           PATH='/venv/main/bin:/usr/local/cuda-12.8/bin:' + os.environ['PATH'],
           RADE_ROOT='/workspace/experiments/igs-source/submodules/RaDe-GS')

def status(stage, **kw):
    tmp = STATE / 'status.tmp'
    tmp.write_text(json.dumps(dict(stage=stage, updated_unix=time.time(), **kw), indent=2))
    tmp.replace(STATE / 'status.json')

def run(stage, cmd, cwd=ROOT, extra=None):
    if (STATE / (stage + '.complete')).exists():
        return
    status(stage, command=list(map(str, cmd)))
    (STATE / (stage + '.command.json')).write_text(json.dumps(list(map(str, cmd))))
    print(stage, flush=True)
    with (STATE / (stage + '.log')).open('a') as log:
        child = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env={**ENV, **(extra or {})},
                                 stdout=log, stderr=subprocess.STDOUT)
        status(stage, pid=child.pid)
        rc = child.wait()
    if rc:
        raise RuntimeError(f'{stage} exited {rc}; see {STATE / (stage + ".log")}')
    (STATE / (stage + '.complete')).write_text(str(time.time()))

def main():
    import fcntl
    lock = (STATE/'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status('waiting_for_environment_install')
    while not Path('/workspace/experiments/install_5090.complete').exists():
        state = subprocess.check_output(['supervisorctl','status','experiment-install-5090'], text=True)
        if 'RUNNING' not in state and 'STARTING' not in state:
            raise RuntimeError('Environment installer stopped without completion; inspect install_5090.log/.err')
        time.sleep(15)
    # Fail rather than switch away from the recorded official GLINT implementation.
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=BASE/'GLINT',text=True).strip()
    assert head == 'a20591b20ef3365c230e3c95f749e65c498c68c4', head
    run('extra_dependencies', [PY,'-m','pip','install','trimesh','einops','lpips==0.1.4'])
    status('waiting_for_train_only_preparation')
    while not Path('/workspace/experiments/prepare_cpu_5090.complete').exists():
        state = subprocess.check_output(['supervisorctl','status','experiment-prepare-cpu-5090'], text=True)
        if 'RUNNING' not in state and 'STARTING' not in state:
            raise RuntimeError('Train-only input preparation stopped; inspect prepare_cpu_5090.log/.err')
        time.sleep(15)
    run('cpu_protocol_tests', [PY,'-m','pytest','tests','-q'])
    run('install_glint_python', [PY,'-m','pip','install','--no-deps','-e',BASE/'GLINT'])
    run('install_gsplat', [PY,'-m','pip','install','--no-build-isolation',
        'git+https://github.com/youngju-na/gsplat.git@83a0dcd3f850cccdc7b92b50fd6a568e3260eae1'],
        extra={'BUILD_2DGS':'1','NUM_CHANNELS':'3,4,6,7','BUILD_EXPERIMENTAL':'0'})
    run('install_optix', [PY,'-m','glint.install_optix','--cuda-home',ENV['CUDA_HOME'],'--jobs','4'])
    run('verify_glint_cuda', [PY,'-m','glint.verify_installation'])
    for name in ['diff-gaussian-rasterization','diff-gaussian-rasterization-clamp','compress-diff-gaussian-rasterization']:
        run('install_rade_'+name, [PY,'-m','pip','install','--no-build-isolation',Path(ENV['RADE_ROOT'])/'submodules'/name])
    dr = Path('/workspace/experiments/diffusion-renderer')
    status('waiting_for_prior_weights')
    while not Path('/workspace/experiments/download_priors_5090.complete').exists():
        state = subprocess.check_output(['supervisorctl','status','experiment-priors-download-5090'], text=True)
        if 'RUNNING' not in state and 'STARTING' not in state:
            raise RuntimeError('Prior downloader stopped; inspect download_priors_5090.log/.err')
        time.sleep(15)
    run('generate_priors_train_only', [PY,'inference_svd_rgbx.py','--config','configs/rgbx_inference.yaml',
        f'inference_input_dir={BASE}/prior_inputs', f'inference_save_dir={BASE}/prior_outputs',
        'inference_res=[384,512]', 'inference_n_frames=24','inference_n_steps=20',
        'model_passes=[normal,depth,diffuse_albedo,basecolor]','seed=0','save_video=false','save_image=true'], cwd=dr)
    run('verify_and_package_priors', [PY,BASE/'prepare_priors.py'])
    cfg = (BASE/'glint_static.yaml').read_text().replace('/opt/experiments/glint_n3dv_static_20261014', str(BASE))
    (BASE/'glint_static_5090.yaml').write_text(cfg)
    # Separate disposable transport smoke test: never reuse its state in training.
    run('glint_transport_smoke', [PY,'-m','glint.train','--config',BASE/'glint_static_5090.yaml',
        '--data-root',BASE/'data/train','--output-dir',BASE/'results/transport_smoke',
        '--max-steps','2','--transmission-start','0','--reflection-start','0',
        '--num-workers','0','--image-every','0','--save-every','0','--seed','0'])
    model = BASE/'results/glint/model'
    if not (STATE/'glint_teacher.complete').exists() and model.exists():
        raise RuntimeError('Partial teacher preserved. Explicit checkpoint resume is required.')
    run('glint_teacher', [PY,'-u',BASE/'run_measured.py','--output',BASE/'results/glint/resources.json',
        '--cwd',BASE/'GLINT','--module','glint.train','--',
        '--config',BASE/'glint_static_5090.yaml','--data-root',BASE/'data/train','--output-dir',model,
        '--max-steps','60000','--transmission-start','5000','--reflection-start','5000',
        '--interface-geometry-freeze','40000','--save-every','10000','--image-every','5000',
        '--seed','0','--num-workers','4'])
    teacher = model/'checkpoints/step_059999.pt'
    h = hashlib.sha256()
    with teacher.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    provenance = dict(teacher_sha256=h.hexdigest(), old_teacher_recovered=False,
        schedule=dict(steps=60000, transmission_start=5000, reflection_start=5000, geometry_freeze=40000),
        note='New teacher from recovered frame0/calibration, re-triangulated train-only points and regenerated priors; not an exact continuation.')
    (STATE/'teacher_provenance.json').write_text(json.dumps(provenance,indent=2))
    run('paired_comparison', [PY,ROOT/'scripts/queue_initialization_comparison.py'], extra={
        'GLINT_BASE':str(BASE),'COMPARISON_RUN':str(ROOT/'runs/rade_vs_sfm_5090_20261017'),
        'RT_PYTHON':PY,'GLINT_PYTHON':PY,'RADE_PYTHON':PY,'TEACHER_SHA256':h.hexdigest(),
        'RESTART_NOTE':provenance['note']})
    status('complete')

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=str(exc)); traceback.print_exc(); sys.exit(1)
