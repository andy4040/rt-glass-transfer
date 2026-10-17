"""Supervisor pipeline: fail closed on validation errors, train only after smoke."""
from pathlib import Path
import os,sys,json,time,subprocess,traceback
root=Path(__file__).resolve().parent
state=root/'pipeline';state.mkdir(exist_ok=True)
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    HF_HOME=str(root/'hf-cache'),PIP_NO_CACHE_DIR='1',QT_QPA_PLATFORM='offscreen')
def status(stage,**extra):
    p=state/'status.tmp';p.write_text(json.dumps(dict(stage=stage,unix=time.time(),**extra),indent=2));p.replace(state/'status.json')
def run(stage,cmd):
    if (state/(stage+'.complete')).exists():return
    status(stage,command=list(map(str,cmd)))
    print(stage,flush=True)
    with (state/(stage+'.log')).open('a') as log:
        process=subprocess.Popen(list(map(str,cmd)),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        status(stage,pid=process.pid,command=list(map(str,cmd)))
        rc=process.wait()
    if rc:raise RuntimeError(f'{stage} exited {rc}; see pipeline/{stage}.log')
    (state/(stage+'.complete')).write_text(str(time.time()))
try:
    assert (root/'gpu_setup.complete').exists() and (root/'triangulation.complete').exists()
    run('prior_dependencies',[sys.executable,'-m','pip','install','diffusers==0.32.0','huggingface_hub==0.25.0',
        'accelerate==1.3.0','transformers==4.47.1','peft==0.7.0','omegaconf','einops','imageio','imageio-ffmpeg','braceexpand'])
    run('prepare_inputs',[sys.executable,'prepare_joint_inputs.py'])
    run('download_weights',[sys.executable,'download_prior_weights.py'])
    run('generate_priors',[sys.executable,'infer_priors_lowmem.py'])
    run('coordinate_validation',[sys.executable,'verify_prior_coordinates.py'])
    signs=json.loads((root/'prior_coordinate_verification.json').read_text())['component_signs']
    run('package_priors',[sys.executable,'prepare_priors.py','--train-images','data/train/images','--diffren-root','diffrens',
        '--output','priors','--normal-signs',*map(str,signs),'--normal-input-space','raw',
        '--coordinate-report','prior_coordinate_verification.json'])
    run('dataset_smoke',[sys.executable,'train.py','--source','data/train','--priors','priors',
        '--output','runs/dataset_smoke','--config','configs/smoke.json'])
    run('training',[sys.executable,'train.py','--source','data/train','--priors','priors',
        '--output','runs/reference_30k','--config','configs/reference_30k.json'])
    status('complete')
except Exception as error:
    previous=json.loads((state/'status.json').read_text()) if (state/'status.json').exists() else {}
    status('failed',failed_stage=previous.get('stage'),error=str(error));traceback.print_exc();sys.exit(1)
