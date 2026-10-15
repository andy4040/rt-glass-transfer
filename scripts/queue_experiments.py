"""Wait for successful GLINT completion, then run both mask methods sequentially."""
import argparse,json,time,subprocess,sys,os,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--glint-base',type=Path,required=True);p.add_argument('--glint-run',type=Path,required=True);args=p.parse_args()
BASE=args.glint_base;PRE=args.glint_run;RUNTIME=ROOT/'runtime';RUNTIME.mkdir(exist_ok=True)
CONFIG=json.loads((ROOT/'config.json').read_text());RP=ROOT/'.venv/bin/python';GP=BASE/'env-glint/bin/python'
def status(phase,**kw):
    temp=RUNTIME/'status.tmp';temp.write_text(json.dumps({'phase':phase,'updated_unix':time.time(),**kw},indent=2));temp.replace(RUNTIME/'status.json')
def run(command,log_name):
    print('RUN',json.dumps([str(x) for x in command]),flush=True)
    with (RUNTIME/log_name).open('a') as log:
        result=subprocess.run([str(x) for x in command],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env={**os.environ,'CUDA_HOME':'/usr/local/cuda-12.1','MAX_JOBS':'2','TORCH_CUDA_ARCH_LIST':'8.6','LD_LIBRARY_PATH':str(BASE/'optix-libs')})
    if result.returncode:raise RuntimeError(f'{log_name}: exit {result.returncode}; inspect log')
def complete_model(out,step):
    folder=out/'point_cloud'/f'iteration_{step}'
    return all((folder/name).is_file() for name in ['point_cloud.ply','light_mlp.pt','dir_encoding.pt'])
def main():
    # Single persistent queue; no dependent CUDA work before completion is proven.
    import fcntl
    lock=(RUNTIME/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        s=json.loads((PRE/'comparison/status.json').read_text(encoding='utf-8-sig'))
        status('waiting_for_glint',predecessor=s)
        if s['phase'] in ['failed','stopped_by_user']:raise RuntimeError('GLINT did not finish successfully; dependent experiments not started')
        cp=PRE/'model/checkpoints/step_059999.pt'
        log=(PRE/'training.log').read_text(errors='replace')
        if s['phase']=='complete' and cp.exists() and 'training complete; checkpoint=' in log and time.time()-cp.stat().st_mtime>30:break
        time.sleep(30)
    if not (ROOT/'install.complete').exists():raise RuntimeError('RT environment installation did not complete; inspect install.log')
    run([RP,'-m','pytest','tests','-q'],'tests.log')
    for method in CONFIG['methods']:
        out=ROOT/'outputs'/method;maskdir=ROOT/'masks'/method;data=ROOT/'data'/method
        if (out/'complete.json').exists():continue
        status('exporting_masks',method=method)
        if not (maskdir/'manifest.json').exists():
            run([GP,ROOT/'scripts/export_masks.py','--method',method,'--glint-root',BASE/'GLINT','--train-root',BASE/'data/train','--checkpoint',cp,'--output',maskdir],f'{method}_masks.log')
        manifest=json.loads((maskdir/'manifest.json').read_text())
        assert manifest['method']==method and manifest['held_out_used'] is False
        status('preparing_dataset',method=method)
        if not (data/'manifest.json').exists():
            run([GP,ROOT/'scripts/prepare_data.py','--source',BASE/'data/2dgs_train','--masks',maskdir,'--output',data],f'{method}_prepare.log')
        from prepare_data import validate_inputs
        validate_inputs(data,maskdir)
        # Short isolated real renderer/backward validation, after GLINT completion.
        smoke=ROOT/'outputs'/f'{method}_smoke'
        if not (RUNTIME/f'{method}_smoke.complete').exists():
            run([RP,ROOT/'scripts/train_rt.py','-s',data,'-m',smoke,'-r','1','--iterations','3','--init_until_iter','0','--norm_loss_from_iter','0','--save_iterations','3'],f'{method}_smoke.log')
            (RUNTIME/f'{method}_smoke.complete').touch()
        status('training',method=method,target=CONFIG['iterations'])
        command=[RP,ROOT/'scripts/train_rt.py','-s',data,'-m',out,'-r','1','--iterations',CONFIG['iterations'],'--init_until_iter',CONFIG['init_until_iter'],'--norm_loss_from_iter',CONFIG['norm_loss_from_iter'],'--save_iterations',*CONFIG['save_iterations']]
        if not complete_model(out,CONFIG['iterations']):
            if (out/'initial_sha256.txt').exists():
                raise RuntimeError(f'{method}: partial training exists. Exact optimizer resume unsupported; preserve outputs and explicitly select a new attempt directory.')
            run(command,f'{method}_train.log')
        if method=='diffusion':
            assert (out/'initial_sha256.txt').read_text()==(ROOT/'outputs/glint/initial_sha256.txt').read_text(),'Initial states differ; comparison invalid'
        status('evaluating',method=method)
        run([RP,ROOT/'scripts/evaluate_rt.py','--model-dir',out,'--iteration',CONFIG['iterations'],'--glint-eval-root',BASE/'data/eval','--evaluation-regions',BASE/'evaluation_regions.json','--output-dir',out/'evaluation','--init-until-iter',CONFIG['init_until_iter']],f'{method}_eval.log')
        (out/'complete.json').write_text(json.dumps({'method':method,'iteration':CONFIG['iterations'],'completed_unix':time.time()}))
    status('complete',methods=CONFIG['methods'])
try:main()
except Exception as e:
    status('failed',error=str(e));traceback.print_exc();sys.exit(1)
