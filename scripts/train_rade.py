"""Run the installed IGS RaDe trainer, adding only train guards and status logging."""
import os, sys, json, time, types
from pathlib import Path
RADE=Path(os.environ.get('RADE_ROOT','/opt/igs-source/submodules/RaDe-GS'))
sys.path.insert(0,str(RADE))
import torch
import scene
from utils import camera_utils
original=camera_utils.loadCam
def load_camera(args, identifier, info, scale):
    assert info.image_name!='cam00', 'cam00 must not enter RaDe training'
    return original(args, identifier, info, scale)
camera_utils.loadCam=load_camera
started=time.time()
def record(model, iteration, path, loss):
    if iteration%100:return
    torch.cuda.synchronize()
    value=dict(iteration=iteration,gaussians=len(model.get_xyz),loss=float(loss.detach()),
               elapsed_seconds=time.time()-started,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    dest=Path(path);temp=dest/'status.tmp';temp.write_text(json.dumps(value));temp.replace(dest/'status.json')
    print('RADE_STATUS',json.dumps(value),flush=True)
source=(RADE/'train.py').read_text()
needle='        iter_end.record()'
assert source.count(needle)==1
source=source.replace(needle,needle+'\n        record(gaussians, iteration, dataset.model_path, loss)')
module=types.ModuleType('__main__');module.__file__=str(RADE/'train.py');module.__dict__['record']=record
sys.modules['__main__']=module
os.chdir(RADE)
exec(compile(source,str(RADE/'train.py'),'exec'),module.__dict__)
