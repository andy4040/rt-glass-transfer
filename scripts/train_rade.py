"""Run IGS RaDe on generic COLMAP inputs, with train guards and status logging."""
import os, sys, json, time, types, hashlib
from pathlib import Path
from rade_training_guard import validate_rade_arguments
validate_rade_arguments(sys.argv[1:])
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
print('RADE_SOURCE', json.dumps(dict(path=str(RADE/'train.py'),
      sha256=hashlib.sha256(source.encode()).hexdigest(),
      loader='generic COLMAP', igs_absolute_z_pruning=False)), flush=True)
needle='        iter_end.record()'
assert source.count(needle)==1
source=source.replace(needle,needle+'\n        record(gaussians, iteration, dataset.model_path, loss)')
module=types.ModuleType('__main__');module.__file__=str(RADE/'train.py');module.__dict__['record']=record
sys.modules['__main__']=module
# Preserve caller-relative paths before entering the vendored source directory.
for i, arg in enumerate(sys.argv[:-1]):
    if arg in ('-s', '--source_path', '-m', '--model_path', '--start_checkpoint'):
        sys.argv[i+1] = str(Path(sys.argv[i+1]).resolve())
os.chdir(RADE)
exec(compile(source,str(RADE/'train.py'),'exec'),module.__dict__)
