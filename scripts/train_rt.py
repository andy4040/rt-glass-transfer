"""Pinned upstream RT trainer with explicit pseudo-mask validity adaptation.

No GLINT module/model/loss is imported. Rendering and representation are upstream.
"""
import sys,os,json,time,hashlib,types,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];UP=ROOT/'vendor/RT-Splatting'
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
sys.path.insert(0,str(UP));sys.path.insert(0,str(ROOT/'scripts'))
from rt_source_guard import patch_training_source, reject_checkpoint_resume
reject_checkpoint_resume(sys.argv[1:])
source=patch_training_source((UP/'train.py').read_text(encoding='utf-8'))
import numpy as np
import torch
from PIL import Image
from mask_loss import mask_terms
import scene  # upstream package initializes camera_utils; avoid circular import
from rade_initialization import install_adapter
install_adapter()
from utils import camera_utils
original_load=camera_utils.loadCam
def load_camera(args,id,info,scale):
    assert info.image_name!='cam00','cam00 must not be loaded by training'
    camera=original_load(args,id,info,scale)
    path=Path(args.source_path)/'mask_valid'/f'{info.image_name}.png'
    valid=np.asarray(Image.open(path).convert('L'))
    assert valid.shape==(camera.image_height,camera.image_width),'Use -r 1; do not resize pseudo masks'
    camera.mask_valid=torch.from_numpy(valid.copy()>127).unsqueeze(0)
    return camera
camera_utils.loadCam=load_camera
started=time.time();view_rng=np.random.default_rng(20261015)
def record(model,iteration,loss,path):
    if iteration%100:return
    dest=Path(path);value={'iteration':iteration,'loss':float(loss.detach()),'gaussians':len(model.get_xyz),'elapsed_seconds':time.time()-started,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated()}
    temp=dest/'status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(dest/'status.json')
    print('STATUS',json.dumps(value),flush=True)
def initial_hash(model,path):
    h=hashlib.sha256()
    modules=hashlib.sha256()
    for key,value in sorted(vars(model).items()):
        if isinstance(value,torch.Tensor):h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
        elif isinstance(value,torch.nn.Module):
            for name,t in sorted(value.state_dict().items()):
                h.update((key+name).encode());h.update(t.detach().cpu().numpy().tobytes())
                modules.update((key+name).encode());modules.update(t.detach().cpu().numpy().tobytes())
    Path(path,'initial_sha256.txt').write_text(h.hexdigest())
    Path(path,'shared_modules_sha256.txt').write_text(modules.hexdigest())
module=types.ModuleType('__main__');module.__file__=str(UP/'train.py')
module.__dict__.update(mask_terms=mask_terms,view_rng=view_rng,record=record,initial_hash=initial_hash)
sys.modules['__main__']=module
# Preserve caller-relative paths when switching to the upstream working directory.
for i, arg in enumerate(sys.argv[:-1]):
    if arg in ('-s','--source_path','-m','--model_path','--start_checkpoint'):
        sys.argv[i+1]=str(Path(sys.argv[i+1]).resolve())
os.chdir(UP)
exec(compile(source,str(UP/'train.py'),'exec'),module.__dict__)
