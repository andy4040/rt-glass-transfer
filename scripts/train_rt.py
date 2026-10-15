"""Pinned upstream RT trainer with explicit pseudo-mask validity adaptation.

No GLINT module/model/loss is imported. Rendering and representation are upstream.
"""
import sys,os,json,time,hashlib,types,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];UP=ROOT/'vendor/RT-Splatting'
sys.path.insert(0,str(UP));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from PIL import Image
from mask_loss import mask_terms
import scene  # upstream package initializes camera_utils; avoid circular import
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
    for key,value in sorted(vars(model).items()):
        if isinstance(value,torch.Tensor):h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
        elif isinstance(value,torch.nn.Module):
            for name,t in sorted(value.state_dict().items()):h.update((key+name).encode());h.update(t.detach().cpu().numpy().tobytes())
    Path(path,'initial_sha256.txt').write_text(h.hexdigest())
source=(UP/'train.py').read_text()
start=source.index('        if opt.mask_loss_weight > 0 and iteration >= opt.mask_loss_from_iter:')
end=source.index('        total_loss = loss',start)
source=source[:start]+'''        terms = mask_terms(surface_opacity, transmissivity, render_scat, gt_transparent_mask, viewpoint_cam.mask_valid.cuda(), opt, iteration)
        for name, term in terms.items():
            loss += term
            loss_dict[name] = float(term.detach())

'''+source[end:]
source=source.replace('data_idx = np.random.randint(len(viewpoint_stack))','data_idx = int(view_rng.integers(len(viewpoint_stack)))')
source=source.replace('    gaussians.training_setup(opt)','    gaussians.training_setup(opt)\n    initial_hash(gaussians, dataset.model_path)',1)
source=source.replace('            iter_end.synchronize()','            iter_end.synchronize()\n            record(gaussians, iteration, total_loss, dataset.model_path)',1)
# Built-in masked-PSNR and train-view reporting are not used as held-out evaluation.
source=source.replace('            training_report(tb_writer, tb_executor, opt, iteration, loss, loss_dict, iter_start.elapsed_time(iter_end), testing_iterations, scene, partial(render, pipe=pipe, bg_color=bg))','            pass  # independent cam00 evaluator runs outside training')
module=types.ModuleType('__main__');module.__file__=str(UP/'train.py')
module.__dict__.update(mask_terms=mask_terms,view_rng=view_rng,record=record,initial_hash=initial_hash)
sys.modules['__main__']=module
os.chdir(UP)
exec(compile(source,str(UP/'train.py'),'exec'),module.__dict__)
