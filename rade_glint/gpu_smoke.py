"""Required real RaDe CUDA forward/backward smoke test before N3DV training.

Synthetic geometry only; no dataset or test camera is read. This script has
not been executed when the server is unavailable. CPU tests are not a substitute.
"""
import argparse
import io
import json
import math
import numpy as np
import torch
from rg import rade
from rg.optics import trace_rgb_checkpointed,secondary_rays,compose


def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    rade.activate()
    from scene import GaussianModel
    from scene.cameras import Camera
    from arguments import OptimizationParams,PipelineParams
    from utils.graphics_utils import BasicPointCloud
    from gaussian_renderer import render
    parser=argparse.ArgumentParser(); op=OptimizationParams(parser); pp=PipelineParams(parser)
    args=parser.parse_args([]); opt,pipe=op.extract(args),pp.extract(args)
    Class=rade.make_gaussian_class(GaussianModel)
    i,t=Class(1,True),Class(1)
    coords=np.array([[x,y,2.] for x in [-.3,0.,.3] for y in [-.3,0.,.3]],dtype=np.float32)
    i.create_from_pcd(BasicPointCloud(coords,np.tile([.8,.1,.1],(9,1)).astype(np.float32),np.zeros_like(coords)),1.)
    i.training_setup(opt)
    camera=Camera(1,np.eye(3),np.zeros(3),math.pi/3,math.pi/3,
                  torch.zeros(3,32,32),None,'synthetic',0)
    i.compute_3D_filter(cameras=[camera]);i.init_material()
    rade.clone_initial_geometry(i,t,opt)
    with torch.no_grad():
        t._xyz[:,2]+=2.
        t._features_dc[:]=t._features_dc.new_tensor([-.2,1.,-.2]).reshape(1,1,3)
    t.compute_3D_filter(cameras=[camera])
    bg=torch.zeros(3,device='cuda')
    original=render(camera,i,pipe,bg,0.,require_coord=False,require_depth=True)
    mat,alpha=rade.render_material(camera,i,pipe,0.)
    mat.mean().backward()
    assert i._xyz.grad is None,'Material feature loss moved geometry'
    assert i.material.logits.grad is not None and i.material.logits.grad.abs().sum()>0
    i.material.optimizer.zero_grad(set_to_none=True)
    ys=torch.tensor([15,15,16,16],device='cuda');xs=torch.tensor([15,16,15,16],device='cuda')
    o,d,length=rade.camera_rays(camera,ys.float(),xs.float())
    n=torch.nn.functional.normalize(original['normal'][:,ys,xs].T,dim=-1)
    to,td,ro,rd=secondary_rays(o,d,original['expected_depth'][0,ys,xs].detach()*length,n.detach(),.001)
    f=rade.field_from_gaussians(t)
    behind=trace_rgb_checkpointed(f,to,td,ray_chunk=2)
    reflection=trace_rgb_checkpointed(f,ro,rd,ray_chunk=2)
    mat,alpha=rade.render_material(camera,i,pipe,0.)
    a=alpha[0,ys,xs]
    result=compose(original['render'][:,ys,xs].T/a[:,None].clamp_min(1e-6),behind,reflection,a,
                   mat[0,ys,xs],mat[1,ys,xs],d,n.detach())
    loss=(result['rgb']-result['rgb'].new_tensor([.1,.7,.2])).square().mean()
    loss.backward()
    for tensor in (i._xyz,t._xyz,i.material.logits):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    assert t._features_dc.grad.abs().sum()>0
    i.optimizer.step();t.optimizer.step();i.material.optimizer.step()
    buffer=io.BytesIO();torch.save({'i':i.checkpoint(),'t':t.checkpoint()},buffer);buffer.seek(0)
    restored=torch.load(buffer,weights_only=False)
    other=Class(1,True);other.load_checkpoint(restored['i'],opt)
    with torch.no_grad():
        a=render(camera,i,pipe,bg,0.,require_coord=False,require_depth=True)['render']
        b=render(camera,other,pipe,bg,0.,require_coord=False,require_depth=True)['render']
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
    torch.cuda.synchronize()
    print(json.dumps({'status':'passed','loss':float(loss.detach()),'peak_memory_bytes':torch.cuda.max_memory_allocated(),
                      'gpu':torch.cuda.get_device_name(),'material_geometry_isolation':True,'checkpoint_render_parity':True},indent=2))


if __name__=='__main__': main()
