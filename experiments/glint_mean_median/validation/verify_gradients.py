"""GPU checks: median depth differentiability and unchanged forward outputs."""
import json,importlib.util,sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'GLINT'))
from glint.renderer import rasterize_glint_2dgs

device='cuda'
torch.manual_seed(0)
xyz=torch.tensor([[-.3,-.3,2.], [.3,-.3,2.2],[-.3,.3,2.1],[.3,.3,2.3]],device=device,requires_grad=True)
q=torch.tensor([[1.,.1,.2,.0],[1.,-.1,.15,.0],[1.,.1,-.1,.0],[1.,-.15,.1,.0]],device=device,requires_grad=True)
args=dict(means=xyz,quats=F.normalize(q,dim=-1),scales=torch.full((4,2),.6,device=device),
 opacities=torch.full((4,),.9,device=device),features=torch.full((4,3),.5,device=device),
 viewmat=torch.eye(4,device=device),K=torch.tensor([[40.,0,32],[0,40,32],[0,0,1]],device=device),width=64,height=64)
out=rasterize_glint_2dgs(**args)
spec=importlib.util.spec_from_file_location('baseline_renderer',str(Path(__file__).with_name('baseline_renderer.py')))
baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
ref=baseline.rasterize_glint_2dgs(**args)
for key in ('render','surf_depth','surf_normal','rend_normal','render_depth_expected','render_depth_median'):
 torch.testing.assert_close(out[key],ref[key],atol=0,rtol=0)
torch.testing.assert_close(out.surf_normal_expected,out.surf_normal,atol=0,rtol=0)
mask=(out.rend_alpha[0].detach()>.8)
assert mask.sum()>100
depth=out.render_depth_median[0][mask].mean()
depth_grad=torch.autograd.grad(depth,xyz,retain_graph=True)[0]
assert torch.isfinite(depth_grad).all() and depth_grad.abs().max()>1e-6
mean=1-F.cosine_similarity(out.rend_normal.unsqueeze(0),out.surf_normal_expected.unsqueeze(0),dim=1)
median=1-F.cosine_similarity(out.rend_normal.unsqueeze(0),out.surf_normal_median.unsqueeze(0),dim=1)
ml=median[0][mask].mean()
grads=torch.autograd.grad(ml,(xyz,q),retain_graph=True)
assert all(torch.isfinite(g).all() for g in grads)
assert sum(g.abs().sum() for g in grads)>1e-6
loss=(.4*mean+.6*median)[0][mask].mean()
g=torch.autograd.grad(loss,(xyz,q))
assert all(torch.isfinite(v).all() for v in g)
record=dict(forward_matches_baseline=True,median_depth_xyz_gradient_max=depth_grad.abs().max().item(),
 median_normal_geometry_gradient_l1=sum(v.abs().sum().item() for v in grads),
 expected_loss=mean[0][mask].mean().item(),median_loss=ml.item(),combined_loss=loss.item())
Path(__file__).with_name('gradient_checks.latest.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
