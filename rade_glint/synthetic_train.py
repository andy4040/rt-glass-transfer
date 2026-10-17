"""CPU end-to-end optics smoke test, NOT an N3DV or original-RaDe benchmark.

One checkpoint contains two trainable Gaussian fields plus Interface material.
Known synthetic depths supervise geometry to test the gradient/optimizer path;
this does not establish unsupervised glass detection or real-scene accuracy.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from torch import nn
from rg.optics import Field,trace,compose,secondary_rays


class JointScene(nn.Module):
    def __init__(self,z_i=2.6,z_t=4.3,tau=.2):
        super().__init__()
        self.interface_z=nn.Parameter(torch.tensor(float(z_i)))
        self.transmission_z=nn.Parameter(torch.tensor(float(z_t)))
        self.transmission_logit=nn.Parameter(torch.logit(torch.tensor(float(tau))))

    def field(self,z,color):
        xyz=torch.stack((z*0,z*0,z))[None]
        return Field(xyz,z.new_tensor([[2.,2.,.05]]),torch.eye(3)[None],z.new_tensor([.97]),z.new_tensor([color]))

    def forward(self):
        origins=torch.tensor([[-.25,0.,0.],[0.,0.,0.],[.25,0.,0.]])
        dirs=torch.tensor([[0.,0.,1.]]*3); normal=-dirs
        i=self.field(self.interface_z,[.8,.1,.1])
        t=self.field(self.transmission_z,[.1,.8,.3])
        first=trace(i,origins,dirs)
        direct=trace(t,origins,dirs)
        to,td,ro,rd=secondary_rays(origins,dirs,first['distance'],normal,.001)
        behind=trace(t,to,td)
        reflected=trace(t,ro,rd)
        result=compose(first['rgb']/first['coverage'][:,None].clamp_min(1e-6),
                       behind['rgb'],reflected['rgb'],first['coverage'],
                       self.transmission_logit.sigmoid().expand(3),torch.zeros(3),dirs,normal)
        return result['rgb'],first['distance'],direct['distance']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/synthetic'))
    p.add_argument('--steps',type=int,default=250)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.manual_seed(0)
    truth=JointScene(2.,5.,.85)
    with torch.no_grad(): target,di,dt=truth()
    model=JointScene(); opt=torch.optim.Adam(model.parameters(),lr=.04)
    def metrics():
        with torch.no_grad():
            pred,pi,pt=model()
            return dict(rgb_mse=float((pred-target).square().mean()),
                        interface_depth_error=float((pi-di).abs().mean()),
                        transmission_depth_error=float((pt-dt).abs().mean()),
                        transmission=float(model.transmission_logit.sigmoid()))
    initial=metrics(); started=time.perf_counter()
    for step in range(args.steps):
        pred,pi,pt=model()
        loss=(pred-target).square().mean()+.1*((pi-di).square().mean()+(pt-dt).square().mean())
        loss.backward();opt.step();opt.zero_grad(set_to_none=True)
    final=metrics()
    checkpoint=args.output/'joint_model.pt'
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=args.steps),checkpoint)
    restored=JointScene(); restored.load_state_dict(torch.load(checkpoint,weights_only=True)['model'])
    with torch.no_grad(): torch.testing.assert_close(restored()[0],model()[0],rtol=0,atol=0)
    report=dict(scope='synthetic CPU, known depth supervision; NOT N3DV/CUDA RaDe validation',
                steps=args.steps,seconds=time.perf_counter()-started,initial=initial,final=final,
                checkpoint_roundtrip_exact=True,torch=torch.__version__)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if final['rgb_mse']>=initial['rgb_mse']*.05 or max(final['interface_depth_error'],final['transmission_depth_error'])>.02:
        raise RuntimeError('Synthetic convergence test failed')


if __name__=='__main__': main()
