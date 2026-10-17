import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rg.optics import Field,trace,compose,secondary_rays,quaternion_matrix,trace_rgb_checkpointed
from rg.losses import glass_targets,material_loss,scale_shift_depth
from rg.material import Material
from rg.data import check_names


def field(z=(2.,4.),opacity=(.8,.9),colors=((1.,0.,0.),(0.,1.,0.))):
    n=len(z)
    xyz=torch.tensor([[0.,0.,v] for v in z],dtype=torch.double,requires_grad=True)
    scales=torch.tensor([[.5,.5,.1]]*n,dtype=torch.double,requires_grad=True)
    op=torch.tensor(opacity,dtype=torch.double,requires_grad=True)
    rgb=torch.tensor(colors,dtype=torch.double,requires_grad=True)
    return Field(xyz,scales,torch.eye(3,dtype=torch.double).repeat(n,1,1),op,rgb)


class Core(unittest.TestCase):
    def test_ray_order_and_coverage(self):
        f=field(); o=torch.zeros(1,3,dtype=torch.double); d=torch.tensor([[0.,0.,1.]],dtype=torch.double)
        result=trace(f,o,d)
        torch.testing.assert_close(result['rgb'],torch.tensor([[.8,.18,0.]],dtype=torch.double))
        torch.testing.assert_close(result['coverage'],torch.tensor([.98],dtype=torch.double))
        torch.testing.assert_close(result['distance'],torch.tensor([(2*.8+4*.18)/.98],dtype=torch.double))

    def test_forward_origin_excludes_front(self):
        result=trace(field(),torch.tensor([[0.,0.,2.2]],dtype=torch.double),torch.tensor([[0.,0.,1.]],dtype=torch.double))
        torch.testing.assert_close(result['rgb'],torch.tensor([[0.,.9,0.]],dtype=torch.double))

    def test_chunk_invariance(self):
        f=field(); o=torch.tensor([[.1,0.,0.],[.2,.1,0.],[-.1,0.,0.]],dtype=torch.double)
        d=torch.tensor([[0.,0.,1.]]*3,dtype=torch.double)
        a,b=trace(f,o,d,ray_chunk=1),trace(f,o,d,ray_chunk=3)
        for key in a: torch.testing.assert_close(a[key],b[key])

    def test_reference_geometry_gradcheck(self):
        f=field(); o=torch.tensor([[.07,.03,0.]],dtype=torch.double); d=torch.tensor([[.01,0.,1.]],dtype=torch.double)
        def fn(x,s,op,c): return trace(Field(x,s,f.rotation,op,c),o,d)['rgb']
        self.assertTrue(torch.autograd.gradcheck(fn,(f.xyz,f.scales,f.opacity,f.colors),eps=1e-6,atol=1e-4))

    def test_checkpointed_trace_gradient_parity(self):
        a,b=field(),field()
        o=torch.tensor([[.1,0.,0.],[.2,.1,0.]],dtype=torch.double)
        d=torch.tensor([[0.,0.,1.]]*2,dtype=torch.double)
        result=trace(a,o,d,ray_chunk=1)['rgb']
        other=trace_rgb_checkpointed(b,o,d,ray_chunk=1)
        result.square().sum().backward();other.square().sum().backward()
        torch.testing.assert_close(result,other)
        for name in ('xyz','scales','opacity','colors'):
            torch.testing.assert_close(getattr(a,name).grad,getattr(b,name).grad)

    def test_empty_and_missed_rays(self):
        f=field(); o=torch.tensor([[20.,0.,0.]],dtype=torch.double); d=torch.tensor([[0.,0.,1.]],dtype=torch.double)
        result=trace(f,o,d)
        self.assertEqual(result['coverage'].item(),0.)
        self.assertTrue(torch.isfinite(result['distance']).all())
        empty=Field(f.xyz[:0],f.scales[:0],f.rotation[:0],f.opacity[:0],f.colors[:0])
        bg=torch.tensor([.2,.3,.4],dtype=torch.double)
        torch.testing.assert_close(trace(empty,o,d,bg)['rgb'],bg[None])

    def test_energy_opaque_and_transparent_limits(self):
        s=torch.tensor([[1.,0.,0.]]); t=torch.tensor([[0.,1.,0.]]); r=torch.tensor([[0.,0.,1.]])
        d=torch.tensor([[0.,0.,1.]]); n=-d; a=torch.ones(1)
        for tau,expected in [(0.,s),(1.,t)]:
            result=compose(s,t,r,a,torch.tensor([tau]),torch.zeros(1),d,n,f0=0.)
            torch.testing.assert_close(result['rgb'],expected)
            torch.testing.assert_close(result['weights'].sum(-1),a)
        result=compose(s,t,r,a,torch.ones(1),torch.zeros(1),d,n)
        torch.testing.assert_close(result['rgb'],.96*t+.04*r)

    def test_material_gradient_separate_from_coverage(self):
        tau=torch.tensor([.3],requires_grad=True)
        a=torch.tensor([.9],requires_grad=True)
        rgb=compose(torch.tensor([[1.,0.,0.]]),torch.tensor([[0.,1.,0.]]),torch.zeros(1,3),
                    a,tau,torch.zeros(1),torch.tensor([[0.,0.,1.]]),torch.tensor([[0.,0.,-1.]]))['rgb']
        rgb[0,1].backward()
        self.assertGreater(tau.grad.item(),0.)
        self.assertGreater(a.grad.abs().item(),0.)

    def test_reflected_direction(self):
        o=torch.zeros(1,3); d=torch.tensor([[0.,0.,1.]]); n=-d
        to,td,ro,rd=secondary_rays(o,d,torch.tensor([2.]),n,.01)
        torch.testing.assert_close(rd,-d)
        torch.testing.assert_close(to,torch.tensor([[0.,0.,2.01]]))
        torch.testing.assert_close(ro,torch.tensor([[0.,0.,1.99]]))

    def test_depth_gap_not_sufficient_and_unknown_no_grad(self):
        di=torch.tensor([2.,2.,2.,2.],requires_grad=True)
        dt=torch.tensor([5.,2.,1.,5.],requires_grad=True)
        target,valid=glass_targets(di,dt,torch.ones(4),torch.ones(4),
                                  torch.tensor([.01,.3,.01,.01]),torch.tensor([.8,.8,.8,.05]),torch.ones(4,dtype=torch.bool))
        self.assertEqual(valid.tolist(),[True,True,False,False])
        pred=torch.full((4,),.5,requires_grad=True)
        material_loss(pred,target,valid).backward()
        self.assertEqual(pred.grad[2:].abs().sum().item(),0.)
        self.assertIsNone(di.grad); self.assertIsNone(dt.grad)

    def test_material_densification_pruning_and_resume(self):
        m=Material(3,'cpu')
        m.values.sum().backward(); m.optimizer.step(); m.optimizer.zero_grad()
        old=m.logits.detach().clone(); moments=m.optimizer.state[m.logits]['exp_avg'].clone()
        m.remap(torch.tensor([0,1,2]),torch.tensor([2,0,2]))
        torch.testing.assert_close(m.logits.detach()[3:],old[[2,0,2]])
        torch.testing.assert_close(m.optimizer.state[m.logits]['exp_avg'][:3],moments)
        self.assertEqual(m.optimizer.state[m.logits]['exp_avg'][3:].abs().sum().item(),0.)
        m.remap(torch.tensor([1,3,5]))
        restored=Material.restore(m.capture(),'cpu')
        for obj in (m,restored):
            obj.values.square().sum().backward();obj.optimizer.step();obj.optimizer.zero_grad()
        torch.testing.assert_close(m.logits,restored.logits)

    def test_relative_depth_scale_shift_and_degenerate(self):
        q=torch.linspace(.1,1,20); p=(3*q+5).requires_grad_()
        self.assertLess(scale_shift_depth(p,q,torch.ones(20,dtype=torch.bool)).item(),1e-6)
        value=scale_shift_depth(p,torch.ones(20),torch.ones(20,dtype=torch.bool))
        value.backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_quaternion_and_holdout(self):
        torch.testing.assert_close(quaternion_matrix(torch.tensor([[1.,0.,0.,0.]])),torch.eye(3)[None])
        self.assertEqual(check_names(['cam01','cam12']),['cam01','cam12'])
        with self.assertRaises(ValueError): check_names(['cam01','cam00'])


if __name__=='__main__': unittest.main()
