from types import SimpleNamespace
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from mask_loss import mask_terms
def test_unknown_has_no_mask_gradient():
    opacity=torch.full((2,2),.5,requires_grad=True)
    trans=torch.full((1,2,2),.5,requires_grad=True)
    scatter=torch.ones((3,2,2),requires_grad=True)
    mask=torch.tensor([[[1,0],[1,0]]],dtype=torch.bool)
    valid=torch.tensor([[[1,1],[0,0]]],dtype=torch.bool)
    opt=SimpleNamespace(mask_loss_weight=.01,mask_loss_from_iter=0,init_until_iter=0,transmissivity_loss_weight=.01,consistency_loss_weight=.01)
    sum(mask_terms(opacity,trans,scatter,mask,valid,opt,1).values()).backward()
    assert opacity.grad[1].abs().sum()==0 and trans.grad[:,1].abs().sum()==0
    for v in (valid*False,valid*True):
        terms=mask_terms(opacity,trans,scatter,mask*False,v,opt,1)
        assert all(torch.isfinite(x) for x in terms.values())
