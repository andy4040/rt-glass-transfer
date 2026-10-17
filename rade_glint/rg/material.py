"""Material optimizer kept separate from unmodified RaDe geometry optimizer."""
import torch
import copy
from torch import nn


class Material:
    def __init__(self,count,device,lr=.002):
        self.lr=lr
        self.logits=nn.Parameter(torch.full((count,2),-3.,device=device))
        self.optimizer=torch.optim.Adam([self.logits],lr=lr)

    @property
    def values(self):
        return self.logits.sigmoid()

    def remap(self,indices,inherited=None):
        """Keep arbitrary rows; optionally append children inheriting parent material.

        Existing moments survive. Children inherit values but start with zero
        Adam moments. No nearest-neighbor material reassignment.
        """
        old=self.logits
        state=self.optimizer.state.get(old,{})
        rows=[old.detach()[indices]]
        if inherited is not None:
            rows.append(old.detach()[inherited])
        self.logits=nn.Parameter(torch.cat(rows))
        self.optimizer.param_groups[0]['params']=[self.logits]
        self.optimizer.state.pop(old,None)
        if state:
            new={}
            for key,value in state.items():
                if isinstance(value,torch.Tensor) and value.shape==old.shape:
                    tensors=[value[indices]]
                    if inherited is not None:
                        tensors.append(torch.zeros_like(value[inherited]))
                    new[key]=torch.cat(tensors)
                else:
                    new[key]=value.clone() if isinstance(value,torch.Tensor) else value
            self.optimizer.state[self.logits]=new

    def capture(self):
        return {'logits':self.logits.detach().clone(),'optimizer':copy.deepcopy(self.optimizer.state_dict()),'lr':self.lr}

    @classmethod
    def restore(cls,state,device):
        result=cls(len(state['logits']),device,state['lr'])
        with torch.no_grad(): result.logits.copy_(state['logits'].to(device))
        result.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        return result
