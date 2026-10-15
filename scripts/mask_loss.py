"""The three upstream mask terms, restricted to reliable pseudo-label pixels."""
import torch
def bce(pred,target):
    pred=pred.clamp(1e-6,1-1e-6)
    return -(target*pred.log()+(1-target)*(1-pred).log()).mean()
def mask_terms(surface_opacity,transmissivity,scatter,mask,valid,opt,iteration):
    surface_opacity=surface_opacity.reshape(mask.shape[-2:])
    mask=mask.bool();valid=valid.bool();positive=mask&valid;negative=(~mask)&valid
    zero=surface_opacity.sum()*0;terms={}
    if opt.mask_loss_weight>0 and iteration>=opt.mask_loss_from_iter:
        terms['mask']=opt.mask_loss_weight*bce(1-surface_opacity[valid.squeeze(0)],mask.squeeze(0)[valid.squeeze(0)].float()) if valid.any() else zero
    if iteration>=opt.init_until_iter:
        terms['transmissivity']=opt.transmissivity_loss_weight*bce(transmissivity[negative],0) if negative.any() else zero
    if opt.consistency_loss_weight>0 and iteration>=opt.init_until_iter:
        if positive.any():
            t=transmissivity[positive];s=scatter[:,positive.squeeze(0)]
            terms['consistency']=opt.consistency_loss_weight*((t-t.mean()).square().sum()+(s-s.mean(-1,keepdim=True)).square().sum(-1).mean())
        else:terms['consistency']=zero
    return terms
