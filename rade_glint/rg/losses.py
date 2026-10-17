"""Independent I/T depth cues; never derive the label from optical depth."""
import torch
import torch.nn.functional as F


def masked_mean(value, mask):
    mask = mask.to(value.dtype)
    return (torch.where(mask>0,value,torch.zeros_like(value))*mask).sum()/mask.sum().clamp_min(1)


def glass_targets(di, dt, ai, at, albedo, basecolor, valid, *,
                  relative_gap=.01, albedo_low=.05, albedo_high=.10,
                  basecolor_high=.25, coverage_min=.7):
    """Fixed heuristic pseudo-labels, NOT calibrated glass probabilities.

    Independent direct I/T depths are detached. Positive gap is signed: a
    Transmission in front of Interface is invalid, not evidence of glass.
    A missing/uncertain cue is unknown rather than an opaque label.
    """
    with torch.no_grad():
        scale = torch.maximum(di.abs(),dt.abs()).clamp_min(1e-6)
        gap = (dt-di)/scale
        usable = valid.bool() & (di>0) & (dt>0) & (ai>coverage_min) & (at>coverage_min)
        usable &= torch.isfinite(di) & torch.isfinite(dt) & torch.isfinite(albedo) & torch.isfinite(basecolor)
        positive = usable & (gap>relative_gap) & (albedo<albedo_low) & (basecolor>basecolor_high)
        negative = usable & (gap.abs()<relative_gap*.5) & (albedo>albedo_high)
        return positive.to(di.dtype), positive|negative


def material_loss(prediction,target,valid):
    loss = F.binary_cross_entropy(prediction.clamp(1e-6,1-1e-6),target,reduction='none')
    return masked_mean(loss,valid)


def scale_shift_depth(prediction, prior, valid):
    """Scale/shift-invariant residual; never interprets prior values as metres."""
    finite = valid.bool() & torch.isfinite(prediction) & torch.isfinite(prior)
    p, q = prediction[finite], prior[finite]
    if p.numel()<16 or q.detach().var(unbiased=False)<1e-10:
        return prediction.sum()*0
    # Fit detached target alignment. It cannot collapse predicted variance by
    # differentiating through the fitted scale.
    with torch.no_grad():
        qc = q-q.mean()
        scale = (qc*(p.detach()-p.detach().mean())).sum()/qc.square().sum().clamp_min(1e-8)
        if scale <= 0:
            return prediction.sum()*0
        target = scale*q+(p.detach().mean()-scale*q.mean())
        norm = p.detach().abs().median().clamp_min(1e-4)
    return ((p-target).abs()/norm).mean()
