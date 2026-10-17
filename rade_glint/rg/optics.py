"""Differentiable reference optics. This is NOT GLINT's 2D OptiX tracer.

Secondary rays query filtered 3D ellipsoids at their Mahalanobis closest
approach, with front-to-back opacity compositing. O(rays * Gaussians), intended
for correctness tests and small sampled-ray experiments, not production speed.
The primary camera passes remain the unmodified IGS RaDe CUDA renderer.
"""
from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class Field:
    xyz: Tensor
    scales: Tensor
    rotation: Tensor  # local-to-world [N,3,3]
    opacity: Tensor  # already corrected for RaDe filter_3D
    colors: Tensor   # [N,3] or callable(view directions [R,N,3])


def quaternion_matrix(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1,3,3)


def trace(field, origins, directions, background=None, *, ray_chunk=32,
          near=1e-5, support_sigma=3.0, exclude_origin_support=True):
    """A discrete ellipsoid-splat reference, NOT exact volumetric integration.

    Sorting/support selection is piecewise differentiable. Geometry/material
    values inside the selected support retain autograd. Nothing is truncated
    to nearest K points. No point budgets silently change the scene.
    """
    if ray_chunk < 1 or support_sigma <= 0 or (field.scales <= 0).any():
        raise ValueError('Invalid trace parameters/scales')
    directions = F.normalize(directions, dim=-1)
    bg = origins.new_zeros(3) if background is None else background
    if len(field.xyz)==0:
        return {'rgb':bg.expand(len(origins),3), 'coverage':origins.new_zeros(len(origins)),
                'distance':origins.new_zeros(len(origins))}
    parts = []
    for start in range(0, len(origins), ray_chunk):
        o, d = origins[start:start+ray_chunk], directions[start:start+ray_chunk]
        delta = o[:,None,:] - field.xyz[None,:,:]
        # R columns are world-space axes, so row-vector local coordinates = v R.
        local_o = torch.einsum('rni,nij->rnj', delta, field.rotation) / field.scales[None]
        local_d = torch.einsum('ri,nij->rnj', d, field.rotation) / field.scales[None]
        denom = local_d.square().sum(-1).clamp_min(1e-12)
        distance = -(local_o*local_d).sum(-1) / denom
        closest = local_o + distance[...,None]*local_d
        power = closest.square().sum(-1)
        valid = (distance > near) & (power <= support_sigma**2)
        if exclude_origin_support:
            # Secondary rays must not immediately re-hit a splat whose support
            # contains their origin. Approximation, not a watertight surface ID.
            valid &= local_o.square().sum(-1)>support_sigma**2
        alpha = (field.opacity.reshape(1,-1) * torch.exp(-.5*power)).clamp(0,.99)
        alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
        order = torch.argsort(torch.where(valid, distance, torch.full_like(distance,float('inf'))), dim=-1)
        alpha = alpha.gather(1,order)
        z = distance.gather(1,order)
        trans = torch.cumprod(torch.cat((torch.ones_like(alpha[:,:1]),1-alpha),1),1)
        weights = trans[:,:-1]*alpha
        if callable(field.colors):
            # SH convention matches primary RaDe: direction from origin to primitive.
            colors = field.colors(F.normalize(field.xyz[None]-o[:,None], dim=-1))
        else:
            colors = field.colors[None].expand(len(o),-1,-1)
        colors = colors.gather(1,order[...,None].expand(-1,-1,3))
        rgb = (weights[...,None]*colors).sum(1) + trans[:,-1,None]*bg
        coverage = weights.sum(1)
        depth = (weights*z).sum(1)/coverage.clamp_min(1e-8)
        parts.append((rgb,coverage,depth))
    if not parts:
        return {'rgb': origins.new_zeros((0,3)), 'coverage': origins.new_zeros(0),
                'distance': origins.new_zeros(0)}
    return dict(zip(('rgb','coverage','distance'),
                    (torch.cat([part[i] for part in parts]) for i in range(3))))


def secondary_rays(origins, directions, distance, normals, epsilon):
    d = F.normalize(directions,dim=-1)
    n = F.normalize(normals,dim=-1)
    # Face normals towards the camera without changing reflection direction.
    n = torch.where((n*d).sum(-1,keepdim=True)>0,-n,n)
    point = origins + distance[...,None]*d
    reflected = F.normalize(d - 2*(d*n).sum(-1,keepdim=True)*n,dim=-1)
    return point + epsilon*d, d, point + epsilon*reflected, reflected


def trace_rgb_checkpointed(field, origins, directions, background=None, *, ray_chunk=16):
    """Recompute secondary-ray intermediates in backward to bound graph memory.

    Still O(rays * Gaussians) work; this is not an acceleration structure.
    """
    from torch.utils.checkpoint import checkpoint
    parts=[]
    for start in range(0,len(origins),ray_chunk):
        def query(o,d):
            return trace(field,o,d,background,ray_chunk=ray_chunk)['rgb']
        parts.append(checkpoint(query,origins[start:start+ray_chunk],
                                directions[start:start+ray_chunk],use_reentrant=False))
    return torch.cat(parts) if parts else origins.new_zeros((0,3))


def compose(surface_rgb, transmitted_rgb, reflected_rgb, coverage,
            transmission, specular, directions, normals, *, f0=.04):
    """GLINT-style diffuse/transmission/reflection balance at ONE interface hit.

    All colors are conditional-on-hit radiance. Coverage is applied once.
    Reflection is queried from the shared Transmission scene, not a third field.
    Thin-sheet, straight-through model: no refractive bending or glass thickness.
    """
    if not 0 <= f0 <= 1:
        raise ValueError('Invalid Fresnel F0')
    cosine = (F.normalize(normals,dim=-1)*-F.normalize(directions,dim=-1)).sum(-1).abs().clamp(0,1)
    fresnel = f0+(1-f0)*(1-cosine)**5
    r = specular.clamp(0,1)+(1-specular.clamp(0,1))*fresnel
    t = transmission.clamp(0,1)
    w_d, w_t, w_r = (1-r)*(1-t), (1-r)*t, r
    hit = w_d[...,None]*surface_rgb + w_t[...,None]*transmitted_rgb + w_r[...,None]*reflected_rgb
    a = coverage.clamp(0,1)[...,None]
    return {'rgb': a*hit+(1-a)*transmitted_rgb,
            'weights': torch.stack((w_d,w_t,w_r),-1)}
