"""Adapters around byte-preserved official IGS/RaDe sources."""
from pathlib import Path
import hashlib
import sys
import math
import torch
from torch import nn
from .material import Material
from .optics import Field, quaternion_matrix

ROOT=Path(__file__).resolve().parents[1]
IGS=ROOT/'vendor/IGS'
RADE=IGS/'submodules/RaDe-GS'
UPSTREAM_COMMIT='f4b1880c5230834e276c8f8793ffaac364210a37'


def activate():
    if not (RADE/'train.py').is_file():
        raise FileNotFoundError('Clone official IGS into '+str(IGS))
    sys.path.insert(0,str(RADE))


def make_gaussian_class(base):
    """Inherit upstream geometry/densification, track exact material parent IDs."""
    class Gaussian(base):
        def __init__(self,degree,with_material=False):
            super().__init__(degree)
            self.material=None
            self.with_material=with_material
            self._parents=None

        def init_material(self):
            if self.with_material:
                self.material=Material(len(self._xyz),self._xyz.device)

        def densify_and_clone(self,grads,threshold,abs_grads,abs_threshold,extent):
            selected=(torch.norm(grads,dim=-1)>=threshold)|(torch.norm(abs_grads,dim=-1)>=abs_threshold)
            selected &= self.get_scaling.max(1).values<=self.percent_dense*extent
            self._parents=selected.nonzero().flatten()
            try: return super().densify_and_clone(grads,threshold,abs_grads,abs_threshold,extent)
            finally: self._parents=None

        def densify_and_split(self,grads,threshold,abs_grads,abs_threshold,extent,N=2):
            padded=self._xyz.new_zeros(len(self._xyz))
            padded_abs=padded.clone()
            padded[:len(grads)]=grads.reshape(-1)
            padded_abs[:len(abs_grads)]=abs_grads.reshape(-1)
            selected=(padded>=threshold)|(padded_abs>=abs_threshold)
            selected &= self.get_scaling.max(1).values>self.percent_dense*extent
            self._parents=selected.nonzero().flatten().repeat(N)
            try: return super().densify_and_split(grads,threshold,abs_grads,abs_threshold,extent,N)
            finally: self._parents=None

        def densification_postfix(self,*args):
            count=len(self._xyz)
            added=len(args[0])
            if self.material is not None and (self._parents is None or len(self._parents)!=added):
                raise RuntimeError('Missing exact parent mapping for material inheritance')
            super().densification_postfix(*args)
            if self.material is not None:
                self.material.remap(torch.arange(count,device=self._xyz.device),self._parents)

        def prune_points(self,mask):
            keep=(~mask).nonzero().flatten()
            super().prune_points(mask)
            if self.material is not None: self.material.remap(keep)

        def checkpoint(self):
            # Upstream capture omits filter_3D and absolute densification stats.
            extras={name:getattr(self,name).detach().clone() for name in
                    ('filter_3D','xyz_gradient_accum_abs','xyz_gradient_accum_abs_max')}
            return {'core':super().capture(),'extras':extras,
                    'material':None if self.material is None else self.material.capture()}

        def load_checkpoint(self,state,opt):
            super().restore(state['core'],opt)
            for name,value in state['extras'].items(): setattr(self,name,value.to(self._xyz.device))
            if state['material'] is not None:
                self.material=Material.restore(state['material'],self._xyz.device)
    return Gaussian


def clone_initial_geometry(source,target,opt):
    for key in ('_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity'):
        setattr(target,key,nn.Parameter(getattr(source,key).detach().clone()))
    target.spatial_lr_scale=source.spatial_lr_scale
    target.active_sh_degree=source.active_sh_degree
    target.training_setup(opt)
    target.filter_3D=source.filter_3D.detach().clone()
    target.max_radii2D=source.max_radii2D.clone()


def render_material(camera,gaussians,pipe,kernel_size):
    """Three-channel feature pass. Material gradients only; geometry detached."""
    from diff_gaussian_rasterization_rade import GaussianRasterizer, GaussianRasterizationSettings
    scales,opacity=gaussians.get_scaling_n_opacity_with_3D_filter
    values=gaussians.material.values
    colors=torch.cat((values,torch.zeros_like(values[:,:1])),1).contiguous()
    raster=GaussianRasterizer(GaussianRasterizationSettings(
        image_height=camera.image_height,image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx/2),tanfovy=math.tan(camera.FoVy/2),
        kernel_size=kernel_size,bg=values.new_zeros(3),scale_modifier=1.,
        viewmatrix=camera.world_view_transform,projmatrix=camera.full_proj_transform,
        sh_degree=0,campos=camera.camera_center,prefiltered=False,
        require_coord=False,require_depth=False,debug=pipe.debug))
    result=raster(means3D=gaussians.get_xyz.detach(),means2D=torch.zeros_like(gaussians.get_xyz),
                  shs=None,colors_precomp=colors,opacities=opacity.detach(),
                  scales=scales.detach(),rotations=gaussians.get_rotation.detach(),cov3D_precomp=None)
    alpha=result[6].detach()
    return (result[0][:2]/alpha.clamp_min(1e-6)).clamp(0,1),alpha


def field_from_gaussians(gaussians):
    from utils.sh_utils import eval_sh
    scales,opacity=gaussians.get_scaling_n_opacity_with_3D_filter
    sh=gaussians.get_features.transpose(1,2)
    def colors(directions):
        features=sh[None].expand(len(directions),-1,-1,-1)
        return (eval_sh(gaussians.active_sh_degree,features,directions)+.5).clamp_min(0)
    return Field(gaussians.get_xyz,scales,quaternion_matrix(gaussians.get_rotation),opacity,colors)


def camera_rays(camera,ys,xs):
    # RaDe NDC -> pixel maps optical axis to (W-1)/2, consistent with integer pixels.
    fx=camera.image_width/(2*math.tan(camera.FoVx/2))
    fy=camera.image_height/(2*math.tan(camera.FoVy/2))
    dcam=torch.stack(((xs+.5-camera.image_width/2)/fx,
                      (ys+.5-camera.image_height/2)/fy,torch.ones_like(xs)),1)
    c2w=camera.world_view_transform.T.inverse()
    world=dcam @ c2w[:3,:3].T
    length=world.norm(dim=-1)
    return c2w[:3,3].expand_as(world),world/length[:,None],length


def source_hashes():
    return {p.relative_to(IGS).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (RADE/'train.py',RADE/'scene/gaussian_model.py',RADE/'gaussian_renderer/__init__.py')}
