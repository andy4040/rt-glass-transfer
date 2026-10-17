"""Joint I/T training using original RaDe primary rendering and geometry updates.

Secondary transport is the slow differentiable 3D reference implementation.
Full-frame direct I/T passes provide independent depth cues. Final transport
RGB is trained on sampled patches to keep the reference tracer bounded.
"""
import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from . import rade
from .data import validate_dataset,load_prior
from .losses import glass_targets,material_loss,scale_shift_depth,masked_mean
from .optics import trace_rgb_checkpointed,secondary_rays,compose


DEFAULTS=dict(iterations=30000,warmup=6000,prior_start=6001,
              checkpoint_every=1000,patch_size=16,ray_chunk=16,seed=0,
              normal_prior_weight=.3,depth_prior_weight=.1,material_weight=.01,
              opaque_alignment_weight=.05,transmission_direct_weight=1.,
              relative_gap=.01,normal_agreement=.5,f0=.04,
              ray_epsilon_extent=1e-4)


def normal_loss(camera,package):
    from utils.graphics_utils import depth_double_to_normal
    normals=depth_double_to_normal(camera,package['expected_depth'],package['median_depth'])
    error=1-(package['normal'].unsqueeze(0)*normals).sum(1)
    return .4*error[0].mean()+.6*error[1].mean()


def photo_loss(prediction,target,opt):
    from utils.loss_utils import l1_loss,ssim
    return (1-opt.lambda_dssim)*l1_loss(prediction,target)+opt.lambda_dssim*(1-ssim(prediction,target.unsqueeze(0)))


def atomic_torch_save(value,path):
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def save_checkpoint(path,step,interface,transmission,config,opt,rng,camera_stack,elapsed,provenance):
    atomic_torch_save(dict(schema=1,step=step,interface=interface.checkpoint(),
                          transmission=transmission.checkpoint(),config=config,
                          optimization=vars(opt),view_rng=rng.getstate(),camera_stack=camera_stack,
                          torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                          numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                          elapsed_seconds=elapsed,provenance=provenance),path)


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError('N3DV training needs CUDA + IGS RaDe extensions. CPU is supported only for reference tests.')
    config=dict(DEFAULTS)
    if args.config: config.update(json.loads(args.config.read_text()))
    unknown=set(config)-set(DEFAULTS)
    if unknown: raise ValueError('Unknown config keys: '+str(unknown))
    if not 0<=config['warmup']<config['iterations'] or config['patch_size']<11:
        raise ValueError('Invalid stage schedule/patch size (SSIM needs >=11)')
    names,prior_manifest=validate_dataset(args.source,args.priors)
    commit=subprocess.run(['git','-C',str(rade.IGS),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    dirty=subprocess.run(['git','-C',str(rade.IGS),'status','--porcelain'],capture_output=True,text=True,check=True).stdout
    if commit!=rade.UPSTREAM_COMMIT or dirty:
        raise ValueError('IGS source is not the pinned, unmodified checkout')
    provenance=dict(igs_commit=rade.UPSTREAM_COMMIT,source_hashes=rade.source_hashes(),
                    priors_manifest_sha256=hashlib.sha256((args.priors/'manifest.json').read_bytes()).hexdigest(),
                    calibration_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        (args.source/'sparse/0/cameras.txt',args.source/'sparse/0/images.txt')},
                    train_cameras=names,cam00_used=False,frame=0,
                    prior_array_sha256={name:hashlib.sha256((args.priors/(name+'.npz')).read_bytes()).hexdigest() for name in names},
                    secondary_renderer='reference_3d_ellipsoid; no BVH; thin straight transmission',
                    reflection_scene='same Transmission field; no separate Reflection Gaussian set')
    rade.activate()
    from arguments import ModelParams,OptimizationParams,PipelineParams
    from scene import Scene,GaussianModel
    from gaussian_renderer import render
    p=argparse.ArgumentParser()
    mp,op,pp=ModelParams(p),OptimizationParams(p),PipelineParams(p)
    default=p.parse_args([])
    dataset,opt,pipe=mp.extract(default),op.extract(default),pp.extract(default)
    dataset.source_path=str(args.source.resolve()); dataset.model_path=str(args.output.resolve())
    dataset.resolution=1; dataset.loader='colmap'; dataset.eval=False
    opt.iterations=config['iterations']
    opt.position_lr_max_steps=config['iterations']
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    saved=None
    if args.resume:
        saved=torch.load(args.resume,map_location='cuda',weights_only=False)
        if saved['config']!=config or saved['provenance']!=provenance:
            raise ValueError('Resume configuration/data/source hashes differ')
    elif args.output.exists():
        raise FileExistsError('Refusing to overwrite run: '+str(args.output))
    args.output.mkdir(parents=True,exist_ok=bool(args.resume))
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    Class=rade.make_gaussian_class(GaussianModel)
    interface=Class(dataset.sh_degree,with_material=True)
    scene=Scene(dataset,interface,shuffle=False)
    cameras=scene.getTrainCameras()
    if sorted(c.image_name for c in cameras)!=sorted(names) or scene.getTestCameras():
        raise ValueError('Scene split does not match train-only manifest')
    interface.training_setup(opt)
    interface.compute_3D_filter(cameras=cameras)
    interface.init_material()
    transmission=Class(dataset.sh_degree)
    rade.clone_initial_geometry(interface,transmission,opt)
    rng=random.Random(config['seed']); stack=[]; first=1; elapsed_before=0.
    if saved:
        interface.load_checkpoint(saved['interface'],opt)
        transmission.load_checkpoint(saved['transmission'],opt)
        if saved['optimization']!=vars(opt): raise ValueError('Optimizer configuration changed')
        rng.setstate(saved['view_rng']); stack=saved['camera_stack']; first=saved['step']+1
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
        elapsed_before=saved['elapsed_seconds']
        del saved
    prior_cache={}
    background=torch.zeros(3,device='cuda')
    started=time.perf_counter()
    status_path=args.output/'status.json'
    for step in range(first,config['iterations']+1):
        joint=step>config['warmup']
        if not stack: stack=list(range(len(cameras))); rng.shuffle(stack)
        camera=cameras[stack.pop()]
        if camera.image_name not in prior_cache:
            prior_cache[camera.image_name]=load_prior(args.priors,camera.image_name,'cuda')
        prior=prior_cache[camera.image_name]
        active=[interface,transmission] if joint else [interface]
        for field in active:
            field.update_learning_rate(step)
            if step%1000==0: field.oneupSHdegree()
        pkg_i=render(camera,interface,pipe,background,dataset.kernel_size,require_coord=False,require_depth=True)
        gt=camera.original_image.cuda()
        losses={}
        if step>=opt.regularization_from_iter:
            losses['normal_interface']=opt.lambda_depth_normal*normal_loss(camera,pkg_i)
        normal_i=F.normalize(pkg_i['normal'].permute(1,2,0),dim=-1)
        agreement=(normal_i*prior['normal']).sum(-1)
        prior_valid=prior['valid'] & (pkg_i['mask'][0].detach()>.7) & (agreement.detach()>config['normal_agreement'])
        if step>=config['prior_start']:
            losses['prior_normal']=config['normal_prior_weight']*masked_mean(1-agreement,prior_valid)
            losses['prior_depth']=config['depth_prior_weight']*scale_shift_depth(pkg_i['expected_depth'][0],prior['depth'],prior_valid)
        diagnostics={}
        pkg_t=None
        if not joint:
            losses['rgb_interface']=photo_loss(pkg_i['render'],gt,opt)
        else:
            pkg_t=render(camera,transmission,pipe,background,dataset.kernel_size,require_coord=False,require_depth=True)
            losses['rgb_transmission_direct']=config['transmission_direct_weight']*photo_loss(pkg_t['render'],gt,opt)
            if step>=opt.regularization_from_iter:
                losses['normal_transmission']=opt.lambda_depth_normal*normal_loss(camera,pkg_t)
            material,coverage=rade.render_material(camera,interface,pipe,dataset.kernel_size)
            target,valid=glass_targets(pkg_i['expected_depth'][0],pkg_t['expected_depth'][0],
                                       pkg_i['mask'][0],pkg_t['mask'][0],prior['diffuse_albedo'].mean(-1),
                                       prior['basecolor'].mean(-1),prior_valid,relative_gap=config['relative_gap'])
            losses['material']=config['material_weight']*material_loss(material[0],target,valid)
            opaque=valid & (target==0)
            denominator=pkg_i['expected_depth'][0].detach().abs().clamp_min(1e-4)
            losses['opaque_depth']=config['opaque_alignment_weight']*masked_mean(
                (pkg_t['expected_depth'][0]-pkg_i['expected_depth'][0].detach()).abs()/denominator,opaque)
            size=config['patch_size']; h,w=gt.shape[1:]
            if min(h,w)<size: raise ValueError('Image smaller than patch')
            top=rng.randrange(h-size+1); left=rng.randrange(w-size+1)
            ys,xs=torch.meshgrid(torch.arange(top,top+size,device='cuda'),
                                 torch.arange(left,left+size,device='cuda'),indexing='ij')
            ys,xs=ys.flatten(),xs.flatten()
            origins,directions,length=rade.camera_rays(camera,ys.float(),xs.float())
            depth=pkg_i['expected_depth'][0,ys,xs]
            # Detach secondary ray origins/normals initially, as an explicit
            # stable-transport approximation; primary geometry still learns.
            w2c=camera.world_view_transform.T
            normals=(normal_i[ys,xs] @ w2c[:3,:3]).detach()
            epsilon=scene.cameras_extent*config['ray_epsilon_extent']
            to,td,ro,rd=secondary_rays(origins,directions,depth.detach()*length,normals,epsilon)
            tfield=rade.field_from_gaussians(transmission)
            transported=trace_rgb_checkpointed(tfield,to,td,background,ray_chunk=config['ray_chunk'])
            reflected=trace_rgb_checkpointed(tfield,ro,rd,background,ray_chunk=config['ray_chunk'])
            a=coverage[0,ys,xs]
            surface=pkg_i['render'][:,ys,xs].T/a[:,None].clamp_min(1e-6)
            combined=compose(surface,transported,reflected,a,
                             material[0,ys,xs],material[1,ys,xs],directions,normals,f0=config['f0'])
            patch=combined['rgb'].reshape(size,size,3).permute(2,0,1)
            losses['rgb_transport']=photo_loss(patch,gt[:,top:top+size,left:left+size],opt)
            diagnostics=dict(positive_pixels=int((target.bool()&valid).sum()),valid_material_pixels=int(valid.sum()),
                             mean_transparency=float(material[0].detach().mean()),transport_rays=size*size)
        loss=sum(losses.values())
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss at '+str(step))
        loss.backward()
        with torch.no_grad():
            # Match upstream maintenance order; child material follows exact parents.
            for field,pkg in ((interface,pkg_i),(transmission,pkg_t)):
                if pkg is None: continue
                if step<opt.densify_until_iter:
                    visible=pkg['visibility_filter']
                    field.max_radii2D[visible]=torch.maximum(field.max_radii2D[visible],pkg['radii'][visible])
                    field.add_densification_stats(pkg['viewspace_points'],visible)
                    if step>opt.densify_from_iter and step%opt.densification_interval==0:
                        field.densify_and_prune(opt.densify_grad_threshold,.05,scene.cameras_extent,
                                               20 if step>opt.opacity_reset_interval else None)
                        if len(field.get_xyz)==0: raise RuntimeError('All Gaussian points pruned')
                        field.compute_3D_filter(cameras=cameras)
                    if step%opt.opacity_reset_interval==0:
                        field.reset_opacity()
                elif step%100==0 and step<config['iterations']-100:
                    field.compute_3D_filter(cameras=cameras)
                field.optimizer.step(); field.optimizer.zero_grad(set_to_none=True)
            interface.material.optimizer.step()
            interface.material.optimizer.zero_grad(set_to_none=True)
        if step%10==0 or step==first or step==config['iterations']:
            torch.cuda.synchronize()
            elapsed=elapsed_before+time.perf_counter()-started
            status=dict(step=step,total=config['iterations'],stage='joint' if joint else 'interface_warmup',
                        elapsed_seconds=elapsed,interface_count=len(interface.get_xyz),transmission_count=len(transmission.get_xyz),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),views_per_iteration=1,
                        losses={k:float(v.detach()) for k,v in losses.items()},**diagnostics)
            temp=status_path.with_suffix('.tmp'); temp.write_text(json.dumps(status,indent=2)); temp.replace(status_path)
            print(json.dumps(status),flush=True)
        if step%config['checkpoint_every']==0 or step==config['iterations']:
            save_checkpoint(args.output/f'step_{step:06d}.pt',step,interface,transmission,config,opt,rng,stack,
                            elapsed_before+time.perf_counter()-started,provenance)
    status['state']='complete'; status_path.write_text(json.dumps(status,indent=2))
