"""Strict train-only COLMAP input and explicitly decoded DiffusionRenderer priors."""
import json
import hashlib
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def check_names(names):
    names=list(names)
    if not names or len(names)!=len(set(names)):
        raise ValueError('Empty or duplicate camera inventory')
    if any(not re.fullmatch(r'cam\d+',n) for n in names) or 'cam00' in names:
        raise ValueError('Only explicit camXX training cameras; cam00 is evaluation-only')
    return names


def validate_dataset(source,prior_root):
    source,prior_root=Path(source),Path(prior_root)
    names=check_names(p.stem for p in sorted((source/'images').glob('*.png')))
    sparse=source/'sparse/0'
    # Reject ambiguous/stale binary precedence in upstream COLMAP loader.
    if (sparse/'cameras.bin').exists() or (sparse/'images.bin').exists():
        raise ValueError('Use audited train-only text COLMAP input; binary calibration is unsupported here')
    cameras={}
    for line in (sparse/'cameras.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'): continue
        p=line.split(); cid,kind,w,h=int(p[0]),p[1],int(p[2]),int(p[3])
        if kind!='PINHOLE' or len(p)!=8: raise ValueError('Expected PINHOLE fx fy cx cy')
        fx,fy,cx,cy=map(float,p[4:])
        if min(fx,fy,w,h)<=0 or not np.allclose((cx,cy),(w/2,h/2),atol=1e-5,rtol=0):
            raise ValueError('Original RaDe requires centered principal point')
        cameras[cid]=(w,h)
    rows=iter((sparse/'images.txt').read_text().splitlines()); mapped={}
    for line in rows:
        if not line.strip() or line.startswith('#'): continue
        p=line.split()
        if len(p)!=10: raise ValueError('Malformed COLMAP image row')
        name=Path(p[9]).stem
        if name in mapped: raise ValueError('Duplicate camera')
        mapped[name]=cameras[int(p[8])]
        if next(rows,None) is None: raise ValueError('Missing COLMAP observation row')
    check_names(mapped)
    if set(mapped)!=set(names): raise ValueError('Calibration/RGB camera mismatch')
    manifest=json.loads((prior_root/'manifest.json').read_text())
    if manifest.get('held_out_used') is not False or manifest.get('frame')!=0:
        raise ValueError('Need same-time frame=0, train-only prior provenance')
    if set(manifest['train_cameras'])!=set(names): raise ValueError('Prior camera mismatch')
    if manifest.get('normal_space')!='camera' or manifest.get('depth_kind')!='relative':
        raise ValueError('Expected signed camera normals and decoded relative depth')
    for name in names:
        rgb_path=source/'images'/(name+'.png')
        if manifest.get('input_rgb_sha256',{}).get(name)!=hashlib.sha256(rgb_path.read_bytes()).hexdigest():
            raise ValueError('Prior and training RGB provenance mismatch: '+name)
        with Image.open(rgb_path) as image:
            if image.size!=mapped[name]: raise ValueError('RGB/calibration size mismatch')
        with np.load(prior_root/(name+'.npz')) as p:
            w,h=mapped[name]
            for key,shape in [('depth',(h,w)),('valid',(h,w)),('normal',(h,w,3)),
                              ('diffuse_albedo',(h,w,3)),('basecolor',(h,w,3))]:
                if p[key].shape!=shape or not np.isfinite(p[key]).all():
                    raise ValueError(f'Invalid prior {name}:{key}')
            if p['valid'].dtype!=np.bool_: raise ValueError('valid must be boolean')
            for key in ('diffuse_albedo','basecolor'):
                if (p[key]<0).any() or (p[key]>1).any(): raise ValueError('Prior outside [0,1]')
            if (np.abs(p['normal'])>1.001).any(): raise ValueError('Invalid signed normal')
    return names,manifest


def load_prior(root,name,device):
    if name=='cam00': raise ValueError('cam00 cannot enter prior supervision')
    with np.load(Path(root)/(name+'.npz')) as p:
        return {k:torch.from_numpy(p[k].copy()).to(device) for k in
                ('depth','normal','diffuse_albedo','basecolor','valid')}
