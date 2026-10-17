"""Validate normal-coordinate convention using training SfM tracks, then package priors."""
from pathlib import Path
import sys
import json
import shutil
import hashlib
import runpy
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'GLINT'))
from glint.camera import load_easyvolcap_cameras
mapping=json.loads((ROOT/'prior_mapping.json').read_text())
cameras=load_easyvolcap_cameras(ROOT/'data/train')
assert 'cam00' not in cameras
loader=runpy.run_path(str(ROOT/'2dgs/scene/colmap_loader.py'))
source=ROOT/'coffee_frame0/colmap_0/sparse/0'
images=loader['read_extrinsics_text'](str(source/'images.txt'))
image_to_name={key:Path(value.name).stem for key,value in images.items()}
xyz=[]; tracks=[]
for line in (source/'points3D.txt').read_text().splitlines():
    if not line.strip() or line.startswith('#'):continue
    values=line.split();xyz.append([float(v) for v in values[1:4]])
    tracks.append({image_to_name[int(v)] for v in values[8::2] if int(v) in image_to_name})
xyz=np.asarray(xyz)
neighbors=cKDTree(xyz).query(xyz,k=16)[1]
patch=xyz[neighbors]; centered=patch-patch.mean(axis=1,keepdims=True)
eigenvalues,eigenvectors=np.linalg.eigh(centered.transpose(0,2,1)@centered)
planar=(eigenvalues[:,0]/np.maximum(eigenvalues[:,1],1e-12)<.1)&(eigenvalues[:,1]/np.maximum(eigenvalues[:,2],1e-12)>.05)
normal_world=eigenvectors[:,:,0]
candidate_signs={'identity':[1,1,1],'flip_yz':[1,-1,-1],'flip_xz':[-1,1,-1],'flip_xy':[-1,-1,1]}
cosines={key:[] for key in candidate_signs};per_camera=[]
for index,row in enumerate(mapping):
    name=row['camera']; assert name!='cam00' and row['frame']==0
    camera=cameras[name]; path=ROOT/'prior_outputs/static_multiview'/f'0000.{index:04d}.normal.png'
    normal_image=np.asarray(Image.open(path).convert('RGB'),dtype=np.float64)/127.5-1
    h,w=normal_image.shape[:2]; R=camera.R.numpy();T=camera.T.numpy().reshape(3)
    pc=xyz@R.T+T
    projected=pc@camera.K.numpy().T; uv=projected[:,:2]/projected[:,2:]*[w/676,h/507]
    inside=(uv[:,0]>=1)&(uv[:,0]<w-1)&(uv[:,1]>=1)&(uv[:,1]<h-1)&(pc[:,2]>.1)
    tracked=np.array([name in track for track in tracks]); valid=inside&planar&tracked
    pixels=np.round(uv[valid]).astype(int)
    predicted=normal_image[pixels[:,1],pixels[:,0]]
    predicted/=np.maximum(np.linalg.norm(predicted,axis=1,keepdims=True),1e-8)
    reference=normal_world[valid]@R.T
    reference*=np.where(np.sum(reference*pc[valid],axis=1,keepdims=True)>0,-1,1)
    stats={'camera':name,'tracked_planar_points':int(valid.sum())}
    for key,sign in candidate_signs.items():
        values=np.sum(predicted*np.array(sign)*reference,axis=1)
        cosines[key].extend(values.tolist())
        stats[key]=float(np.median(values)) if len(values) else None
    per_camera.append(stats)
scores={key:float(np.median(value)) for key,value in cosines.items()}
ranked=sorted(scores,key=scores.get,reverse=True); chosen=ranked[0]
report={'source':'training-only fixed-calibration SfM tracks and PCA plane normals oriented toward each camera',
        'candidate_median_cosine':scores,'per_camera':per_camera,'selected_transform':chosen,
        'component_signs':candidate_signs[chosen],'test_camera_used':False,
        'interpretation':'Coordinate convention check, not a prior accuracy or glass geometry metric',
        'prior_resolution':[512,384],'target_resolution':[676,507],
        'depth':'Official relative RGB prior preserved; GLINT performs its own depth decode and scale/shift invariant supervision'}
(ROOT/'prior_coordinate_verification.json').write_text(json.dumps(report,indent=2))
assert len(cosines[chosen])>=100,'Insufficient training SfM correspondences for normal convention validation'
assert scores[chosen]>.3 and scores[chosen]-scores[ranked[1]]>.05,report
manifest=[]
for index,row in enumerate(mapping):
    for kind in ['normal','depth','diffuse_albedo','basecolor']:
        source=ROOT/'prior_outputs/static_multiview'/f'0000.{index:04d}.{kind}.png'
        destination=ROOT/'data/train/diffrens'/kind/row['camera']/'000000.png'
        destination.parent.mkdir(parents=True,exist_ok=True)
        image=np.asarray(Image.open(source).convert('RGB'))
        assert image.shape==(384,512,3),(source,image.shape)
        if kind=='normal':
            image=image.copy()
            for axis,sign in enumerate(candidate_signs[chosen]):
                if sign==-1:image[...,axis]=255-image[...,axis]
            Image.fromarray(image).save(destination)
        else:shutil.copy2(source,destination)
        manifest.append({'camera':row['camera'],'pass':kind,'source':str(source),'target':str(destination),
                         'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()})
(ROOT/'prior_manifest.json').write_text(json.dumps(manifest,indent=2))
preview=[]
first=mapping[0]['camera']
preview.append(Image.open(ROOT/'data/train/images'/first/'000000.png').convert('RGB').resize((512,384)))
for kind in ['normal','depth','diffuse_albedo','basecolor']:
    preview.append(Image.open(ROOT/'data/train/diffrens'/kind/first/'000000.png').convert('RGB'))
montage=Image.new('RGB',(512*len(preview),384))
for i,image in enumerate(preview):montage.paste(image,(i*512,0))
montage.save(ROOT/'prior_preview_rgb_normal_depth_albedo_basecolor.png')
print(json.dumps(report,indent=2))
