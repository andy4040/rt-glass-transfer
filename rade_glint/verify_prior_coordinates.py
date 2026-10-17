"""Choose normal convention from train-only SfM planar tracks; export raw maps."""
from pathlib import Path
import json,runpy,shutil,hashlib
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
root=Path(__file__).resolve().parent
loader=runpy.run_path(str(root/'vendor/IGS/submodules/RaDe-GS/scene/colmap_loader.py'))
sparse=root/'coffee_frame0/colmap_0/sparse/0'
images=loader['read_extrinsics_text'](str(sparse/'images.txt'))
lookup={Path(v.name).stem:v for v in images.values()}
id_to_name={i:Path(v.name).stem for i,v in images.items()}
cameras=loader['read_intrinsics_text'](str(root/'data/train/sparse/0/cameras.txt'))
mapping=json.loads((root/'prior_mapping.json').read_text())
xyz=[];tracks=[]
for line in (sparse/'points3D.txt').read_text().splitlines():
    if not line.strip() or line.startswith('#'):continue
    p=line.split();xyz.append(list(map(float,p[1:4])))
    tracks.append({id_to_name[int(v)] for v in p[8::2]})
xyz=np.array(xyz);assert len(xyz)>=16
neighbors=cKDTree(xyz).query(xyz,k=16)[1]
patch=xyz[neighbors];patch-=patch.mean(1,keepdims=True)
ev,vec=np.linalg.eigh(patch.transpose(0,2,1)@patch)
planar=(ev[:,0]/np.maximum(ev[:,1],1e-12)<.1)&(ev[:,1]/np.maximum(ev[:,2],1e-12)>.05)
signs={'identity':[1,1,1],'flip_yz':[1,-1,-1],'flip_xz':[-1,1,-1],'flip_xy':[-1,-1,1]}
values={k:[] for k in signs};per_camera=[];hashes={}
for row in mapping:
    name=row['camera'];assert name!='cam00' and row['frame']==0
    info=lookup[name];r=info.qvec2rotmat();pc=xyz@r.T+info.tvec
    intr=cameras[info.camera_id];fx,fy,cx,cy=intr.params
    path=root/'prior_outputs/static_multiview'/f"0000.{row['index']:04d}.normal.png"
    pred=np.asarray(Image.open(path).convert('RGB'),dtype=np.float64)/127.5-1
    h,w=pred.shape[:2]
    uv=pc[:,:2]/np.maximum(pc[:,2:],1e-12)*[fx,fy]+[cx,cy]
    uv*=np.array([w/intr.width,h/intr.height])
    valid=planar&(pc[:,2]>.1)&(uv[:,0]>=1)&(uv[:,0]<w-1)&(uv[:,1]>=1)&(uv[:,1]<h-1)
    valid&=np.array([name in t for t in tracks])
    pixel=np.round(uv[valid]).astype(int);p=pred[pixel[:,1],pixel[:,0]]
    p/=np.maximum(np.linalg.norm(p,axis=1,keepdims=True),1e-8)
    ref=vec[valid,:,0]@r.T
    ref*=np.where((ref*pc[valid]).sum(1,keepdims=True)>0,-1,1)
    stats=dict(camera=name,tracked_planar_points=int(valid.sum()))
    for key,sign in signs.items():
        score=(p*np.array(sign)*ref).sum(1)
        values[key].extend(score.tolist());stats[key]=float(np.median(score)) if len(score) else None
    per_camera.append(stats)
scores={k:float(np.median(v)) for k,v in values.items()}
ranked=sorted(scores,key=scores.get,reverse=True);best=ranked[0]
assert len(values[best])>=100 and scores[best]>.3 and scores[best]-scores[ranked[1]]>.05,scores
report=dict(test_camera_used=False,component_signs=signs[best],selected_transform=best,
    candidate_median_cosine=scores,per_camera=per_camera,
    source='Training SfM tracked planar neighborhoods; normals oriented toward camera')
(root/'prior_coordinate_verification.json').write_text(json.dumps(report,indent=2))
for row in mapping:
    for kind in ('normal','depth','diffuse_albedo','basecolor'):
        src=root/'prior_outputs/static_multiview'/f"0000.{row['index']:04d}.{kind}.png"
        dest=root/'diffrens'/kind/row['camera']/'000000.png'
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        hashes[str(dest.relative_to(root))]=hashlib.sha256(src.read_bytes()).hexdigest()
(root/'prior_png_hashes.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps(report,indent=2))
