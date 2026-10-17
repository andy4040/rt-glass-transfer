"""Build flat train-only COLMAP and ordered same-time prior inputs."""
from pathlib import Path
import json, shutil, hashlib
import cv2
import numpy as np
from prepare_n3dv import _calibration_records

root=Path(__file__).resolve().parent
src=root/'coffee_frame0'
proof=json.loads((src/'colmap_0/initialization_manifest.json').read_text())
assert proof['held_out_rgb_used'] is False and proof['supplied_calibration_verified']
manifest=json.loads((src/'preparation_manifest.json').read_text())
poses=np.load(src/'poses_bounds.npy')
records=_calibration_records(poses,manifest['camera_order'])
out=root/'data/train'
if out.exists(): raise FileExistsError(out)
(out/'images').mkdir(parents=True)
sparse=out/'sparse/0'; sparse.mkdir(parents=True)
# Use the original fixed calibration, selecting camera rows by training image IDs.
original=src/'colmap_0/sparse/0'
rows=iter((original/'images.txt').read_text().splitlines())
images=[]; camera_ids=set(); centers={}
from scipy.spatial.transform import Rotation
for line in rows:
    if not line.strip() or line.startswith('#'): continue
    fields=line.split(); observations=next(rows)
    name=Path(fields[9]).stem
    if name=='cam00': continue
    assert name in proof['train_cameras']
    camera_ids.add(int(fields[8]))
    q=np.array(list(map(float,fields[1:5])))
    r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    centers[name]=-r.T@np.array(list(map(float,fields[5:8])))
    fields[9]=name+'.png'
    images.append(' '.join(fields)+'\n\n')
    image=cv2.imread(str(src/name/'000000.png'))
    assert image is not None
    image=cv2.resize(image,(676,507),interpolation=cv2.INTER_AREA)
    assert cv2.imwrite(str(out/'images'/(name+'.png')),image)
cameras=[]
for line in (original/'cameras.txt').read_text().splitlines():
    if not line.strip() or line.startswith('#'): continue
    f=line.split()
    if int(f[0]) not in camera_ids: continue
    w,h=map(int,f[2:4]); sx,sy=676/w,507/h
    if f[1]=='SIMPLE_PINHOLE':
        focal,cx,cy=map(float,f[4:]); fx,fy=focal*sx,focal*sy
    elif f[1]=='PINHOLE':
        fx,fy,cx,cy=map(float,f[4:]); fx*=sx; fy*=sy
    else: raise ValueError(f[1])
    cameras.append(f'{f[0]} PINHOLE 676 507 {fx:.17g} {fy:.17g} {cx*sx:.17g} {cy*sy:.17g}\n')
(sparse/'cameras.txt').write_text(''.join(cameras))
(sparse/'images.txt').write_text(''.join(images))
shutil.copy2(original/'points3D.txt',sparse/'points3D.txt')
assert set(centers)==set(proof['train_cameras'])
remaining=sorted(centers); ordered=[remaining.pop(0)]
while remaining:
    name=min(remaining,key=lambda n:np.linalg.norm(centers[n]-centers[ordered[-1]]))
    ordered.append(name); remaining.remove(name)
prior=root/'prior_inputs/static_multiview';prior.mkdir(parents=True)
mapping=[]
for index,name in enumerate(ordered):
    rgb=out/'images'/(name+'.png')
    (prior/f'{index:04d}.png').symlink_to(rgb)
    mapping.append(dict(index=index,camera=name,frame=0,sha256=hashlib.sha256(rgb.read_bytes()).hexdigest()))
(root/'prior_mapping.json').write_text(json.dumps(mapping,indent=2))
print(json.dumps(dict(train_cameras=ordered,cam00_used=False,resolution=[676,507],frame=0)))
