"""Prepare isolated train/eval inputs from synchronized N3DV frame zero."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from plyfile import PlyData, PlyElement


def write_ply(path,xyz,rgb):
    path.parent.mkdir(parents=True,exist_ok=True)
    array=np.empty(len(xyz),dtype=[(n,'f4') for n in ['x','y','z','nx','ny','nz']]+[(n,'u1') for n in ['red','green','blue']])
    for i,n in enumerate(['x','y','z']): array[n]=xyz[:,i]
    for n in ['nx','ny','nz']: array[n]=0
    for i,n in enumerate(['red','green','blue']): array[n]=rgb[:,i]
    PlyData([PlyElement.describe(array,'vertex')]).write(str(path))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    src=root/'coffee_frame0'
    manifest=json.loads((src/'preparation_manifest.json').read_text())
    names=manifest['camera_order']
    assert names[0]=='cam00'
    poses=np.load(src/'poses_bounds.npy')[:,:15].reshape(-1,3,5)
    initialization=json.loads((src/'colmap_0/initialization_manifest.json').read_text())
    assert initialization['held_out_rgb_used'] is False
    train=root/'data/train'
    evaluation=root/'data/eval'
    baseline=root/'data/2dgs_train'
    for out in [train,evaluation,baseline]:
        if out.exists(): raise FileExistsError(out)
        out.mkdir(parents=True)
    camera_records=[]
    for i,(name,pose) in enumerate(zip(names,poses),1):
        h,w,f=pose[:,4]
        image=cv2.imread(str(src/name/'000000.png'))
        assert image.shape[:2]==(int(h),int(w))
        tw,th=676,507
        assert abs(tw/w-th/h)<1e-10
        scale=tw/w
        K=np.array([[f*scale,0,tw/2],[0,f*scale,th/2],[0,0,1]],dtype=np.float64)
        c2w=np.eye(4)
        c2w[:3,:4]=np.stack([pose[:,1],pose[:,0],-pose[:,2],pose[:,3]],axis=1)
        w2c=np.linalg.inv(c2w)
        assert np.allclose(w2c[:3,:3]@w2c[:3,:3].T,np.eye(3),atol=1e-5)
        assert abs(np.linalg.det(w2c[:3,:3])-1)<1e-5
        resized=cv2.resize(image,(tw,th),interpolation=cv2.INTER_AREA)
        dest=(evaluation if name=='cam00' else train)/'images'/name/'000000.png'
        dest.parent.mkdir(parents=True)
        assert cv2.imwrite(str(dest),resized)
        camera_records.append(dict(name=name,id=i,K=K,R=w2c[:3,:3],T=w2c[:3,3:4],center=c2w[:3,3],w=tw,h=th,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    for out,records in [(train,camera_records[1:]),(evaluation,camera_records[:1])]:
        intri=cv2.FileStorage(str(out/'intri.yml'),cv2.FILE_STORAGE_WRITE)
        extri=cv2.FileStorage(str(out/'extri.yml'),cv2.FILE_STORAGE_WRITE)
        for fs in [intri,extri]:
            fs.startWriteStruct('names',cv2.FileNode_SEQ)
            for c in records: fs.write('',c['name'])
            fs.endWriteStruct()
        for c in records:
            n=c['name']
            intri.write('K_'+n,c['K']); intri.write('H_'+n,c['h']); intri.write('W_'+n,c['w'])
            intri.write('D_'+n,np.zeros((5,1)))
            extri.write('Rot_'+n,c['R']); extri.write('T_'+n,c['T'])
        intri.release(); extri.release()
    points=[]; colors=[]
    for line in (src/'colmap_0/sparse/0/points3D.txt').read_text().splitlines():
        if not line or line.startswith('#'): continue
        row=line.split(); points.append([float(x) for x in row[1:4]]); colors.append([int(x) for x in row[4:7]])
    xyz=np.array(points,dtype=np.float32); rgb=np.array(colors,dtype=np.uint8)
    assert len(xyz)>0 and np.isfinite(xyz).all()
    write_ply(train/'sparse/0/points3D.ply',xyz,rgb)
    # Paper initialization: 32^3 cells, 5 random reflection primitives per cell.
    rng=np.random.default_rng(0)
    low,high=xyz.min(0),xyz.max(0)
    cells=np.stack(np.meshgrid(*[np.arange(32)]*3,indexing='ij'),axis=-1).reshape(-1,3)
    cells=np.repeat(cells,5,axis=0)
    env=low+(cells+rng.random(cells.shape))/32*(high-low)
    write_ply(train/'envs/points3D.ply',env,np.full(env.shape,127,dtype=np.uint8))
    # Baseline receives exactly the same pixels and sparse points, with no cam00 file.
    (baseline/'images').mkdir()
    sparse=baseline/'sparse/0'; sparse.mkdir(parents=True)
    colmap_cameras=[]; colmap_images=[]
    for c in camera_records[1:]:
        (baseline/'images'/f"{c['name']}.png").symlink_to(train/'images'/c['name']/'000000.png')
        # Positive-w unit quaternion, COLMAP world-to-camera rotation.
        from scipy.spatial.transform import Rotation
        xyzw=Rotation.from_matrix(c['R']).as_quat()
        q=xyzw[[3,0,1,2]]
        values=' '.join(format(float(x),'.17g') for x in [*q,*c['T'].ravel()])
        colmap_images.append(f"{c['id']} {values} {c['id']} {c['name']}.png\n\n")
        k=c['K']
        colmap_cameras.append(f"{c['id']} PINHOLE {c['w']} {c['h']} {k[0,0]:.17g} {k[1,1]:.17g} {k[0,2]:.17g} {k[1,2]:.17g}\n")
    (sparse/'images.txt').write_text(''.join(colmap_images))
    (sparse/'cameras.txt').write_text(''.join(colmap_cameras))
    shutil.copy2(src/'colmap_0/sparse/0/points3D.txt',sparse/'points3D.txt')
    shutil.copy2(train/'sparse/0/points3D.ply',sparse/'points3D.ply')
    # Walk nearby camera centers at this single time; held-out view never participates.
    remaining=list(camera_records[1:]); ordered=[remaining.pop(0)]
    while remaining:
        index=int(np.argmin([np.linalg.norm(c['center']-ordered[-1]['center']) for c in remaining]))
        ordered.append(remaining.pop(index))
    inputs=root/'prior_inputs/static_multiview'; inputs.mkdir(parents=True)
    mapping=[]
    for index,c in enumerate(ordered):
        dest=inputs/f'{index:04d}.png'
        dest.symlink_to(train/'images'/c['name']/'000000.png')
        mapping.append(dict(index=index,camera=c['name'],frame=0,sha256=c['sha256']))
    (root/'prior_mapping.json').write_text(json.dumps(mapping,indent=2))
    serial=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()} for c in camera_records]
    proof=dict(cameras=serial,train_cameras=names[1:],test_cameras=['cam00'],resolution=[676,507],frame=0,
               points=len(xyz),reflection_points=len(env),reflection_bbox=[low.tolist(),high.tolist()],
               camera_convention='OpenCV world-to-camera: right/down/forward; LLFF [down,right,back] columns converted to [right,down,-back]',
               prior_resolution=[512,384],prior_camera_scale=[512/676,384/507],
               image_conversion='OpenCV INTER_AREA from 1352x1014 to 676x507; K rows 0,1 scaled by 0.5; no crop',
               annotation_sha256=hashlib.sha256((root/'evaluation_regions.json').read_bytes()).hexdigest())
    (root/'input_verification.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps({k:v for k,v in proof.items() if k!='cameras'},indent=2))


if __name__=='__main__': main()
