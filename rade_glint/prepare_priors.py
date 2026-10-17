"""Convert existing GLINT-compatible DiffusionRenderer prior PNGs, not a teacher.

The supplied normal signs must come from a train-view coordinate validation.
No GLINT checkpoint/model is read, and no cam00 file is opened.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from rg.data import check_names


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-images',type=Path,required=True,help='Flat train-only camXX.png directory')
    p.add_argument('--diffren-root',type=Path,required=True,help='diffrens/{kind}/{cam}/000000.png')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--normal-signs',type=float,nargs=3,required=True)
    p.add_argument('--normal-input-space',choices=['raw','camera'],required=True,
                   help='camera: PNGs already transformed; do NOT apply component signs twice')
    p.add_argument('--coordinate-report',type=Path,required=True,help='Train-only validation of supplied normal transform')
    args=p.parse_args()
    if any(abs(s)!=1 for s in args.normal_signs): p.error('Signs must be +/-1')
    report=json.loads(args.coordinate_report.read_text())
    if report.get('test_camera_used') is not False: p.error('Coordinate report must exclude cam00')
    if list(report.get('component_signs',[]))!=args.normal_signs: p.error('Signs disagree with coordinate report')
    names=check_names(f.stem for f in sorted(args.train_images.glob('*.png')))
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    hashes={}; rgb_hashes={}
    for name in names:
        with Image.open(args.train_images/(name+'.png')) as im: size=im.size
        rgb_hashes[name]=hashlib.sha256((args.train_images/(name+'.png')).read_bytes()).hexdigest()
        maps={}
        for kind in ('depth','normal','diffuse_albedo','basecolor'):
            path=args.diffren_root/kind/name/'000000.png'
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as im:
                rgb=np.asarray(im.convert('RGB').resize(size,Image.Resampling.BILINEAR),dtype=np.float32)/255.
            maps[kind]=rgb
        maps['depth']=np.clip((maps['depth']*np.array([1,1/256,1/65536],dtype=np.float32)).sum(-1),0,1)
        normal=maps['normal']*2-1
        if args.normal_input_space=='raw': normal=normal*np.asarray(args.normal_signs,dtype=np.float32)
        length=np.linalg.norm(normal,axis=-1,keepdims=True)
        maps['normal']=normal/np.maximum(length,1e-6)
        maps['valid']=(length[...,0]>.1)&(maps['depth']>0)
        np.savez_compressed(args.output/(name+'.npz'),**maps)
    manifest=dict(frame=0,train_cameras=names,held_out_used=False,normal_space='camera',depth_kind='relative',
                  input_rgb_sha256=rgb_hashes,normal_input_space=args.normal_input_space,
                  normal_signs=args.normal_signs,coordinate_report_sha256=hashlib.sha256(args.coordinate_report.read_bytes()).hexdigest(),
                  prior_sha256=hashes,valid_semantics='finite nonzero depth/normal, NOT calibrated confidence',
                  depth_decode='GLINT RGB packing: R + G/256 + B/65536; scalar prior is relative',
                  resize='PIL bilinear uint8 before decode; not bit-identical to original OpenCV loader')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
