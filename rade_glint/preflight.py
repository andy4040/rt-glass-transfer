"""Read-only runtime/source check; never starts training or installs packages."""
import argparse
import importlib.util
import json
import subprocess
import torch
from rg import rade
from rg.data import validate_dataset


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source');p.add_argument('--priors');args=p.parse_args()
    commit=subprocess.run(['git','-C',str(rade.IGS),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    dirty=subprocess.run(['git','-C',str(rade.IGS),'status','--porcelain'],capture_output=True,text=True,check=True).stdout
    result=dict(torch=torch.__version__,cuda_available=torch.cuda.is_available(),igs_commit=commit,
                pinned_commit_matches=commit==rade.UPSTREAM_COMMIT,upstream_dirty=dirty,
                source_hashes=rade.source_hashes(),secondary_renderer='slow 3D reference, no BVH/OptiX')
    result['modules']={n:importlib.util.find_spec(n) is not None for n in
                       ('diff_gaussian_rasterization_rade','diff_gaussian_rasterization_compress','simple_knn',
                        'plyfile','scipy','cv2','trimesh','open3d','torchvision')}
    if bool(args.source)!=bool(args.priors): p.error('Supply both --source and --priors')
    if args.source: result['train_cameras']=validate_dataset(args.source,args.priors)[0]
    result['can_attempt_gpu_smoke']=bool(result['cuda_available'] and all(result['modules'].values()) and result['pinned_commit_matches'] and not dirty)
    print(json.dumps(result,indent=2))
    return 0 if result['can_attempt_gpu_smoke'] else 2


if __name__=='__main__': raise SystemExit(main())
