"""CPU-only verification before arming the waiting queue."""
import argparse,json,sys
from pathlib import Path
from export_masks import read_train_cameras,load_prior
from prepare_data import read_colmap_text
p=argparse.ArgumentParser();p.add_argument('--glint-base',type=Path,required=True);args=p.parse_args();base=args.glint_base
train=base/'data/train';names=read_train_cameras(train)
cameras,images=read_colmap_text(base/'data/2dgs_train/sparse/0')
assert len(names)==17 and set(names)=={i['camera'] for i in images} and 'cam00' not in names
for name,size in names.items():
    for kind in ['diffuse_albedo','basecolor']:
        prior=load_prior(train/'diffrens'/kind/name/'000000.png',size)
        assert prior.shape==(size[1],size[0],3)
assert (base/'data/2dgs_train/sparse/0/points3D.ply').is_file()
print(json.dumps({'preflight':'PASS','train_cameras':list(names),'count':len(names),'cam00_in_train':False,'priors_verified':34,'GPU_used':False}))
