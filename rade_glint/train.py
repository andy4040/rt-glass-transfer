import argparse
from pathlib import Path
from rg.experiment import train


if __name__=='__main__':
    p=argparse.ArgumentParser(description='Joint RaDe Interface/Transmission; reference 3D secondary tracer')
    p.add_argument('--source',type=Path,required=True,help='Train-only text COLMAP dataset')
    p.add_argument('--priors',type=Path,required=True,help='Prepared DiffusionRenderer NPZ directory')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--resume',type=Path)
    train(p.parse_args())
