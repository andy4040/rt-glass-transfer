"""Launch the GLINT-only 60k mean/median-normal experiment from fresh inputs."""
import argparse,json,os,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--train-root',type=Path,required=True,help='Prepared same-time train-only images, cameras, SfM and DiffusionRenderer priors')
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--print-only',action='store_true')
 a=p.parse_args();root=a.train_root.resolve();out=a.output.resolve()
 source=Path(__file__).resolve().parent/'GLINT'
 assert root.is_dir()
 assert not (root/'images/cam00').exists(),'cam00 must remain held out'
 assert (root/'sparse/0/points3D.ply').is_file()
 assert (root/'envs/points3D.ply').is_file()
 assert not out.exists(),'Use a new output directory; do not overwrite or implicitly resume'
 config=dict(configs=[str(source/'configs/base.yaml'),str(source/'configs/models/glint.yaml')],
  dataloader_cfg=dict(dataset_cfg=dict(data_root=str(root),ratio=1.0,view_sample=[0,None,1],frame_sample=[0,1,1],
  use_diffren=True,use_normals=True,diffren_params=['normal','depth','diffuse_albedo','basecolor'])))
 # JSON is a valid YAML subset accepted by the official config loader.
 cfg=out/'input_config.yaml'
 cmd=[sys.executable,'-u','-m','glint.train','--config',str(cfg),'--data-root',str(root),'--output-dir',str(out/'model'),
  '--max-steps','60000','--transmission-start','5000','--reflection-start','5000','--interface-geometry-freeze','40000',
  '--save-every','10000','--image-every','5000','--seed','0','--num-workers','4']
 print(json.dumps(dict(command=cmd,config=config),indent=2))
 if a.print_only:return
 out.mkdir(parents=True);cfg.write_text(json.dumps(config,indent=2))
 (out/'command.json').write_text(json.dumps(cmd,indent=2))
 env=dict(os.environ,PYTHONPATH=str(source)+os.pathsep+os.environ.get('PYTHONPATH',''))
 subprocess.run(cmd,cwd=source,env=env,check=True)
if __name__=='__main__':main()
