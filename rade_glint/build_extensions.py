"""Build original IGS extensions from staged copies, preserving vendor sources.

Linux/CUDA only. Installs into the currently selected Python environment.
Only missing standard headers are patched in staging, with a build manifest.
"""
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import torch
from rg import rade


def main():
    if platform.system()!='Linux' or not torch.cuda.is_available():
        raise RuntimeError('Run inside the intended Linux CUDA environment')
    root=rade.ROOT/'build'
    if root.exists(): raise FileExistsError('Preserve previous build; use a new checkout/build directory')
    root.mkdir()
    sources={name:rade.RADE/'submodules'/name for name in
             ('diff-gaussian-rasterization','compress-diff-gaussian-rasterization')}
    sources['simple-knn']=rade.ROOT/'vendor/simple-knn'
    glm_source=rade.RADE/'submodules/diff-gaussian-rasterization/third_party/glm'
    if not (glm_source/'glm/glm.hpp').is_file():
        raise FileNotFoundError('Run git submodule update --init --recursive in vendor/IGS')
    manifest={'igs_commit':rade.UPSTREAM_COMMIT,'patches':[],
              'simple_knn_commit':subprocess.run(['git','-C',str(sources['simple-knn']),'rev-parse','HEAD'],
                                               capture_output=True,text=True,check=True).stdout.strip()}
    for name,source in sources.items():
        if not source.is_dir(): raise FileNotFoundError(source)
        target=root/name
        shutil.copytree(source,target,ignore=shutil.ignore_patterns('.git','__pycache__','build','*.egg-info'))
        glm=target/'third_party/glm'
        if name!='simple-knn' and glm.is_file():
            # Windows Git checkouts may store a symlink as its target text.
            glm.unlink()
            shutil.copytree(glm_source,glm,ignore=shutil.ignore_patterns('.git'))
        patches=[(target/'cuda_rasterizer/rasterizer_impl.h','cstdint'),
                 (target/'simple_knn.cu','cfloat')]
        for path,header in patches:
            if path.is_file() and f'#include <{header}>' not in path.read_text():
                path.write_text(f'#include <{header}>\n'+path.read_text())
                manifest['patches'].append({'path':str(path.relative_to(root)),'added_header':header})
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    env=dict(os.environ)
    capability=torch.cuda.get_device_capability()
    env.setdefault('TORCH_CUDA_ARCH_LIST',f'{capability[0]}.{capability[1]}')
    env.setdefault('MAX_JOBS','4')
    for name in sources:
        subprocess.run([sys.executable,'-m','pip','install','--no-build-isolation',str(root/name)],env=env,check=True)


if __name__=='__main__': main()
