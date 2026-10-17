"""Pin GLINT's gsplat fork and correct the flagged-event CUDA API spelling."""
from pathlib import Path
import subprocess
ROOT=Path('/workspace/experiments/gsplat-glint-pinned')
COMMIT='83a0dcd3f850cccdc7b92b50fd6a568e3260eae1'
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True)
if not ROOT.exists():
    ROOT.mkdir()
    git('init');git('remote','add','origin','https://github.com/youngju-na/gsplat.git')
    git('fetch','--depth','1','origin',COMMIT);git('checkout','--detach','FETCH_HEAD')
assert git('rev-parse','HEAD').strip()==COMMIT
git('submodule','update','--init','--recursive','--depth','1')
p=ROOT/'gsplat/cuda/csrc/Utils.cpp'
s=p.read_text()
for arg in ['&events[device_id]','&merge_event']:
    old=f'cudaEventCreate({arg}, cudaEventDisableTiming)'
    new=f'cudaEventCreateWithFlags({arg}, cudaEventDisableTiming)'
    assert old in s or new in s
    s=s.replace(old,new)
p.write_text(s)
print(git('diff','--stat'))
