"""Declare FLT_MAX explicitly for CUDA 12.8; no numerical algorithm changes."""
from pathlib import Path
import os
root=Path(__file__).resolve().parents[1]
p=root/'vendor/RT-Splatting/submodules/simple-knn/simple_knn.cu'
s=p.read_text()
if '#include <cfloat>' not in s:
    assert '#include <vector>' in s
    p.write_text(s.replace('#include <vector>','#include <vector>\n#include <cfloat>',1))
print('simple-knn explicit cfloat header ready')
p=root/'vendor/RT-Splatting/submodules/diff-surfel-anych/cuda_rasterizer/rasterizer_impl.h'
s=p.read_text()
if '#include <cstdint>' not in s:
    p.write_text('#include <cstdint>\n'+s)
print('RT rasterizer explicit cstdint header ready')
rade=Path(os.environ.get('RADE_ROOT','/workspace/experiments/igs-source/submodules/RaDe-GS'))
for name in ['diff-gaussian-rasterization','diff-gaussian-rasterization-clamp','compress-diff-gaussian-rasterization']:
    glm=rade/'submodules'/name/'third_party/glm'
    if glm.is_file() and not glm.is_symlink():
        link=glm.read_text().strip()
        target=(glm.parent/link).resolve()
        target.relative_to(rade.resolve())
        assert (target/'glm/glm.hpp').is_file(), target
        glm.rename(glm.with_name('glm.windows-link'))
        glm.symlink_to(link,target_is_directory=True)
        print(name,'restored GLM symlink')
    p=rade/'submodules'/name/'cuda_rasterizer/rasterizer_impl.h'
    if p.exists():
        s=p.read_text()
        if '#include <cstdint>' not in s:p.write_text('#include <cstdint>\n'+s)
        print(name, 'explicit cstdint header ready')
