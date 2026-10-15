"""CPU check of a saved RT PLY and its required shading artifacts."""
import json,sys
from pathlib import Path
import numpy as np
from plyfile import PlyData
folder=Path(sys.argv[1])
for name in ('point_cloud.ply','light_mlp.pt','dir_encoding.pt'):
    assert (folder/name).stat().st_size>0, name
v=PlyData.read(folder/'point_cloud.ply')['vertex'].data
assert len(v)>0
assert all(np.isfinite(v[k]).all() for k in v.dtype.names), 'Nonfinite saved Gaussian attribute'
result={'count':len(v),'finite_gaussian_attributes':True,'shading_files_present':True}
(folder/'validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
