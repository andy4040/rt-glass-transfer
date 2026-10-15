import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from scipy.spatial.transform import Rotation
from rade_initialization import geometry_arrays

def test_axis_conversion_and_filter():
    fields=['x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3','filter_3D','opacity']
    v=np.zeros(3,dtype=[(k,'f4') for k in fields])
    scales=np.array([[1,2,3],[3,1,2],[2,3,1]],dtype=float)
    for i in range(3):v[f'scale_{i}']=np.log(scales[:,i])
    v['rot_0']=1;v['filter_3D']=.1
    a=geometry_arrays(v)
    r=Rotation.from_quat(a['rotation'][:,[1,2,3,0]]).as_matrix()
    np.testing.assert_allclose(np.linalg.det(r),1,atol=1e-6)
    normals=np.eye(3)[[0,1,2]]
    np.testing.assert_allclose(np.abs(r[:,:,2]),normals,atol=1e-6)
    np.testing.assert_allclose(np.exp(a['scaling']),np.tile(np.sqrt([9.01,4.01]),(3,1)),rtol=1e-6)
    expected=.5*np.prod(scales/np.sqrt(scales**2+.01),axis=1)
    np.testing.assert_allclose(a['occupancy'][:,0],expected,rtol=1e-6)
