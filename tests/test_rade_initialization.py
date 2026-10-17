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


def make_vertices(scales, rotations, filtering=0.0):
    fields=['x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3','filter_3D','opacity']
    v=np.zeros(len(scales),dtype=[(k,'f4') for k in fields])
    for i in range(3):v[f'scale_{i}']=np.log(scales[:,i])
    quats=rotations.as_quat()[:,[3,0,1,2]]
    for i in range(4):v[f'rot_{i}']=quats[:,i]
    v['filter_3D']=filtering
    return v


def test_rotated_axes_preserve_world_tangent_covariance():
    rng=np.random.default_rng(17)
    rotations=Rotation.random(40,random_state=rng)
    scales=np.exp(rng.uniform(-4,2,(40,3)))
    v=make_vertices(scales,rotations,filtering=0.03)
    v['x']=rng.normal(size=40);v['y']=rng.normal(size=40);v['z']=rng.normal(size=40)
    a=geometry_arrays(v)
    result_r=Rotation.from_quat(a['rotation'][:,[1,2,3,0]]).as_matrix()
    source_r=Rotation.from_quat(np.column_stack([v[f'rot_{i}'] for i in (1,2,3,0)])).as_matrix()
    source_scales=np.exp(np.column_stack([v[f'scale_{i}'] for i in range(3)]).astype(float))
    filtered_sq=source_scales**2+float(v['filter_3D'][0])**2
    shortest=np.argmin(filtered_sq,axis=1)
    for i in range(len(v)):
        normal=source_r[i,:,shortest[i]]
        covariance=source_r[i] @ np.diag(filtered_sq[i]) @ source_r[i].T
        target=covariance-filtered_sq[i,shortest[i]]*np.outer(normal,normal)
        actual=result_r[i,:,:2] @ np.diag(np.exp(a['scaling'][i].astype(float))**2) @ result_r[i,:,:2].T
        np.testing.assert_allclose(actual,target,rtol=3e-5,atol=1e-6)
        np.testing.assert_allclose(abs(result_r[i,:,2] @ normal),1,atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(result_r),1,atol=1e-6)
    np.testing.assert_array_equal(a['xyz'],np.column_stack([v[k] for k in ('x','y','z')]))
    assert a['scaling'].shape==(40,2) and a['rotation'].shape==(40,4) and a['occupancy'].shape==(40,1)


def test_equal_scales_are_deterministic_without_claiming_unique_surface_normal():
    rotations=Rotation.from_euler('xyz',[[.3,.4,.5],[.9,-.2,.1]])
    v=make_vertices(np.ones((2,3)),rotations)
    a=geometry_arrays(v)
    b=geometry_arrays(v)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_allclose(a['occupancy'],.5,atol=1e-6)


def test_invalid_quaternion_and_nonfinite_position_are_rejected():
    v=make_vertices(np.ones((1,3)),Rotation.identity(1))
    v['x']=np.nan
    try:
        geometry_arrays(v)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite position was accepted')
    v['x']=0
    for i in range(4):v[f'rot_{i}']=0
    try:
        geometry_arrays(v)
    except ValueError:
        pass
    else:
        raise AssertionError('Zero quaternion was accepted')
