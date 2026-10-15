"""Convert RaDe ellipsoids to RT surfels without transferring learned appearance.

The shortest principal axis becomes the surfel normal. The two other axes and
RaDe's baked 3D filter define tangent scales. This is a lossy 3D -> 2D conversion,
not an exact renderer conversion. All positions remain in COLMAP coordinates.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def geometry_arrays(vertex):
    xyz = np.column_stack([vertex[k] for k in ('x', 'y', 'z')]).astype(np.float32)
    scale = np.exp(np.column_stack([vertex[f'scale_{i}'] for i in range(3)]).astype(np.float64))
    quat = np.column_stack([vertex[f'rot_{i}'] for i in range(4)]).astype(np.float64)
    filt = np.asarray(vertex['filter_3D'], dtype=np.float64)[:, None]
    filtered = np.sqrt(scale**2 + filt**2)
    opacity = 1 / (1 + np.exp(-np.asarray(vertex['opacity'], dtype=np.float64)))
    occupancy = opacity * np.prod(scale / filtered, axis=1)
    order = np.argsort(-filtered, axis=1, kind='stable')
    R = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
    tangent = np.take_along_axis(R, order[:, None, :], axis=2)
    tangent[:, :, 2] = np.cross(tangent[:, :, 0], tangent[:, :, 1])
    q = Rotation.from_matrix(tangent).as_quat()[:, [3, 0, 1, 2]]
    scales2 = np.take_along_axis(filtered, order[:, :2], axis=1)
    arrays = dict(xyz=xyz, scaling=np.log(scales2).astype(np.float32),
                  rotation=q.astype(np.float32), occupancy=np.clip(occupancy, 1e-6, 1-1e-6).astype(np.float32)[:, None])
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite RaDe geometry')
    return arrays


def install_adapter():
    """Called before upstream Scene creates its initial Gaussians."""
    import os
    if not os.environ.get('RADE_INIT_PLY'):
        return
    import torch
    from plyfile import PlyData
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud
    original = GaussianModel.create_from_pcd
    path = Path(os.environ['RADE_INIT_PLY'])
    def create(self, pcd, spatial_lr_scale):
        arrays = geometry_arrays(PlyData.read(path)['vertex'].data)
        # Identical appearance source in both arms: original train-only SfM RGB.
        # Never transfer RaDe SH or GLINT positions/depths.
        _, ids = cKDTree(np.asarray(pcd.points)).query(arrays['xyz'], workers=1)
        colors = np.asarray(pcd.colors)[ids]
        original(self, BasicPointCloud(points=arrays['xyz'], colors=colors,
                                      normals=np.zeros_like(arrays['xyz'])), spatial_lr_scale)
        for name in ('scaling', 'rotation'):
            setattr(self, '_'+name, torch.nn.Parameter(torch.from_numpy(arrays[name]).cuda()))
        occ = torch.from_numpy(arrays['occupancy']).cuda()
        self._occupancy = torch.nn.Parameter(torch.logit(occ))
        record = dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      count=len(ids), conversion='two largest filtered axes as tangent; shortest axis as normal',
                      appearance='nearest original SfM RGB; SH higher orders reset',
                      material_opacity=0.5, material_transmissivity=0.5,
                      geometry_frozen=False, filtering='RaDe 3D filter baked into scales and occupancy')
        target = os.environ.get('INIT_REPORT_DIR')
        if target:
            Path(target).mkdir(parents=True, exist_ok=True)
            Path(target, 'geometry_initialization.json').write_text(json.dumps(record, indent=2))
        print('RADE_INITIALIZATION', json.dumps(record), flush=True)
    GaussianModel.create_from_pcd = create
