"""Regression tests for the mesh distance-field used by clippicks/clipseg.

Guards the origin-alignment fix: the rasterized mesh field must be aligned to the SAME origin the
caller uses to index pick positions, otherwise distances are meaningless.
"""

import numpy as np
import trimesh as tm
from copick_utils.logical.distance_operations import _create_distance_field_from_mesh


def _flat_sheet(size=1000.0):
    """A flat square sheet in the z=0 plane (an open mesh, like a single slab cap)."""
    v = np.array([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]], dtype=float)
    return tm.Trimesh(vertices=v, faces=np.array([[0, 1, 2], [0, 2, 3]]))


def test_distance_field_is_origin_aligned():
    sheet = _flat_sheet(1000.0)
    spacing = 20.0
    # Origin deliberately NOT equal to mesh.bounds[0] (which is [0,0,0]).
    origin = np.array([-100.0, -100.0, -300.0])
    # Grid covers x[-100,1100], y[-100,1100], z[-300,300].
    shape = (int(1200 / spacing), int(1200 / spacing), int(600 / spacing))

    df = _create_distance_field_from_mesh(sheet, shape, spacing, origin=origin)

    def dist_at(p):
        idx = np.floor((np.asarray(p) - origin) / spacing).astype(int)
        return df[idx[0], idx[1], idx[2]]

    # A point 200 A above the sheet: true distance to the surface is 200.
    assert abs(dist_at([500.0, 500.0, 200.0]) - 200.0) <= spacing
    # A point 120 A below the sheet: true distance is 120.
    assert abs(dist_at([500.0, 500.0, -120.0]) - 120.0) <= spacing
    # A point on the sheet: distance ~0.
    assert dist_at([500.0, 500.0, 0.0]) <= spacing


def test_distance_field_default_origin_is_mesh_bounds():
    # With origin=None the field falls back to mesh.bounds[0]; a point near the surface should be
    # close, confirming the rasterization itself is sound.
    sheet = _flat_sheet(400.0)
    spacing = 10.0
    shape = (40, 40, 1)
    df = _create_distance_field_from_mesh(sheet, shape, spacing)  # origin defaults to [0,0,0]
    # Surface voxels exist -> some zero-distance entries.
    assert df.min() <= spacing
    assert df.shape == shape
