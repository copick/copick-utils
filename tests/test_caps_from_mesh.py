"""Unit tests for the pure slab cap-extraction geometry (no copick I/O required)."""

import numpy as np
import pytest
import trimesh as tm

from copick_utils.converters.caps_from_mesh import extract_slab_caps

COS45 = np.cos(np.deg2rad(45.0))


def _thin_box():
    """A thin, axis-aligned slab box: x=100, y=80, z=12 (z is the slab normal)."""
    return tm.creation.box(extents=(100.0, 80.0, 12.0))


def test_axis_aligned_box_both_caps():
    box = _thin_box()  # 12 triangular faces: 2 top + 2 bottom + 8 side-wall
    caps = extract_slab_caps(box, axis="z", angle_threshold=45.0, which="both")
    assert caps is not None
    assert caps.faces.shape[0] == 4  # the 8 side-wall faces are dropped
    # Every selected face is near-horizontal (normal close to +/- z).
    assert np.all(np.abs(np.asarray(caps.face_normals)[:, 2]) >= COS45)


def test_axis_aligned_top_and_bottom_split():
    box = _thin_box()
    top = extract_slab_caps(box, axis="z", which="top")
    bottom = extract_slab_caps(box, axis="z", which="bottom")
    assert top is not None and bottom is not None
    assert top.faces.shape[0] == 2
    assert bottom.faces.shape[0] == 2
    # Top faces sit above bottom faces along z.
    assert top.triangles_center[:, 2].min() > bottom.triangles_center[:, 2].max()
    # Normal sign matches the side.
    assert np.all(np.asarray(top.face_normals)[:, 2] > 0)
    assert np.all(np.asarray(bottom.face_normals)[:, 2] < 0)


def test_tilted_box_still_finds_caps():
    box = _thin_box()
    box.apply_transform(tm.transformations.rotation_matrix(np.deg2rad(30.0), [1, 0, 0]))
    caps = extract_slab_caps(box, axis="z", angle_threshold=45.0, which="both")
    assert caps is not None
    # A 30-degree tilt keeps caps within the 45-degree threshold and walls outside it.
    assert caps.faces.shape[0] == 4
    # Normal-sign labelling survives tilt where centroid-z would be marginal.
    top = extract_slab_caps(box, axis="z", which="top")
    bottom = extract_slab_caps(box, axis="z", which="bottom")
    assert top.triangles_center[:, 2].mean() > bottom.triangles_center[:, 2].mean()


def test_tight_threshold_rejects_tilted_caps():
    box = _thin_box()
    box.apply_transform(tm.transformations.rotation_matrix(np.deg2rad(30.0), [1, 0, 0]))
    # Caps are now 30 degrees off z; a 5-degree threshold should find none.
    assert extract_slab_caps(box, axis="z", angle_threshold=5.0, which="both") is None


def test_axis_is_configurable():
    box = _thin_box()
    # With axis=x the "caps" are the +/-x faces (4 triangles), proving axis configurability and
    # documenting why the default must be z for a slab whose normal is z.
    caps = extract_slab_caps(box, axis="x", angle_threshold=45.0, which="both")
    assert caps is not None
    assert caps.faces.shape[0] == 4
    assert np.all(np.abs(np.asarray(caps.face_normals)[:, 0]) >= COS45)


def test_auto_axis_infers_slab_normal():
    # Slab normal is the thinnest extent (here y=8); auto-axis should pick it without being told.
    box = tm.creation.box(extents=(100.0, 8.0, 90.0))
    caps = extract_slab_caps(box, angle_threshold=45.0, which="both", auto_axis=True)
    assert caps is not None
    assert caps.faces.shape[0] == 4
    assert np.all(np.abs(np.asarray(caps.face_normals)[:, 1]) >= COS45)


def test_degenerate_faces_excluded():
    # A flat cap sheet (2 good triangles) plus a zero-area face; the degenerate face must not appear.
    verts = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 1]])  # last face is degenerate
    mesh = tm.Trimesh(vertices=verts, faces=faces, process=False)
    caps = extract_slab_caps(mesh, axis="z", angle_threshold=45.0, which="both")
    assert caps is not None
    assert caps.faces.shape[0] == 2


def test_retriangulated_boolean_input():
    # The real valid-sample mesh is a boolean intersection (re-triangulated), so any original
    # face ordering is gone. The geometric classifier must still recover clean caps.
    a = tm.creation.box(extents=(100.0, 80.0, 12.0))
    b = tm.creation.box(extents=(60.0, 60.0, 40.0))
    try:
        inter = a.intersection(b)
    except BaseException as e:  # pragma: no cover - depends on a boolean backend
        pytest.skip(f"trimesh boolean backend unavailable: {e}")
    if inter is None or inter.faces.shape[0] == 0:
        pytest.skip("boolean intersection produced no geometry")

    caps = extract_slab_caps(inter, axis="z", angle_threshold=45.0, which="both")
    assert caps is not None
    assert caps.faces.shape[0] < inter.faces.shape[0]  # walls were dropped
    assert np.all(np.abs(np.asarray(caps.face_normals)[:, 2]) >= COS45)
    # Caps span the slab thickness (z-extent ~ 12), not the full box.
    z_extent = caps.bounds[1][2] - caps.bounds[0][2]
    assert np.isclose(z_extent, 12.0, atol=1e-3)


def test_caps_are_open_but_voxelizable():
    # The property clippicks relies on: an open mesh is not watertight, yet still rasterizes to a
    # populated voxel grid (surface voxelization), so it works as a distance-field reference.
    box = _thin_box()
    caps = extract_slab_caps(box, axis="z", which="both")
    assert caps is not None
    assert caps.is_watertight is False
    matrix = caps.voxelized(pitch=4.0).matrix
    assert matrix.sum() > 0


def test_no_faces_returns_none():
    empty = tm.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int), process=False)
    assert extract_slab_caps(empty, axis="z", which="both") is None


def test_invalid_arguments():
    box = _thin_box()
    with pytest.raises(ValueError):
        extract_slab_caps(box, which="sideways")
    with pytest.raises(ValueError):
        extract_slab_caps(box, axis="w")
    with pytest.raises(ValueError):
        extract_slab_caps(box, angle_threshold=120.0)
