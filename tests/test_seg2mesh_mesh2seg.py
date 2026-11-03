"""seg2mesh and mesh2seg: meshes are (x, y, z) in Å (voxel index i at i * spacing), closed, and convert back to the
voxels they came from."""

import copick
import numpy as np
import pytest
import trimesh as tm
from click.testing import CliRunner
from copick_utils.converters.mesh_from_segmentation import label_meshes, mesh_from_segmentation, volume_to_mesh
from copick_utils.converters.segmentation_from_mesh import mesh_to_volume

VS = 10.0


def box(shape_zyx, lo_xyz, hi_xyz, value=1, volume=None):
    """A (z, y, x) volume with ``value`` in the voxels from ``lo_xyz`` to ``hi_xyz`` (inclusive)."""
    vol = np.zeros(shape_zyx, np.uint16) if volume is None else volume
    vol[lo_xyz[2] : hi_xyz[2] + 1, lo_xyz[1] : hi_xyz[1] + 1, lo_xyz[0] : hi_xyz[0] + 1] = value
    return vol


def ball(shape_zyx, centre_xyz, radius):
    z, y, x = np.indices(shape_zyx)
    cx, cy, cz = centre_xyz
    return ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < radius**2).astype(np.uint8)


def ring(shape_zyx):
    """A thick ring around the z axis (a hole through it, so inside/outside alternates along rays)."""
    z, y, x = np.indices(shape_zyx)
    r = np.hypot(x - shape_zyx[2] / 2, y - shape_zyx[1] / 2)
    return ((r > 3) & (r < 7) & (z > 1) & (z < shape_zyx[0] - 2)).astype(np.uint8)


# --- seg2mesh ----------------------------------------------------------------------------------------------------------


def test_mesh_is_xyz_in_angstrom():
    # Voxels x 3-10, y 2-5, z 1-2 of a (z, y, x) = (6, 8, 14) volume: the surface lies half a voxel outside them.
    mesh = volume_to_mesh(box((6, 8, 14), (3, 2, 1), (10, 5, 2)), VS)
    np.testing.assert_allclose(mesh.bounds, [[2.5 * VS, 1.5 * VS, 0.5 * VS], [10.5 * VS, 5.5 * VS, 2.5 * VS]])


def test_mesh_is_closed_and_faces_outward():
    mesh = volume_to_mesh(ball((14, 14, 14), (7, 7, 7), 5), VS)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume > 0
    outward = np.einsum("ij,ij->i", mesh.vertex_normals, mesh.vertices - 7 * VS)
    assert (outward > 0).all()


def test_mesh_is_closed_where_the_segmentation_touches_the_volume_edge():
    mesh = volume_to_mesh(box((6, 6, 6), (0, 0, 0), (2, 5, 5)), VS)
    assert mesh.is_watertight
    np.testing.assert_allclose(mesh.bounds, [[-0.5 * VS] * 3, [2.5 * VS, 5.5 * VS, 5.5 * VS]])


def test_label_meshes_one_per_label_in_place():
    vol = box((6, 8, 20), (1, 1, 1), (3, 3, 3), value=2)
    box((6, 8, 20), (12, 2, 1), (15, 5, 4), value=7, volume=vol)
    meshes = label_meshes(vol, VS)
    assert list(meshes) == [2, 7]
    np.testing.assert_allclose(meshes[7].bounds, [[11.5 * VS, 1.5 * VS, 0.5 * VS], [15.5 * VS, 5.5 * VS, 4.5 * VS]])
    assert all(m.is_watertight and m.volume > 0 for m in meshes.values())


# --- mesh2seg ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "volume",
    [
        pytest.param(box((6, 8, 14), (3, 2, 1), (10, 5, 2)), id="box"),
        pytest.param(ball((14, 16, 18), (9, 8, 7), 5), id="ball"),
        pytest.param(ring((8, 16, 16)), id="ring"),
        pytest.param(box((6, 6, 6), (0, 0, 0), (2, 5, 5)), id="at the volume edge"),
        pytest.param(box((8, 10, 12), (2, 2, 4), (9, 7, 4)), id="one voxel thin"),
    ],
)
def test_mesh_to_volume_gives_back_the_segmentation(volume):
    # Marching-cubes meshes have their vertices and edges on the voxel grid, where the rays were cast.
    mesh = volume_to_mesh(volume, VS)
    np.testing.assert_array_equal(mesh_to_volume(mesh, volume.shape[::-1], VS), volume > 0)


def test_voxels_are_inside_when_their_centres_are():
    # x 22-61, y 31-79, z 47-93 Å: the voxel centres inside are x 30-60, y 40-70, z 50-90.
    mesh = tm.creation.box(bounds=[[22, 31, 47], [61, 79, 93]])
    expected = np.zeros((12, 10, 10), bool)
    expected[5:10, 4:8, 3:7] = True
    np.testing.assert_array_equal(mesh_to_volume(mesh, (10, 10, 12), VS), expected)


def test_a_box_on_the_volume_edges_fills_the_volume():
    # Faces through the first voxel centres (0) and half a voxel past the last ones, like a valid-area box.
    mesh = tm.creation.box(bounds=[[0, 0, 0], [100, 100, 120]])
    assert mesh_to_volume(mesh, (10, 10, 12), VS).all()


def test_a_mesh_reaching_beyond_the_volume():
    mesh = tm.creation.box(bounds=[[-55, -55, -55], [55, 45, 65]])
    expected = np.zeros((12, 10, 10), bool)
    expected[:7, :5, :6] = True
    np.testing.assert_array_equal(mesh_to_volume(mesh, (10, 10, 12), VS), expected)


# --- end to end ------------------------------------------------------------------------------------------------------


def _invoke(command, args):
    result = CliRunner().invoke(command, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def test_seg2mesh_then_mesh2seg_round_trip(run, config_path):
    from copick_utils.cli.mesh2seg import mesh2seg
    from copick_utils.cli.seg2mesh import seg2mesh

    shape = (12, 16, 20)
    volume = ball(shape, (12, 7, 5), 4) | box(shape, (0, 10, 6), (19, 15, 8)).astype(np.uint8)
    run.new_voxel_spacing(VS, exist_ok=True)
    run.get_voxel_spacing(VS).new_tomogram("wbp").from_numpy(np.zeros(shape, np.float32))
    run.new_segmentation(VS, "membrane", "1", is_multilabel=False, user_id="test").from_numpy(volume)

    _invoke(seg2mesh, ["-c", config_path, "-i", "membrane:test/1@10.0", "-o", "membrane:mesh/1", "-w", "1"])
    (mesh,) = run.get_meshes(object_name="membrane", user_id="mesh", session_id="1")
    zyx = np.nonzero(volume)
    expected = [[(zyx[a].min() - 0.5) * VS for a in (2, 1, 0)], [(zyx[a].max() + 0.5) * VS for a in (2, 1, 0)]]
    np.testing.assert_allclose(mesh.mesh.bounds, expected)

    _invoke(mesh2seg, ["-c", config_path, "-i", "membrane:mesh/1", "-o", "membrane:back/1@10.0", "-w", "1"])
    # The CLI wrote through its own copick root: open the project again to see its output.
    run = copick.from_file(config_path).get_run(run.name)
    (back,) = run.get_segmentations(name="membrane", user_id="back", session_id="1", voxel_size=VS)
    np.testing.assert_array_equal(back.numpy() > 0, volume > 0)


def test_seg2mesh_individual_meshes(run, config_path):
    from copick_utils.cli.seg2mesh import seg2mesh

    vol = box((6, 8, 20), (1, 1, 1), (3, 3, 3), value=2)
    box((6, 8, 20), (12, 2, 1), (15, 5, 4), value=7, volume=vol)
    run.new_voxel_spacing(VS, exist_ok=True)
    run.new_segmentation(VS, "membrane", "1", is_multilabel=False, user_id="test").from_numpy(vol)

    _invoke(
        seg2mesh,
        ["-c", config_path, "-i", "membrane:test/1@10.0", "-o", "membrane:mesh/inst-{instance_id}", "-im", "-w", "1"],
    )
    meshes = {m.session_id: m for m in run.get_meshes(object_name="membrane", user_id="mesh")}
    assert sorted(meshes) == ["inst-2", "inst-7"]
    np.testing.assert_allclose(meshes["inst-2"].mesh.bounds, [[0.5 * VS] * 3, [3.5 * VS] * 3])


def test_mesh_from_segmentation_individual_meshes_without_template(run):
    vol = box((6, 8, 20), (1, 1, 1), (3, 3, 3), value=2)
    run.new_voxel_spacing(VS, exist_ok=True)
    seg = run.new_segmentation(VS, "membrane", "1", is_multilabel=False, user_id="test")
    seg.from_numpy(vol)
    mesh, stats = mesh_from_segmentation(seg, run, "membrane", "split", "mesh", individual_meshes=True)
    assert mesh.session_id == "split-2"
    assert stats["faces_created"] > 0
