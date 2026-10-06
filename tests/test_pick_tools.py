"""Pick tools keep each pick's identity and place or test it at its particle centre (location plus shift)."""

import numpy as np
import pytest
from copick_utils.converters.segmentation_from_picks import from_picks
from copick_utils.io.readers import coordinates
from copick_utils.logical.distance_operations import limit_picks_by_distance
from copick_utils.logical.point_operations import picks_exclusion_by_mesh, picks_inclusion_by_mesh
from scipy.spatial.transform import Rotation

VS = 10.0


def _transforms(n, shifts=None, seed=0):
    rng = np.random.default_rng(seed)
    t = np.tile(np.eye(4), (n, 1, 1))
    t[:, :3, :3] = Rotation.random(n, random_state=rng).as_matrix()
    if shifts is not None:
        t[:, :3, 3] = shifts
    return t


def _picks(run, positions, transforms, ids, scores, session="in"):
    picks = run.new_picks("microtubule", session, "test", exist_ok=True)
    picks.from_numpy(np.asarray(positions, float), transforms, instance_ids=np.asarray(ids), scores=np.asarray(scores))
    return picks


def _segmentation(run, array, name="membrane", session="ref", **kwargs):
    run.new_voxel_spacing(VS, exist_ok=True)
    seg = run.new_segmentation(VS, name, session, user_id="test", exist_ok=True, **kwargs)
    seg.from_numpy(array)
    return seg


def _state(points):
    return [
        (
            (p.location.x, p.location.y, p.location.z),
            np.round(np.asarray(p.transformation), 9).tolist(),
            p.instance_id,
            p.score,
        )
        for p in points
    ]


@pytest.fixture
def box_reference(run):
    """A segmentation whose voxels x, y, z in [0, 10) are inside (shape z=12, y=14, x=16, not cubic)."""
    array = np.zeros((12, 14, 16), dtype=np.uint8)
    array[:10, :10, :10] = 1
    return _segmentation(run, array)


def test_picksin_keeps_identity_order_and_tests_centres(run, box_reference):
    # Five picks along a filament; the third has its location outside the box but its centre inside, the fourth
    # the opposite. Positions are in Angstrom.
    positions = np.array([[10, 10, 10], [30, 30, 30], [120, 50, 50], [50, 50, 50], [80, 80, 80]], float)
    shifts = np.zeros((5, 3))
    shifts[2] = [-80, 0, 0]  # centre (40, 50, 50): inside
    shifts[3] = [100, 0, 0]  # centre (150, 50, 50): outside
    transforms = _transforms(5, shifts)
    picks = _picks(run, positions, transforms, ids=[1, 1, 1, 1, 2], scores=[0.9, 0.8, 0.7, 0.6, 0.5])

    out, stats = picks_inclusion_by_mesh(
        picks,
        run,
        "microtubule",
        "kept",
        "test",
        reference_segmentation=box_reference,
    )

    expected = [p for i, p in enumerate(picks.points) if i in (0, 1, 2, 4)]
    assert stats["points_created"] == 4
    assert _state(out.points) == _state(expected)

    out, stats = picks_exclusion_by_mesh(
        picks,
        run,
        "microtubule",
        "dropped",
        "test",
        reference_segmentation=box_reference,
    )
    assert _state(out.points) == _state([picks.points[3]])


def test_empty_result_replaces_an_earlier_output(run, box_reference):
    picks = _picks(run, [[10, 10, 10]], _transforms(1), ids=[3], scores=[1.0])
    picks_inclusion_by_mesh(picks, run, "microtubule", "result", "test", reference_segmentation=box_reference)
    assert len(run.get_picks("microtubule", "test", "result")[0].points) == 1

    # Every pick is now inside, so nothing survives exclusion: the earlier result must not be left behind.
    out, stats = picks_exclusion_by_mesh(
        picks,
        run,
        "microtubule",
        "result",
        "test",
        reference_segmentation=box_reference,
    )
    assert stats["points_created"] == 0
    run.refresh_picks()
    stored = run.get_picks("microtubule", "test", "result")
    assert len(stored) == 1 and len(stored[0].points) == 0


def test_instance_reference_counts_every_instance_as_inside(run):
    array = np.zeros((12, 14, 16), dtype=np.uint16)
    array[2:4, 2:4, 2:4] = 1
    array[6:8, 6:8, 6:8] = 7
    instances = _segmentation(run, array, name="microtubule", session="inst", is_instance=True)
    picks = _picks(run, [[30, 30, 30], [70, 70, 70], [100, 100, 100]], _transforms(3), ids=[1, 2, 3], scores=[1, 1, 1])
    out, stats = picks_inclusion_by_mesh(picks, run, "microtubule", "kept", "test", reference_segmentation=instances)
    assert [p.instance_id for p in out.points] == [1, 2]


def test_clippicks_reference_segmentation_is_indexed_zyx(run):
    # Non-cubic reference (z=4, y=20, x=60) with one voxel set at x=50, y=10, z=2. With x/z swapped, the pick near
    # it would index out of bounds and be dropped.
    array = np.zeros((4, 20, 60), dtype=np.uint8)
    array[2, 10, 50] = 1
    reference = _segmentation(run, array, session="clip")
    positions = [[500, 100, 20], [100, 100, 20]]  # 0 and 400 Angstrom from the voxel
    shifts = np.array([[0, 0, 0], [380, 0, 0]], float)  # the second's centre is 20 Angstrom away
    picks = _picks(run, positions, _transforms(2, shifts), ids=[4, 5], scores=[0.3, 0.4])

    out, stats = limit_picks_by_distance(
        picks,
        run,
        "microtubule",
        "clipped",
        "test",
        reference_segmentation=reference,
        max_distance=50.0,
    )
    assert stats["points_created"] == 2
    assert _state(out.points) == _state(picks.points)

    out, stats = limit_picks_by_distance(
        picks,
        run,
        "microtubule",
        "far",
        "test",
        reference_segmentation=reference,
        max_distance=50.0,
        invert=True,
    )
    assert stats["points_created"] == 0 and len(out.points) == 0


def test_painting_and_coordinates_use_centres(run):
    picks = _picks(run, [[30, 40, 50]], _transforms(1, np.array([[20, 0, -10]])), ids=[1], scores=[1.0])
    volume = from_picks(picks, np.zeros((10, 10, 10), np.uint8), radius=5.0, label_value=4, voxel_spacing=VS)
    assert volume[4, 4, 5] == 4  # centre (50, 40, 40) Angstrom -> voxel z=4, y=4, x=5
    assert volume[5, 4, 3] == 0  # location (30, 40, 50) is not painted

    zyx = coordinates(run, "microtubule", "test", "in", voxel_size=VS)
    np.testing.assert_allclose(zyx, [[4.0, 4.0, 5.0]])
