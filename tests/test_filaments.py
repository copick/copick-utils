"""Filament tracing: seg2fil, fil2picks, fil2seg, and the fit-spline and skeletonize fixes, on synthetic tubes."""

import numpy as np
import pytest
from click.testing import CliRunner
from copick.util.filaments import curve_is_current, evaluate_curve
from copick.util.relion import filament_track_lengths
from copick_utils.cli.fil2picks import fil2picks
from copick_utils.cli.fil2seg import fil2seg
from copick_utils.cli.fit_spline import fit_spline
from copick_utils.cli.seg2fil import seg2fil
from copick_utils.cli.skeletonize import skeletonize
from copick_utils.process.filament_tracing import (
    TraceParameters,
    evaluate_curve_with_tangents,
    evaluate_spline,
    sample_curve,
    trace_centrelines,
)

VS = 10.0
SHAPE = (40, 80, 80)


def tube(volume, start, end, radius, value=1):
    """Paint a straight tube from ``start`` to ``end`` (z, y, x voxels)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    zz, yy, xx = np.indices(volume.shape)
    grid = np.stack([zz, yy, xx], -1).astype(float)
    d = end - start
    t = np.clip(((grid - start) @ d) / (d @ d), 0, 1)
    volume[np.linalg.norm(grid - (start + t[..., None] * d), axis=-1) <= radius] = value
    return volume


def arc(volume, radius_px=25, tube_radius=3):
    zz, yy, xx = np.indices(volume.shape)
    r = np.hypot(yy - 40, xx - 40)
    volume[(np.abs(r - radius_px) <= tube_radius) & (np.abs(zz - 20) <= tube_radius) & (xx >= 40)] = 1
    return volume


def _invoke(command, args):
    result = CliRunner().invoke(command, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def _write_seg(run, array, name="microtubule", session="s", **kwargs):
    run.new_voxel_spacing(VS, exist_ok=True)
    seg = run.new_segmentation(VS, name, session, user_id="test", exist_ok=True, **kwargs)
    seg.from_numpy(array)
    return seg


def _bspline(c):
    knots, coefficients, k = c.tck
    return {"kind": "bspline", "knots": knots, "control_points": np.stack(coefficients, 1), "degree": k, "step": VS}


# --- the engine --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, build, expected",
    [
        ("straight", lambda v: tube(v, (20, 40, 10), (20, 40, 70), 3), 1),
        ("along y", lambda v: tube(v, (20, 5, 40), (20, 75, 40), 3), 1),
        ("arc", arc, 1),
        (
            "crossing in one component",
            lambda v: tube(tube(v, (20, 40, 5), (20, 40, 75), 3), (20, 5, 40), (20, 75, 40), 3),
            2,
        ),
        ("two tubes", lambda v: tube(tube(v, (20, 20, 5), (20, 20, 75), 3), (20, 60, 5), (20, 60, 75), 3), 2),
    ],
)
def test_tracing_finds_each_filament(name, build, expected):
    volume = build(np.zeros(SHAPE, np.uint8))
    centrelines, assignment, report = trace_centrelines(volume, VS, TraceParameters(min_length=150))
    assert len(centrelines) == expected
    assert [c.instance_id for c in centrelines] == list(range(1, expected + 1))
    assert sorted(np.unique(assignment[volume > 0]).tolist()) == list(range(1, expected + 1))
    assert np.all(assignment[volume == 0] == 0)
    for c in centrelines:
        assert c.score > 0.95  # the fitted centreline stays inside the label
        assert abs(c.radius - 30.0) < 6.0


@pytest.mark.parametrize("seed", [0, 2, 6, 8])
def test_a_thick_end_does_not_turn_the_filament_back(seed):
    # A filament ending in a thick, rough blob (a splayed microtubule end on 10521) skeletonizes to a tangle of small
    # loops that merge into one large junction; continuing "straight" through it must not leave by a branch that
    # heads back alongside the filament.
    volume = tube(np.zeros((40, 80, 100), np.uint8), (20, 40, 5), (20, 40, 70), 3.5)
    zz, yy, xx = np.indices(volume.shape)
    ellipsoid = ((zz - 20) / 6.5) ** 2 + ((yy - 41) / 9) ** 2 + ((xx - 76) / 13) ** 2
    volume[ellipsoid + np.random.default_rng(seed).normal(0, 0.15, volume.shape) <= 1] = 1
    centrelines, _, _ = trace_centrelines(volume, VS, TraceParameters(min_radius=20.0))
    assert len(centrelines) == 1
    points, tangents, _ = evaluate_spline(centrelines[0].tck, step=VS / 2)
    overall = (points[-1] - points[0]) / np.linalg.norm(points[-1] - points[0])
    assert np.degrees(np.arccos(np.clip(tangents @ overall, -1, 1))).max() < 45.0


def test_a_tube_labelled_by_its_wall_traces_as_one_filament():
    zz, yy, xx = np.indices(SHAPE)
    r = np.hypot(zz - 20, yy - 40)
    volume = ((r <= 6) & (r >= 3) & (xx >= 10) & (xx <= 70)).astype(np.uint8)  # an empty lumen
    centrelines, _, _ = trace_centrelines(volume, VS, TraceParameters(fill_lumen=60.0, min_radius=40.0))
    assert len(centrelines) == 1 and centrelines[0].length > 550 and centrelines[0].radius > 40.0
    unfilled, _, _ = trace_centrelines(volume, VS, TraceParameters(min_radius=40.0))
    assert len(unfilled) != 1 or unfilled[0].length < 550


def test_specks_blobs_and_thin_slivers_are_rejected():
    volume = tube(np.zeros(SHAPE, np.uint8), (20, 40, 10), (20, 40, 70), 3)
    volume[2:4, 2:4, 2:4] = 1  # speck
    volume[28:38, 60:72, 60:72] = 1  # compact blob
    tube(volume, (5, 10, 10), (5, 10, 60), 0.6)  # thin sliver, label radius ~1 voxel
    centrelines, _, report = trace_centrelines(volume, VS, TraceParameters(min_radius=20.0))
    assert len(centrelines) == 1 and centrelines[0].length > 500
    assert report.rejected_length + report.rejected_aspect >= 2 and report.rejected_radius == 1


def test_y_junction_continues_the_straightest_pair():
    volume = tube(np.zeros(SHAPE, np.uint8), (20, 40, 5), (20, 40, 75), 3)  # straight through
    tube(volume, (20, 40, 40), (20, 70, 55), 3)  # side branch at ~63 degrees
    centrelines, _, report = trace_centrelines(volume, VS, TraceParameters(min_length=150))
    lengths = sorted(c.length for c in centrelines)
    assert len(centrelines) == 2 and lengths[-1] > 650  # the through-filament is not split at the junction
    assert report.junctions_resolved == 1


def test_ring_is_one_closed_filament():
    volume = np.zeros(SHAPE, np.uint8)
    zz, yy, xx = np.indices(SHAPE)
    volume[(np.abs(np.hypot(yy - 40, xx - 40) - 25) <= 3) & (np.abs(zz - 20) <= 3)] = 1
    centrelines, _, _ = trace_centrelines(volume, VS)
    assert len(centrelines) == 1 and centrelines[0].metadata["closed"]
    assert abs(centrelines[0].length - 2 * np.pi * 250) < 60


def test_sampling_spacing_frames_and_track_lengths():
    centrelines, _, _ = trace_centrelines(tube(np.zeros(SHAPE, np.uint8), (20, 5, 40), (20, 75, 40), 3), VS)
    curve = _bspline(centrelines[0])
    positions, rotations, along = sample_curve(curve, 82.0)
    np.testing.assert_allclose(np.diff(along), 82.0)
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-9)
    # +Z is the tangent in point order (the filament starts at its smaller-y end)
    np.testing.assert_allclose(rotations[:, :, 2], np.tile([0.0, 1.0, 0.0], (len(rotations), 1)), atol=1e-3)
    # The frames along Y (the tilt axis) never jump
    x = rotations[:, :, 0]
    assert np.max(np.degrees(np.arccos(np.clip(np.sum(x[1:] * x[:-1], axis=1), -1, 1)))) < 1.0
    np.testing.assert_allclose(filament_track_lengths(positions, np.ones(len(positions), int)), along - along[0])


def test_bspline_tangent_matches_the_curve_direction():
    centrelines, _, _ = trace_centrelines(arc(np.zeros(SHAPE, np.uint8)), VS)
    curve = _bspline(centrelines[0])
    points, tangents, _ = evaluate_curve_with_tangents(curve, 1.0)
    chords = np.diff(points, axis=0)
    chords /= np.linalg.norm(chords, axis=1, keepdims=True)
    assert np.min(np.sum(chords * tangents[:-1], axis=1)) > 0.999


# --- the commands ------------------------------------------------------------------------------------------------------


def _traced(run, config_path, volume):
    _write_seg(run, volume)
    _invoke(
        seg2fil,
        [
            "-c",
            config_path,
            "-i",
            "microtubule:test/s@10.0",
            "-o",
            "microtubule:trace/s",
            "--instances",
            "microtubule:trace/s@10.0?instance=true",
            "--min-length",
            "150",
        ],
    )
    run.refresh()
    filaments = run.get_filaments("microtubule", "trace", "s")[0]
    instances = run.get_segmentations(name="microtubule", user_id="trace", session_id="s", is_instance=True)[0]
    return filaments, instances


def test_ids_are_shared_by_instances_filaments_and_picks(run, config_path):
    volume = tube(tube(np.zeros(SHAPE, np.uint8), (20, 40, 5), (20, 40, 75), 3), (20, 5, 40), (20, 75, 40), 3)
    filaments, instances = _traced(run, config_path, volume)
    ids = sorted(f.instance_id for f in filaments.filaments)
    assert ids == [1, 2]
    assert sorted(np.unique(instances.numpy()[volume > 0]).tolist()) == ids

    # Each stored curve is the exact fit, and copick's points are its evaluation
    for f in filaments.filaments:
        assert f.curve.kind == "bspline" and f.curve_is_current()
        regenerated = evaluate_curve(
            f.curve.control_points,
            f.curve.step,
            "bspline",
            None,
            f.curve.degree,
            f.curve.knots,
        )
        np.testing.assert_allclose(np.asarray(f.points), regenerated, atol=1e-6)
        assert f.metadata["fit"]["method"] == "splprep" and f.metadata["fit"]["tool"] == "copick-utils seg2fil"

    _invoke(fil2picks, ["-c", config_path, "-i", "microtubule:trace/s", "-o", "microtubule:trace/s", "--spacing", "82"])
    run.refresh()
    picks = run.get_picks("microtubule", "trace", "s")[0]
    pick_ids = picks.instance_ids()
    assert sorted(set(pick_ids.tolist())) == ids
    assert np.all(np.diff(pick_ids) >= 0)  # grouped by filament
    # every pick sits inside its own filament's voxels
    zyx = np.round(picks.full_positions()[:, ::-1] / VS).astype(int)
    assert np.all(instances.numpy()[tuple(zyx.T)] == pick_ids)


def test_retracing_the_instance_segmentation_reproduces_and_honours_edits(run, config_path):
    volume = tube(tube(np.zeros(SHAPE, np.uint8), (20, 20, 5), (20, 20, 75), 3), (20, 60, 5), (20, 60, 75), 3)
    filaments, instances = _traced(run, config_path, volume)
    first = {f.instance_id: np.asarray(f.points) for f in filaments.filaments}

    _invoke(
        seg2fil,
        ["-c", config_path, "-i", "microtubule:trace/s@10.0?instance=true", "-o", "microtubule:retrace/s"],
    )
    run.refresh()
    again = {f.instance_id: np.asarray(f.points) for f in run.get_filaments("microtubule", "retrace", "s")[0].filaments}
    assert set(again) == set(first)
    for fid in first:
        assert np.max(np.min(np.linalg.norm(again[fid][:, None] - first[fid][None], axis=-1), axis=1)) < VS

    # An edit (renumber filament 2 to 7, delete filament 1) is what the re-trace returns
    edited = instances.numpy().astype(np.uint16)
    edited[edited == 1] = 0
    edited[edited == 2] = 7
    _write_seg(run, edited, session="edited", is_instance=True)
    _invoke(
        seg2fil,
        ["-c", config_path, "-i", "microtubule:test/edited@10.0?instance=true", "-o", "microtubule:retrace/edited"],
    )
    run.refresh()
    assert [f.instance_id for f in run.get_filaments("microtubule", "retrace", "edited")[0].filaments] == [7]


def test_hand_traced_and_plain_filaments_sample_smoothly(run, config_path):
    run.new_voxel_spacing(VS, exist_ok=True)
    clicks = np.array([[100.0, 100.0, 200.0], [300.0, 250.0, 200.0], [500.0, 300.0, 210.0], [700.0, 260.0, 220.0]])
    hand = run.new_filaments("microtubule", "hand", "test", exist_ok=True)
    hand.from_control_points([clicks], voxel_spacing=VS)
    plain = run.new_filaments("microtubule", "plain", "test", exist_ok=True)
    plain.from_numpy([np.asarray(hand.filaments[0].points)], voxel_spacing=VS)

    for session in ("hand", "plain"):
        _invoke(
            fil2picks,
            [
                "-c",
                config_path,
                "-i",
                f"microtubule:test/{session}",
                "-o",
                f"microtubule:picks/{session}",
                "--spacing",
                "40",
            ],
        )
        run.refresh()
        picks = run.get_picks("microtubule", "picks", session)[0]
        _, transforms = picks.numpy()
        z = transforms[:, :3, 2]
        chords = np.diff(picks.full_positions(), axis=0)
        chords /= np.linalg.norm(chords, axis=1, keepdims=True)
        assert np.min(np.sum(z[:-1] * chords, axis=1)) > 0.95  # +Z follows the filament
        assert np.min(np.sum(z[1:] * z[:-1], axis=1)) > 0.95  # and turns smoothly


def test_reversed_filament_gives_reversed_picks():
    centrelines, _, _ = trace_centrelines(tube(np.zeros(SHAPE, np.uint8), (20, 40, 10), (20, 40, 70), 3), VS)
    forward = _bspline(centrelines[0])
    knots = np.asarray(forward["knots"])
    backward = dict(forward, knots=(knots[0] + knots[-1] - knots[::-1]), control_points=forward["control_points"][::-1])
    p1, r1, _ = sample_curve(forward, 82.0)
    p2, r2, _ = sample_curve(backward, 82.0)
    np.testing.assert_allclose(p1, p2[::-1], atol=1e-6)
    np.testing.assert_allclose(r1[:, :, 2], -r2[::-1, :, 2], atol=1e-6)


def test_fil2seg_paints_tubes_that_retrace_to_the_filaments(run, config_path):
    volume = tube(np.zeros(SHAPE, np.uint8), (20, 40, 10), (20, 40, 70), 3)
    filaments, _ = _traced(run, config_path, volume)
    run.get_voxel_spacing(VS).new_tomogram("wbp").from_numpy(np.zeros(SHAPE, np.float32))
    _invoke(fil2seg, ["-c", config_path, "-i", "microtubule:trace/s", "-o", "microtubule:tubes/s@10.0"])
    run.refresh()
    painted = run.get_segmentations(name="microtubule", user_id="tubes", session_id="s", is_instance=True)[0].numpy()
    assert sorted(np.unique(painted).tolist()) == [0, 1]
    centrelines, _, _ = trace_centrelines(painted, VS, instances=True)
    original = np.asarray(filaments.filaments[0].points)
    # Every original point lies on the re-traced centreline (which runs on into the tube's rounded caps)
    assert np.max(np.min(np.linalg.norm(original[:, None] - centrelines[0].points[None], axis=-1), axis=1)) < VS


# --- the kept commands -------------------------------------------------------------------------------------------------


def test_fit_spline_fits_every_filament_and_can_store_them(run, config_path):
    volume = tube(tube(np.zeros(SHAPE, np.uint8), (20, 20, 5), (20, 20, 75), 3), (20, 60, 5), (20, 60, 75), 3)
    _write_seg(run, volume)
    _invoke(
        fit_spline,
        [
            "-c",
            config_path,
            "-i",
            "microtubule:test/s@10.0",
            "-o",
            "microtubule:spline/s",
            "--spacing-distance",
            "8",
            "--filaments",
            "microtubule:spline/s",
        ],
    )
    run.refresh()
    picks = run.get_picks("microtubule", "spline", "s")[0]
    ids = picks.instance_ids()
    assert sorted(set(ids.tolist())) == [1, 2]
    for fid in (1, 2):
        np.testing.assert_allclose(np.linalg.norm(np.diff(picks.full_positions()[ids == fid], axis=0), axis=1), 80.0)
    filaments = run.get_filaments("microtubule", "spline", "s")[0]
    assert len(filaments.filaments) == 2
    for f in filaments.filaments:
        assert f.curve.kind == "bspline" and f.metadata["fit"]["coordinates"] == "voxels"
        assert curve_is_current(
            f.points,
            f.curve.control_points,
            f.curve.step,
            "bspline",
            f.curve.degree,
            f.curve.knots,
        )
        # the picks lie on the stored curve
        points = np.asarray(f.points)
        mine = picks.full_positions()[ids == f.instance_id]
        assert np.max(np.min(np.linalg.norm(mine[:, None] - points[None], axis=-1), axis=1)) < 1.0


def test_skeletonize_prunes_spurs_and_keeps_the_output_name(run, config_path):
    volume = np.zeros(SHAPE, np.uint8)
    volume[20, 40, 5:75] = 1
    volume[20, 41:44, 30] = 1  # a 3-voxel spur
    _write_seg(run, volume)
    _invoke(
        skeletonize,
        [
            "-c",
            config_path,
            "-i",
            "microtubule:test/s@10.0",
            "-o",
            "membrane:skel/s@10.0",
            "--prune-length",
            "50",
            "--keep-noise",
        ],
    )
    run.refresh()
    out = run.get_segmentations(name="membrane", user_id="skel", session_id="s")
    assert len(out) == 1
    skeleton = out[0].numpy() > 0
    # Thinning moves the junction one voxel into the spur; pruning removes the rest of the spur
    assert skeleton[20, 42:44, 30].sum() == 0
    from scipy.ndimage import label

    assert label(skeleton, structure=np.ones((3, 3, 3)))[1] == 1 and skeleton.sum() >= 68
