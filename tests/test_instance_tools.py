"""Instance and panoptic segmentations through the component tools, seg-stats, combine and split."""

import csv

import numpy as np
from click.testing import CliRunner
from copick_utils.cli.combine_labels import combine
from copick_utils.cli.filter_components import filter_components
from copick_utils.cli.seg_stats import seg_stats
from copick_utils.cli.separate_components import separate_components
from copick_utils.cli.split_labels import split
from copick_utils.converters.lazy_converter import discover_tasks_for_run
from copick_utils.io import readers
from copick_utils.util.config_models import SelectorConfig

VS = 10.0
SHAPE = (20, 40, 60)


def _write(run, name, session, array, user="test", **kwargs):
    run.new_voxel_spacing(VS, exist_ok=True)
    seg = run.new_segmentation(VS, name, session, user_id=user, exist_ok=True, **kwargs)
    seg.from_numpy(array)
    return seg


def _get(run, name, user, session, **kwargs):
    run.refresh()
    found = run.get_segmentations(name=name, user_id=user, session_id=session, voxel_size=VS, **kwargs)
    assert len(found) == 1, found
    return found[0]


def _invoke(command, args):
    result = CliRunner().invoke(command, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def _three_blobs():
    volume = np.zeros(SHAPE, dtype=np.uint8)
    volume[2:6, 2:6, 2:6] = 1  # 64 voxels
    volume[10:16, 10:16, 10:16] = 1  # 216 voxels
    volume[2:4, 30:32, 50:52] = 1  # 8 voxels
    return volume


def test_separate_components_writes_one_instance_segmentation(run, config_path):
    _write(run, "microtubule", "s1", _three_blobs())
    _invoke(
        separate_components,
        [
            "-c",
            config_path,
            "-i",
            "microtubule:test/s1@10.0",
            "--binary",
            "-o",
            "microtubule:components/s1@10.0?instance=true",
        ],
    )
    instances = _get(run, "microtubule", "components", "s1", is_instance=True).numpy()
    assert sorted(np.unique(instances).tolist()) == [0, 1, 2, 3]
    assert (instances == 1).sum() == 216 and (instances == 2).sum() == 64 and (instances == 3).sum() == 8


def test_filter_instances_keeps_ids_and_skeleton_length(run, config_path):
    instances = np.zeros(SHAPE, dtype=np.uint16)
    instances[2:6, 2:6, 2:6] = 4  # compact blob, 64 voxels
    instances[10:13, 10:13, 5:55] = 7  # rod, 450 voxels, skeleton ~45 voxels
    instances[2:4, 30:32, 50:52] = 9  # speck, 8 voxels
    _write(run, "microtubule", "s2", instances, is_instance=True)

    _invoke(
        filter_components,
        [
            "-c",
            config_path,
            "-i",
            "microtubule:test/s2@10.0?instance=true",
            "-o",
            "microtubule:filtered/s2@10.0",
            "--min-size",
            "20",
            "--size-unit",
            "voxel",
        ],
    )
    kept = _get(run, "microtubule", "filtered", "s2", is_instance=True).numpy()
    assert sorted(np.unique(kept).tolist()) == [0, 4, 7]

    _invoke(
        filter_components,
        [
            "-c",
            config_path,
            "-i",
            "microtubule:test/s2@10.0?instance=true",
            "-o",
            "microtubule:long/s2@10.0",
            "--min-skeleton-length",
            "200",
        ],
    )
    long_only = _get(run, "microtubule", "long", "s2", is_instance=True).numpy()
    assert sorted(np.unique(long_only).tolist()) == [0, 7]


def test_filter_multilabel_keeps_labels(run, config_path):
    labels = np.zeros(SHAPE, dtype=np.uint8)
    labels[2:6, 2:6, 2:6] = 1
    labels[10:16, 10:16, 10:16] = 3
    labels[2:4, 30:32, 50:52] = 3  # small component of label 3
    _write(run, "labels", "s3", labels, is_multilabel=True)
    _invoke(
        filter_components,
        [
            "-c",
            config_path,
            "-i",
            "labels:test/s3@10.0?multilabel=true",
            "-o",
            "labels:filtered/s3@10.0",
            "--min-size",
            "20",
            "--size-unit",
            "voxel",
        ],
    )
    out = _get(run, "labels", "filtered", "s3", is_multilabel=True)
    assert out.is_multilabel
    values = out.numpy()
    assert sorted(np.unique(values).tolist()) == [0, 1, 3]
    assert values[3, 31, 51] == 0


def test_seg_stats_instance_rows(run, config_path, tmp_path):
    instances = np.zeros(SHAPE, dtype=np.uint16)
    instances[10:13, 10:13, 5:55] = 7
    instances[2:6, 2:6, 2:6] = 4
    _write(run, "microtubule", "s4", instances, is_instance=True)
    out = tmp_path / "stats.csv"
    _invoke(
        seg_stats,
        [
            "-c",
            config_path,
            "-i",
            "microtubule:test/s4@10.0?instance=true",
            "--skeleton",
            "--no-include-background",
            "-f",
            "csv",
            "-op",
            str(out),
        ],
    )
    rows = {int(r["instance_id"]): r for r in csv.DictReader(out.open())}
    assert set(rows) == {4, 7}
    assert rows[7]["segmentation_type"] == "instance" and int(rows[7]["label"]) == 1
    assert int(rows[7]["volume_voxels"]) == 450
    assert 400 < float(rows[7]["skeleton_length_angstroms"]) <= 500


def test_untyped_selection_ignores_a_same_key_instance_store(run):
    binary = np.zeros(SHAPE, dtype=np.uint8)
    binary[1:3, 1:3, 1:3] = 1
    instances = np.zeros(SHAPE, dtype=np.uint16)
    instances[5:8, 5:8, 5:8] = 3
    _write(run, "microtubule", "same", binary)
    _write(run, "microtubule", "same", instances, is_instance=True)

    selector = SelectorConfig.from_uris(
        "microtubule:test/same@10.0",
        "segmentation",
        "microtubule:out/same@10.0",
        "segmentation",
    )
    tasks = discover_tasks_for_run(run, selector)
    assert len(tasks) == 1 and not tasks[0]["segmentation"].is_instance

    # An output without a voxel spacing inherits the input's spacing and type
    selector = SelectorConfig.from_uris(
        "microtubule:test/same@10.0?instance=true",
        "segmentation",
        "microtubule:out/same",
        "segmentation",
    )
    tasks = discover_tasks_for_run(run, selector)
    assert len(tasks) == 1 and tasks[0]["segmentation"].is_instance
    assert tasks[0]["output_segmentation_type"] == "instance"

    np.testing.assert_array_equal(readers.segmentation(run, VS, "microtubule", "test", "same"), binary)
    np.testing.assert_array_equal(
        readers.segmentation(run, VS, "microtubule", "test", "same", is_instance=True),
        instances,
    )


def test_panoptic_combine_and_split_round_trip(run, config_path):
    membrane = np.zeros(SHAPE, dtype=np.uint8)
    membrane[0:2] = 1
    instances = np.zeros(SHAPE, dtype=np.uint16)
    instances[5:8, 5:8, 5:50] = 2
    instances[12:15, 20:23, 5:50] = 5
    _write(run, "membrane", "p", membrane)
    _write(run, "microtubule", "p", instances, is_instance=True)

    _invoke(
        combine,
        [
            "-c",
            config_path,
            "-i",
            "membrane:test/p@10.0",
            "--instances",
            "microtubule:test/p@10.0",
            "-o",
            "cell:combine/p@10.0?panoptic=true",
        ],
    )
    panoptic = _get(run, "cell", "combine", "p", is_panoptic=True)
    assert sorted(panoptic.segments()) == [("membrane", 0), ("microtubule", 2), ("microtubule", 5)]

    _invoke(split, ["-c", config_path, "-i", "cell:combine/p@10.0?panoptic=true", "--output-user-id", "back"])
    np.testing.assert_array_equal(_get(run, "microtubule", "back", "p", is_instance=True).numpy(), instances)
    np.testing.assert_array_equal(_get(run, "membrane", "back", "p", is_instance=False).numpy(), membrane)
