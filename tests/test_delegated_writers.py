"""Integration coverage for utility writes delegated to copick core."""

from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from copick.impl.filesystem import CopickConfigFSSpec, CopickRootFSSpec
from copick.util.ome import get_level_path
from copick_utils.converters import lazy_converter
from copick_utils.io import writers
from copick_utils.process.rescale import rescale_segmentation


@pytest.fixture
def run(tmp_path):
    config = CopickConfigFSSpec(
        pickable_objects=[],
        overlay_root=f"local://{tmp_path}",
        overlay_fs_args={"auto_mkdir": True},
    )
    return CopickRootFSSpec(config).new_run("run")


def _assert_canonical_volume(entity, expected):
    group = zarr.open_group(store=entity.zarr(), mode="r")
    level = group[get_level_path(group, 0)]

    assert group.metadata.zarr_format == 3
    assert group.attrs["ome"]["version"] == "0.5"
    assert level.metadata.dimension_names == ("z", "y", "x")
    assert level.chunks == (128, 128, 128)
    np.testing.assert_array_equal(level[:], expected)


def test_io_writers_delegate_canonical_tomogram_and_segmentation_output(run):
    volume = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    labels = (volume % 3).astype(np.uint8)

    writers.tomogram(run, volume, voxel_size=10, algorithm="wbp")
    writers.segmentation(run, labels, "writer", name="source", voxel_size=10)

    tomogram = run.get_voxel_spacing(10).get_tomogram("wbp")
    segmentation = run.get_segmentations(name="source")[0]
    _assert_canonical_volume(tomogram, volume)
    _assert_canonical_volume(segmentation, labels)


def test_processor_reads_and_writes_through_copick_entities(run):
    volume = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    labels = (volume % 3).astype(np.uint8)
    writers.tomogram(run, volume, voxel_size=10, algorithm="wbp")
    writers.segmentation(run, labels, "writer", name="source", voxel_size=10)
    source = run.get_segmentations(name="source")[0]

    derived, stats = rescale_segmentation(
        source,
        run,
        object_name="derived",
        session_id="1",
        user_id="processor",
        target_voxel_spacing=10,
        tomo_type="wbp",
    )

    assert stats == {"rescaled": 1, "labels_preserved": 1}
    _assert_canonical_volume(derived, labels)


def test_lazy_worker_preserves_core_writer_failure(monkeypatch):
    task = {
        "segmentation": SimpleNamespace(session_id="input"),
        "object_name": "derived",
        "user_id": "processor",
        "session_id": "1",
        "voxel_spacing": 10,
    }
    monkeypatch.setattr(lazy_converter, "discover_tasks_for_run", lambda run, selector: [task])

    def reject_write(**kwargs):
        raise ValueError("requested shard exceeds the configured size limit")

    result = lazy_converter.lazy_conversion_worker(
        run=SimpleNamespace(name="run"),
        config=SimpleNamespace(type="single_selector", selector=object()),
        converter_func=reject_write,
    )

    assert result["processed"] == 0
    assert result["errors"] == [
        "Error processing task in run: requested shard exceeds the configured size limit",
    ]
