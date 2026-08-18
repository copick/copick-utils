"""Executable baseline for the copick 2 / Zarr 3 migration.

This module intentionally records the storage assumptions present before the
migration.  Later stack layers tighten the inventories as each assumption is
removed, while the feature checksum remains a behavioral regression oracle.
"""

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest
import zarr

from copick_utils.features.skimage import compute_skimage_features


SOURCE_ROOT = Path(__file__).parents[1] / "src" / "copick_utils"
DIRECT_LEVEL_READERS = {
    "converters/picks_from_mesh.py",
    "converters/segmentation_from_mesh.py",
    "converters/segmentation_from_picks.py",
    "features/skimage.py",
    "logical/distance_operations.py",
    "pickers/grid_picker.py",
    "process/rescale.py",
    "process/validbox.py",
}


def _memory_store():
    return zarr.storage.MemoryStore()


def _tomogram_store(path="0"):
    store = _memory_store()
    group = zarr.group(store=store)
    data = ((np.indices((5, 6, 7)) * np.array([11, 5, 2])[:, None, None, None]).sum(0) % 17).astype(np.float32)
    group.create_dataset(path, data=data, chunks=(3, 4, 5))
    group.attrs["multiscales"] = [{"datasets": [{"path": path}]}]
    return store, data


class _Features:
    def __init__(self):
        self.store = _memory_store()

    def zarr(self):
        return self.store


class _Tomogram:
    def __init__(self, store):
        self.store = store
        self.features = None

    def zarr(self):
        return self.store

    def new_features(self, feature_type):
        assert feature_type == "golden"
        self.features = _Features()
        return self.features


@pytest.mark.parametrize("path", ["0", "s0"])
def test_local_ome_zarr_fixture_declares_its_level_path(path):
    store, expected = _tomogram_store(path)
    group = zarr.open_group(store=store, mode="r")

    declared_path = group.attrs["multiscales"][0]["datasets"][0]["path"]
    np.testing.assert_array_equal(group[declared_path][:], expected)


def test_pre_migration_feature_result_is_frozen():
    """Protect the existing chunk subdivision and boundary behavior."""
    store, _ = _tomogram_store()
    features = compute_skimage_features(
        _Tomogram(store),
        "golden",
        None,
        sigma_min=0.5,
        sigma_max=0.5,
        feature_chunk_size=(3, 4, 5),
    )
    result = zarr.open(features.zarr(), mode="r")[:]

    assert result.shape == (5, 5, 6, 7)
    assert result.dtype == np.float32
    rounded_digest = hashlib.sha256(np.round(result, 5).tobytes()).hexdigest()
    assert rounded_digest == "8364181d58811d79fe86847872316a97370ed2737aeee6411a38753124305312"


def test_pre_migration_feature_store_documents_reader_incompatibility():
    store, _ = _tomogram_store()
    features = compute_skimage_features(
        _Tomogram(store),
        "golden",
        None,
        intensity=True,
        edges=False,
        texture=False,
        sigma_min=0.5,
        sigma_max=0.5,
        feature_chunk_size=(3, 4, 5),
    )

    # The old implementation writes an array at the store root.  CopickFeatures
    # expects an OME group and therefore cannot resolve a metadata-defined level.
    root = zarr.open(features.zarr(), mode="r")
    assert isinstance(root, zarr.Array)
    with pytest.raises((AttributeError, TypeError, zarr.errors.ContainsArrayError)):
        zarr.open_group(store=features.zarr(), mode="r")


def test_direct_zarr_boundary_inventory_is_complete():
    direct_level_readers = set()
    feature_array_creators = set()

    for source in SOURCE_ROOT.rglob("*.py"):
        relative = source.relative_to(SOURCE_ROOT).as_posix()
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "zarr":
                continue
            if node.func.attr == "open":
                direct_level_readers.add(relative)
            elif node.func.attr == "create":
                feature_array_creators.add(relative)

    assert direct_level_readers == DIRECT_LEVEL_READERS
    assert feature_array_creators == {"features/skimage.py"}
