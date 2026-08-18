"""Executable baseline for the copick 2 / Zarr 3 migration.

This module records the observable storage behavior present before the
migration so the migrated implementation can be checked against it.
"""

import hashlib

import numpy as np
import pytest
import zarr
from copick_utils.features.skimage import compute_skimage_features
from copick_utils.io.zarr import get_level_array


def _memory_store():
    return zarr.storage.MemoryStore()


def _tomogram_store(path="0", zarr_format=3):
    store = _memory_store()
    group = zarr.open_group(store=store, mode="w", zarr_format=zarr_format)
    data = ((np.indices((5, 6, 7)) * np.array([11, 5, 2])[:, None, None, None]).sum(0) % 17).astype(np.float32)
    group.create_array(path, data=data, chunks=(3, 4, 5))
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


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("path", ["0", "s0"])
def test_level_array_follows_ome_metadata(path, zarr_format):
    store, expected = _tomogram_store(path, zarr_format)
    group = zarr.open_group(store=store, mode="r")

    declared_path = group.attrs["multiscales"][0]["datasets"][0]["path"]
    np.testing.assert_array_equal(group[declared_path][:], expected)
    np.testing.assert_array_equal(get_level_array(_Tomogram(store))[:], expected)


@pytest.mark.parametrize("level", [-1, 1])
def test_level_array_rejects_out_of_range_levels(level):
    store, _ = _tomogram_store("s0")

    with pytest.raises(ValueError, match=f"Level {level} not found"):
        get_level_array(_Tomogram(store), level)


@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="The retained feature writer is migrated in U3",
)
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


@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="The retained feature writer is migrated in U3",
)
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
