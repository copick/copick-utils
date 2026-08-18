"""Executable baseline for the copick 2 / Zarr 3 migration.

This module records the observable storage behavior present before the
migration so the migrated implementation can be checked against it.
"""

import hashlib
import inspect
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import zarr

from copick_utils.features.skimage import compute_skimage_features
from copick_utils.io.zarr import get_level_array


def _memory_store():
    return zarr.storage.MemoryStore()


def _tomogram_store(path="0", zarr_format=3, shape=(5, 6, 7), chunks=(3, 4, 5)):
    store = _memory_store()
    group = zarr.open_group(store=store, mode="w", zarr_format=zarr_format)
    data = ((np.indices(shape) * np.array([11, 5, 2])[:, None, None, None]).sum(0) % 17).astype(np.float32)
    group.create_array(path, data=data, chunks=chunks)
    group.attrs["multiscales"] = [{"datasets": [{"path": path}]}]
    return store, data


def _empty_tomogram_store(shape, chunks):
    store = _memory_store()
    group = zarr.open_group(store=store, mode="w", zarr_format=3)
    group.create_array("0", shape=shape, chunks=chunks, dtype=np.float32, fill_value=0)
    group.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
    return store


class _Features:
    def __init__(self):
        self.data = None
        self.input_is_memmap = False
        self.staging_path = None
        self.staging_existed_during_write = False
        self.write_calls = []

    def from_numpy(self, data, **kwargs):
        self.input_is_memmap = isinstance(data, np.memmap)
        self.staging_path = Path(data.filename)
        self.staging_existed_during_write = self.staging_path.exists()
        self.data = np.array(data, copy=True)
        self.write_calls.append(kwargs)

    def numpy(self):
        return np.array(self.data, copy=True)


class _NoCopyFeatures(_Features):
    def from_numpy(self, data, **kwargs):
        self.input_is_memmap = isinstance(data, np.memmap)
        self.staging_path = Path(data.filename)
        self.staging_existed_during_write = self.staging_path.exists()
        self.data = (data.shape, data.dtype)
        self.write_calls.append(kwargs)


class _FailingFeatures(_NoCopyFeatures):
    def from_numpy(self, data, **kwargs):
        super().from_numpy(data, **kwargs)
        raise RuntimeError("feature write failed")


class _Tomogram:
    def __init__(self, store, features_factory=_Features):
        self.store = store
        self.features = None
        self.features_factory = features_factory

    def zarr(self):
        return self.store

    def new_features(self, feature_type):
        assert feature_type == "golden"
        self.features = self.features_factory()
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
    result = features.numpy()

    assert result.shape == (5, 5, 6, 7)
    assert result.dtype == np.float32
    rounded_digest = hashlib.sha256(np.round(result, 5).tobytes()).hexdigest()
    assert rounded_digest == "8364181d58811d79fe86847872316a97370ed2737aeee6411a38753124305312"


def test_feature_writer_delegates_one_final_float32_write():
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

    assert features.data.shape == (1, 5, 6, 7)
    assert features.data.dtype == np.float32
    assert features.input_is_memmap
    assert features.staging_existed_during_write
    assert not features.staging_path.exists()
    assert features.write_calls == [
        {
            "chunks": (3, 4, 5),
            "shards": None,
            "dtype": np.float32,
            "overwrite": True,
        },
    ]


@pytest.mark.parametrize("shape", [(13, 8, 8), (8, 13, 8), (8, 8, 13)])
def test_feature_overlap_trimming_handles_image_end_inside_overlap(monkeypatch, shape):
    def constant_features(chunk, **kwargs):
        return np.full((*chunk.shape, 1), 7, dtype=np.float32)

    monkeypatch.setattr("copick_utils.features.skimage.multiscale_basic_features", constant_features)
    store, _ = _tomogram_store(shape=shape, chunks=(4, 4, 4))

    features = compute_skimage_features(
        _Tomogram(store),
        "golden",
        None,
        intensity=True,
        edges=False,
        texture=False,
        feature_chunk_size=(4, 4, 4),
    )

    assert features.data.shape == (1, *shape)
    np.testing.assert_array_equal(features.data, np.full((1, *shape), 7, dtype=np.float32))


def test_feature_staging_is_removed_when_final_write_fails():
    store, _ = _tomogram_store()
    tomogram = _Tomogram(store, _FailingFeatures)

    with pytest.raises(RuntimeError, match="feature write failed"):
        compute_skimage_features(
            tomogram,
            "golden",
            None,
            intensity=True,
            edges=False,
            texture=False,
            sigma_min=0.5,
            sigma_max=0.5,
            feature_chunk_size=(3, 4, 5),
        )

    assert tomogram.features.input_is_memmap
    assert tomogram.features.staging_existed_during_write
    assert not tomogram.features.staging_path.exists()


def test_feature_staging_has_bounded_python_memory(monkeypatch):
    shape = (96, 96, 96)
    feature_count = 30

    def constant_features(chunk, **kwargs):
        return np.full((*chunk.shape, feature_count), 7, dtype=np.float32)

    monkeypatch.setattr("copick_utils.features.skimage.multiscale_basic_features", constant_features)
    tomogram = _Tomogram(_empty_tomogram_store(shape, (16, 16, 16)), _NoCopyFeatures)

    tracemalloc.start()
    features = compute_skimage_features(
        tomogram,
        "golden",
        None,
        intensity=True,
        edges=False,
        texture=False,
        feature_chunk_size=(16, 16, 16),
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logical_size = feature_count * np.prod(shape) * np.dtype(np.float32).itemsize
    assert peak < 64 * 1024 * 1024
    assert peak < logical_size * 0.75
    assert features.data == ((feature_count, *shape), np.dtype(np.float32))
    assert features.input_is_memmap
    assert not features.staging_path.exists()


def test_feature_layout_controls_are_optional_keyword_only():
    signature = inspect.signature(compute_skimage_features)

    # All pre-migration parameters still bind positionally in their original order.
    signature.bind(object(), "features", object(), True, True, True, 0.5, 16.0, (32, 32, 32))

    assert signature.parameters["chunks"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["chunks"].default is None
    assert signature.parameters["shards"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["shards"].default is None
