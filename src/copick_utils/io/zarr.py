"""Metadata-aware access to arrays stored by copick entities."""

from typing import Any

import zarr
from copick.util.ome import get_level_path


def get_level_array(entity: Any, level: int = 0) -> zarr.Array:
    """Open a copick entity's metadata-declared pyramid level read-only.

    The integer level is an index into OME ``multiscales.datasets``; it is not
    assumed to be the array's literal path.
    """
    group = zarr.open_group(store=entity.zarr(), mode="r")
    return group[get_level_path(group, level)]
