"""Shared fixtures: a small filesystem copick project in a temporary directory."""

import json

import copick
import pytest

OBJECTS = [
    {
        "name": "microtubule",
        "is_particle": True,
        "label": 1,
        "radius": 120.0,
        "metadata": {"copick": {"filament": {"polar": True}}},
    },
    {"name": "ribosome", "is_particle": True, "label": 2, "radius": 150.0},
    {"name": "membrane", "is_particle": False, "label": 3},
]


@pytest.fixture
def copick_root(tmp_path):
    """A copick root with one run (``run1``), a filament object, a particle object and a segmentation object."""
    config = {
        "config_type": "filesystem",
        "name": "test",
        "description": "copick-utils test project",
        "version": "1.0.0",
        "pickable_objects": OBJECTS,
        "overlay_root": f"local://{tmp_path / 'overlay'}",
        "overlay_fs_args": {"auto_mkdir": True},
    }
    path = tmp_path / "copick_config.json"
    path.write_text(json.dumps(config))
    root = copick.from_file(str(path))
    root.new_run("run1")
    return root


@pytest.fixture
def run(copick_root):
    """The fixture project's single run."""
    return copick_root.get_run("run1")
