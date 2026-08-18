"""Installed metadata must prevent pre-migration runtime resolution."""

import importlib.metadata


def test_installed_distribution_declares_migration_runtime():
    metadata = importlib.metadata.metadata("copick-utils")
    requirements = {value.replace(" ", "") for value in metadata.get_all("Requires-Dist", [])}

    assert metadata["Requires-Python"] == ">=3.11"
    assert "copick>=2.0.0a1" in requirements
    assert "zarr<4,>=3.1.6" in requirements
