"""Basic installed-package metadata checks."""

import importlib.metadata


def test_installed_distribution_declares_supported_python():
    metadata = importlib.metadata.metadata("copick-utils")

    assert metadata["Requires-Python"] == ">=3.11"
