"""CLI commands for copick-utils."""

from .conversion_commands import (
    mesh2picks,
    mesh2seg,
    picks2ellipsoid,
    picks2mesh,
    picks2plane,
    picks2seg,
    picks2sphere,
    picks2surface,
    seg2mesh,
    seg2picks,
)
from .processing_commands import fit_spline, separate_components, skeletonize

__all__ = [
    # Conversion commands
    "picks2seg",
    "seg2picks",
    "mesh2seg",
    "seg2mesh",
    "picks2mesh",
    "mesh2picks",
    "picks2surface",
    "picks2plane",
    "picks2sphere",
    "picks2ellipsoid",
    # Processing commands
    "separate_components",
    "skeletonize",
    "fit_spline",
]
