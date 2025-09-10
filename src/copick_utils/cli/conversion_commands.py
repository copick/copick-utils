"""CLI commands for data conversion between different copick formats.

This module imports all conversion commands from specialized files for better organization.
"""

from copick_utils.cli.mesh2picks import mesh2picks
from copick_utils.cli.mesh2seg import mesh2seg
from copick_utils.cli.picks2ellipsoid import picks2ellipsoid
from copick_utils.cli.picks2mesh import picks2mesh
from copick_utils.cli.picks2plane import picks2plane
from copick_utils.cli.picks2seg import picks2seg
from copick_utils.cli.picks2sphere import picks2sphere
from copick_utils.cli.picks2surface import picks2surface
from copick_utils.cli.seg2mesh import seg2mesh
from copick_utils.cli.seg2picks import seg2picks

# All commands are now available for import by the main CLI
__all__ = [
    # Picks to mesh commands
    "picks2mesh",
    "picks2sphere",
    "picks2ellipsoid",
    "picks2plane",
    "picks2surface",
    # Mesh to picks commands
    "mesh2picks",
    # Segmentation conversion commands
    "picks2seg",
    "seg2picks",
    "mesh2seg",
    "seg2mesh",
]
