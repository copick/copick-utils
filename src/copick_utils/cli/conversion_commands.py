"""CLI commands for data conversion between different copick formats.

This module imports all conversion commands from specialized files for better organization.
"""

# Import picks-to-mesh conversion commands
# Import mesh-to-picks conversion commands
from copick_utils.cli.mesh_to_picks_commands import mesh2picks
from copick_utils.cli.picks_to_mesh_commands import (
    picks2ellipsoid,
    picks2mesh,
    picks2plane,
    picks2sphere,
    picks2surface,
)

# Import segmentation conversion commands
from copick_utils.cli.segmentation_conversion_commands import (
    mesh2seg,
    picks2seg,
    seg2mesh,
    seg2picks,
)

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
