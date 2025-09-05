"""CLI commands for logical operations (boolean operations, distance limiting, point filtering).

This module imports all logical operation commands from specialized files for better organization.
"""

# Import mesh boolean operation commands
# Import distance-based limiting commands
from copick_utils.cli.distance_limiting_commands import (
    clipmesh,
    clippicks,
    clipseg,
)
from copick_utils.cli.mesh_logical_commands import meshop

# Import point filtering commands
from copick_utils.cli.point_filtering_commands import (
    picksin,
    picksout,
)

# Import segmentation boolean operation commands
from copick_utils.cli.segmentation_logical_commands import segop

# All commands are now available for import by the main CLI
__all__ = [
    # Boolean operation commands
    "meshop",
    "segop",
    # Distance limiting commands
    "clipmesh",
    "clipseg",
    "clippicks",
    # Point filtering commands
    "picksin",
    "picksout",
]
