"""CLI commands for logical operations (boolean operations, distance limiting, point filtering).

This module imports all logical operation commands from specialized files for better organization.
"""


from copick_utils.cli.clipmesh import clipmesh
from copick_utils.cli.clippicks import clippicks
from copick_utils.cli.clipseg import clipseg
from copick_utils.cli.enclosed import enclosed
from copick_utils.cli.meshop import meshop
from copick_utils.cli.picksin import picksin
from copick_utils.cli.picksout import picksout
from copick_utils.cli.segop import segop

# All commands are now available for import by the main CLI
__all__ = [
    # Boolean operation commands
    "meshop",
    "segop",
    "enclosed",
    # Distance limiting commands
    "clipmesh",
    "clipseg",
    "clippicks",
    # Point filtering commands
    "picksin",
    "picksout",
]
