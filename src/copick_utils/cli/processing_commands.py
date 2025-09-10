"""CLI commands for segmentation processing operations."""

from copick_utils.cli.fit_spline import fit_spline
from copick_utils.cli.separate_components import separate_components
from copick_utils.cli.skeletonize import skeletonize
from copick_utils.cli.validbox import validbox

# All commands are now available for import by the main CLI
__all__ = [
    "validbox",
    "skeletonize",
    "separate_components",
    "fit_spline",
]
