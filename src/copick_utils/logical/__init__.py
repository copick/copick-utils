"""Logical operations for copick objects (meshes, segmentations, picks)."""

from copick_utils.logical.distance_operations import (
    limit_mesh_by_distance,
    limit_picks_by_distance,
    limit_segmentation_by_distance,
)
from copick_utils.logical.mesh_operations import (
    mesh_difference,
    mesh_exclusion,
    mesh_intersection,
    mesh_union,
)
from copick_utils.logical.point_operations import (
    picks_exclusion_by_mesh,
    picks_inclusion_by_mesh,
)
from copick_utils.logical.segmentation_operations import (
    segmentation_difference,
    segmentation_exclusion,
    segmentation_intersection,
    segmentation_union,
)

__all__ = [
    # Mesh boolean operations
    "mesh_union",
    "mesh_difference",
    "mesh_exclusion",
    "mesh_intersection",
    # Segmentation boolean operations
    "segmentation_union",
    "segmentation_difference",
    "segmentation_exclusion",
    "segmentation_intersection",
    # Distance-based limiting operations
    "limit_mesh_by_distance",
    "limit_segmentation_by_distance",
    "limit_picks_by_distance",
    # Point inclusion/exclusion operations
    "picks_inclusion_by_mesh",
    "picks_exclusion_by_mesh",
]
