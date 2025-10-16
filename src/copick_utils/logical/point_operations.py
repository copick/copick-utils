"""Point inclusion/exclusion operations for picks relative to meshes and segmentations."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger

from copick_utils.converters.converter_common import (
    create_batch_converter,
    create_batch_worker,
)
from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickPicks, CopickRun, CopickSegmentation

logger = get_logger(__name__)


def _check_points_in_mesh(points: np.ndarray, mesh: tm.Trimesh) -> np.ndarray:
    """
    Check which points are inside a watertight mesh.

    Args:
        points: Array of points to check (N, 3)
        mesh: Watertight trimesh object

    Returns:
        Boolean array indicating which points are inside the mesh
    """
    try:
        # Check if mesh is watertight
        if not mesh.is_watertight:
            logger.warning("Mesh is not watertight, using bounding box approximation")
            # Fallback: use bounding box
            bounds = mesh.bounds
            inside = np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)
            return inside

        # Use contains method for watertight meshes
        inside = mesh.contains(points)
        return inside

    except Exception as e:
        logger.warning(f"Error checking point containment: {e}")
        # Fallback: use bounding box
        bounds = mesh.bounds
        inside = np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)
        return inside


def _check_points_in_segmentation(
    points: np.ndarray,
    segmentation_array: np.ndarray,
    voxel_spacing: float,
) -> np.ndarray:
    """
    Check which points are inside a segmentation volume.

    Args:
        points: Array of points to check (N, 3) in physical coordinates
        segmentation_array: Binary segmentation array
        voxel_spacing: Spacing between voxels

    Returns:
        Boolean array indicating which points are inside the segmentation
    """
    # Convert points to voxel coordinates
    voxel_coords = np.round(points / voxel_spacing).astype(int)

    # Check bounds
    valid_bounds = (
        (voxel_coords[:, 0] >= 0)
        & (voxel_coords[:, 0] < segmentation_array.shape[2])
        & (voxel_coords[:, 1] >= 0)
        & (voxel_coords[:, 1] < segmentation_array.shape[1])
        & (voxel_coords[:, 2] >= 0)
        & (voxel_coords[:, 2] < segmentation_array.shape[0])
    )

    inside = np.zeros(len(points), dtype=bool)

    # Check only points within bounds
    valid_coords = voxel_coords[valid_bounds]
    if len(valid_coords) > 0:
        # Check if voxels are non-zero (inside segmentation)
        voxel_values = segmentation_array[valid_coords[:, 2], valid_coords[:, 1], valid_coords[:, 0]]
        inside[valid_bounds] = voxel_values > 0

    return inside


def picks_inclusion_by_mesh(
    picks: "CopickPicks",
    run: "CopickRun",
    pick_object_name: str,
    pick_session_id: str,
    pick_user_id: str,
    reference_mesh: Optional["CopickMesh"] = None,
    reference_segmentation: Optional["CopickSegmentation"] = None,
) -> Optional[Tuple["CopickPicks", Dict[str, int]]]:
    """
    Filter picks to include only those inside a reference mesh or segmentation.

    Args:
        picks: CopickPicks to filter
        reference_mesh: Reference CopickMesh (either this or reference_segmentation must be provided)
        reference_segmentation: Reference CopickSegmentation
        run: CopickRun object
        pick_object_name: Name for the output picks
        pick_session_id: Session ID for the output picks
        pick_user_id: User ID for the output picks
        **kwargs: Additional keyword arguments

    Returns:
        Tuple of (CopickPicks object, stats dict) or None if operation failed.
        Stats dict contains 'points_created'.
    """
    try:
        if reference_mesh is None and reference_segmentation is None:
            raise ValueError("Either reference_mesh or reference_segmentation must be provided")

        # Load pick data
        points, transforms = picks.numpy()
        if points is None or len(points) == 0:
            logger.error("Could not load pick data")
            return None

        pick_positions = points[:, :3]  # Use only x, y, z coordinates

        # Check which points are inside the reference
        if reference_mesh is not None:
            ref_mesh = reference_mesh.mesh
            if ref_mesh is None:
                logger.error("Could not load reference mesh data")
                return None

            if isinstance(ref_mesh, tm.Scene):
                if len(ref_mesh.geometry) == 0:
                    logger.error("Reference mesh is empty")
                    return None
                ref_mesh = tm.util.concatenate(list(ref_mesh.geometry.values()))

            inside_mask = _check_points_in_mesh(pick_positions, ref_mesh)

        else:  # reference_segmentation is not None
            ref_seg_array = reference_segmentation.numpy()
            if ref_seg_array is None or ref_seg_array.size == 0:
                logger.error("Could not load reference segmentation data")
                return None

            inside_mask = _check_points_in_segmentation(
                pick_positions,
                ref_seg_array,
                reference_segmentation.voxel_size,
            )

        if not np.any(inside_mask):
            logger.warning("No picks found inside reference volume")
            return None

        # Filter picks to include only those inside
        included_points = points[inside_mask]
        included_transforms = transforms[inside_mask] if transforms is not None else None

        # Create output picks
        output_picks = run.new_picks(pick_object_name, pick_session_id, pick_user_id, exist_ok=True)
        output_picks.from_numpy(positions=included_points, transforms=included_transforms)
        output_picks.store()

        stats = {"points_created": len(included_points)}
        logger.info(f"Included {stats['points_created']} picks inside reference volume")
        return output_picks, stats

    except Exception as e:
        logger.error(f"Error filtering picks by inclusion: {e}")
        return None


def picks_exclusion_by_mesh(
    picks: "CopickPicks",
    run: "CopickRun",
    pick_object_name: str,
    pick_session_id: str,
    pick_user_id: str,
    reference_mesh: Optional["CopickMesh"] = None,
    reference_segmentation: Optional["CopickSegmentation"] = None,
) -> Optional[Tuple["CopickPicks", Dict[str, int]]]:
    """
    Filter picks to exclude those inside a reference mesh or segmentation.

    Args:
        picks: CopickPicks to filter
        reference_mesh: Reference CopickMesh (either this or reference_segmentation must be provided)
        reference_segmentation: Reference CopickSegmentation
        run: CopickRun object
        pick_object_name: Name for the output picks
        pick_session_id: Session ID for the output picks
        pick_user_id: User ID for the output picks
        **kwargs: Additional keyword arguments

    Returns:
        Tuple of (CopickPicks object, stats dict) or None if operation failed.
        Stats dict contains 'points_created'.
    """
    try:
        if reference_mesh is None and reference_segmentation is None:
            raise ValueError("Either reference_mesh or reference_segmentation must be provided")

        # Load pick data
        points, transforms = picks.numpy()
        if points is None or len(points) == 0:
            logger.error("Could not load pick data")
            return None

        pick_positions = points[:, :3]  # Use only x, y, z coordinates

        # Check which points are inside the reference
        if reference_mesh is not None:
            ref_mesh = reference_mesh.mesh
            if ref_mesh is None:
                logger.error("Could not load reference mesh data")
                return None

            if isinstance(ref_mesh, tm.Scene):
                if len(ref_mesh.geometry) == 0:
                    logger.error("Reference mesh is empty")
                    return None
                ref_mesh = tm.util.concatenate(list(ref_mesh.geometry.values()))

            inside_mask = _check_points_in_mesh(pick_positions, ref_mesh)

        else:  # reference_segmentation is not None
            ref_seg_array = reference_segmentation.numpy()
            if ref_seg_array is None or ref_seg_array.size == 0:
                logger.error("Could not load reference segmentation data")
                return None

            inside_mask = _check_points_in_segmentation(
                pick_positions,
                ref_seg_array,
                reference_segmentation.voxel_size,
            )

        # Invert mask to exclude points inside
        outside_mask = ~inside_mask

        if not np.any(outside_mask):
            logger.warning("No picks found outside reference volume")
            return None

        # Filter picks to exclude those inside
        excluded_points = points[outside_mask]
        excluded_transforms = transforms[outside_mask] if transforms is not None else None

        # Create output picks
        output_picks = run.new_picks(pick_object_name, pick_session_id, pick_user_id, exist_ok=True)
        output_picks.from_numpy(positions=excluded_points, transforms=excluded_transforms)
        output_picks.store()

        stats = {"points_created": len(excluded_points)}
        logger.info(
            f"Excluded {len(points) - stats['points_created']} picks inside reference volume, kept {stats['points_created']} picks",
        )
        return output_picks, stats

    except Exception as e:
        logger.error(f"Error filtering picks by exclusion: {e}")
        return None


# Create batch workers
_picks_inclusion_by_mesh_worker = create_batch_worker(picks_inclusion_by_mesh, "picks", "picks", min_points=1)
_picks_exclusion_by_mesh_worker = create_batch_worker(picks_exclusion_by_mesh, "picks", "picks", min_points=1)

# Create batch converters
picks_inclusion_by_mesh_batch = create_batch_converter(
    picks_inclusion_by_mesh,
    "Filtering picks by inclusion",
    "picks",
    "picks",
    min_points=1,
)

picks_exclusion_by_mesh_batch = create_batch_converter(
    picks_exclusion_by_mesh,
    "Filtering picks by exclusion",
    "picks",
    "picks",
    min_points=1,
)

# Lazy batch converters for new architecture
picks_inclusion_by_mesh_lazy_batch = create_lazy_batch_converter(
    converter_func=picks_inclusion_by_mesh,
    task_description="Filtering picks by inclusion",
)

picks_exclusion_by_mesh_lazy_batch = create_lazy_batch_converter(
    converter_func=picks_exclusion_by_mesh,
    task_description="Filtering picks by exclusion",
)
