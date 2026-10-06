"""Point inclusion/exclusion operations for picks relative to meshes and segmentations."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger

from copick_utils.converters.lazy_converter import create_lazy_batch_converter
from copick_utils.util.picks import point_centres, segmentation_label_volume, store_pick_subset, voxel_indices

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
        points: Array of points to check (N, 3) in physical (x, y, z) coordinates
        segmentation_array: Segmentation array (z, y, x); any non-zero voxel is inside (binary, multilabel, or an
            instance segmentation's IDs)
        voxel_spacing: Spacing between voxels

    Returns:
        Boolean array indicating which points are inside the segmentation
    """
    zyx = voxel_indices(points, voxel_spacing)
    valid_bounds = np.all((zyx >= 0) & (zyx < np.array(segmentation_array.shape)), axis=1)

    inside = np.zeros(len(points), dtype=bool)
    valid = zyx[valid_bounds]
    if len(valid) > 0:
        inside[valid_bounds] = segmentation_array[valid[:, 0], valid[:, 1], valid[:, 2]] > 0
    return inside


def _inside_reference(
    centres: np.ndarray,
    reference_mesh: Optional["CopickMesh"],
    reference_segmentation: Optional["CopickSegmentation"],
) -> Optional[np.ndarray]:
    """Which particle centres lie inside the reference mesh or segmentation (None if the reference cannot be read)."""
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

        return _check_points_in_mesh(centres, ref_mesh)

    ref_seg_array = segmentation_label_volume(reference_segmentation)
    if ref_seg_array is None or ref_seg_array.size == 0:
        logger.error("Could not load reference segmentation data")
        return None
    return _check_points_in_segmentation(centres, ref_seg_array, reference_segmentation.voxel_size)


def _filter_picks_by_reference(
    picks: "CopickPicks",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    reference_mesh: Optional["CopickMesh"],
    reference_segmentation: Optional["CopickSegmentation"],
    keep_inside: bool,
) -> Optional[Tuple["CopickPicks", Dict[str, int]]]:
    """Keep the picks whose particle centres are inside (or outside) a reference, unchanged and in order."""
    if reference_mesh is None and reference_segmentation is None:
        raise ValueError("Either reference_mesh or reference_segmentation must be provided")

    points = picks.points or []
    if len(points) == 0:
        logger.error("Could not load pick data")
        return None

    inside_mask = _inside_reference(point_centres(points), reference_mesh, reference_segmentation)
    if inside_mask is None:
        return None

    keep = inside_mask if keep_inside else ~inside_mask
    where = "inside" if keep_inside else "outside"
    if not np.any(keep):
        logger.warning(f"No picks found {where} reference volume; writing an empty pick set")

    output_picks = store_pick_subset(points, keep, run, object_name, session_id, user_id)
    stats = {"points_created": int(np.count_nonzero(keep))}
    logger.info(f"Kept {stats['points_created']} of {len(points)} picks {where} reference volume")
    return output_picks, stats


def picks_inclusion_by_mesh(
    picks: "CopickPicks",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    reference_mesh: Optional["CopickMesh"] = None,
    reference_segmentation: Optional["CopickSegmentation"] = None,
    **kwargs,
) -> Optional[Tuple["CopickPicks", Dict[str, int]]]:
    """
    Filter picks to include only those inside a reference mesh or segmentation.

    A pick is tested at its particle centre (``location`` plus the transform's shift). Kept picks are written
    unchanged and in their original order: transform, ``instance_id`` and ``score`` survive. If no pick is inside,
    an empty pick set is written.

    Args:
        picks: CopickPicks to filter
        reference_mesh: Reference CopickMesh (either this or reference_segmentation must be provided)
        reference_segmentation: Reference CopickSegmentation (binary, multilabel, instance, or the label channel
            of a panoptic segmentation; any non-zero voxel is inside)
        run: CopickRun object
        object_name: Name for the output picks
        session_id: Session ID for the output picks
        user_id: User ID for the output picks
        **kwargs: Additional keyword arguments (e.g. voxel_spacing, reference_tomogram_info; ignored)

    Returns:
        Tuple of (CopickPicks object, stats dict) or None if operation failed.
        Stats dict contains 'points_created'.
    """
    try:
        return _filter_picks_by_reference(
            picks,
            run,
            object_name,
            session_id,
            user_id,
            reference_mesh,
            reference_segmentation,
            keep_inside=True,
        )
    except Exception as e:
        logger.error(f"Error filtering picks by inclusion: {e}")
        return None


def picks_exclusion_by_mesh(
    picks: "CopickPicks",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    reference_mesh: Optional["CopickMesh"] = None,
    reference_segmentation: Optional["CopickSegmentation"] = None,
    **kwargs,
) -> Optional[Tuple["CopickPicks", Dict[str, int]]]:
    """
    Filter picks to exclude those inside a reference mesh or segmentation.

    A pick is tested at its particle centre (``location`` plus the transform's shift). Kept picks are written
    unchanged and in their original order: transform, ``instance_id`` and ``score`` survive. If every pick is
    inside, an empty pick set is written.

    Args:
        picks: CopickPicks to filter
        reference_mesh: Reference CopickMesh (either this or reference_segmentation must be provided)
        reference_segmentation: Reference CopickSegmentation (binary, multilabel, instance, or the label channel
            of a panoptic segmentation; any non-zero voxel is inside)
        run: CopickRun object
        object_name: Name for the output picks
        session_id: Session ID for the output picks
        user_id: User ID for the output picks
        **kwargs: Additional keyword arguments (e.g. voxel_spacing, reference_tomogram_info; ignored)

    Returns:
        Tuple of (CopickPicks object, stats dict) or None if operation failed.
        Stats dict contains 'points_created'.
    """
    try:
        return _filter_picks_by_reference(
            picks,
            run,
            object_name,
            session_id,
            user_id,
            reference_mesh,
            reference_segmentation,
            keep_inside=False,
        )
    except Exception as e:
        logger.error(f"Error filtering picks by exclusion: {e}")
        return None


# Lazy batch converters for new architecture
picks_inclusion_by_mesh_lazy_batch = create_lazy_batch_converter(
    converter_func=picks_inclusion_by_mesh,
    task_description="Filtering picks by inclusion",
)

picks_exclusion_by_mesh_lazy_batch = create_lazy_batch_converter(
    converter_func=picks_exclusion_by_mesh,
    task_description="Filtering picks by exclusion",
)
