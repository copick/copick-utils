"""Expand labels in segmentations to fill holes and gaps."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from copick.util.log import get_logger
from skimage.segmentation import expand_labels as skimage_expand_labels

if TYPE_CHECKING:
    from copick.models import CopickRoot, CopickRun, CopickSegmentation

logger = get_logger(__name__)


def _expand_labels_single(
    seg: np.ndarray,
    distance_voxels: float,
) -> np.ndarray:
    """
    Expand labels in a segmentation by a given distance in voxels.

    Uses skimage.segmentation.expand_labels to grow label regions outward
    without overlapping into neighboring regions.

    Args:
        seg: Label image (integer numpy array).
        distance_voxels: Euclidean distance in voxels by which to expand labels.

    Returns:
        Expanded label array with the same dtype as input.
    """
    return skimage_expand_labels(seg, distance=distance_voxels)


def expand_labels_segmentation(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    voxel_spacing: float,
    is_multilabel: bool = False,
    distance: float = 10.0,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Expand labels in a segmentation to fill holes and gaps.

    Args:
        segmentation: Input CopickSegmentation object.
        run: CopickRun object.
        object_name: Name for the output segmentation.
        session_id: Session ID for the output segmentation.
        user_id: User ID for the output segmentation.
        voxel_spacing: Voxel spacing in angstroms.
        is_multilabel: Whether the segmentation is multilabel.
        distance: Distance in angstroms by which to expand labels.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple of (CopickSegmentation object, stats dict) or None if operation failed.
        Stats dict contains 'voxels_before', 'voxels_after', 'voxels_added'.
    """
    try:
        seg_array = segmentation.numpy()

        if seg_array is None:
            logger.error("Could not load segmentation data")
            return None

        if seg_array.size == 0:
            logger.error("Empty segmentation data")
            return None

        # Convert distance from angstroms to voxels
        distance_voxels = distance / voxel_spacing

        voxels_before = int(np.count_nonzero(seg_array))

        # Expand labels
        result_array = _expand_labels_single(seg_array, distance_voxels=distance_voxels)

        voxels_after = int(np.count_nonzero(result_array))

        # Create output segmentation
        output_seg = run.new_segmentation(
            name=object_name,
            user_id=user_id,
            session_id=session_id,
            is_multilabel=is_multilabel,
            voxel_size=voxel_spacing,
            exist_ok=True,
        )

        output_seg.from_numpy(result_array)

        stats = {
            "voxels_before": voxels_before,
            "voxels_after": voxels_after,
            "voxels_added": voxels_after - voxels_before,
        }
        logger.info(
            f"Expanded labels: {stats['voxels_before']} -> {stats['voxels_after']} voxels "
            f"(+{stats['voxels_added']} voxels, distance={distance:.1f} Å)",
        )
        return output_seg, stats

    except Exception as e:
        logger.error(f"Error expanding labels: {e}")
        return None


def _expand_labels_worker(
    run: "CopickRun",
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str,
    voxel_spacing: float,
    distance: float,
    output_user_id: str,
    output_session_id: str,
    is_multilabel: bool,
) -> Dict[str, Any]:
    """Worker function for batch label expansion."""
    try:
        segmentations = run.get_segmentations(
            name=segmentation_name,
            user_id=segmentation_user_id,
            session_id=segmentation_session_id,
            voxel_size=voxel_spacing,
        )

        if not segmentations:
            return {"processed": 0, "errors": [f"No segmentation found for {run.name}"]}

        segmentation = segmentations[0]

        result = expand_labels_segmentation(
            segmentation=segmentation,
            run=run,
            object_name=segmentation_name,
            session_id=output_session_id,
            user_id=output_user_id,
            voxel_spacing=voxel_spacing,
            is_multilabel=is_multilabel,
            distance=distance,
        )

        if result is None:
            return {"processed": 0, "errors": [f"Failed to expand labels for {run.name}"]}

        output_seg, stats = result

        return {
            "processed": 1,
            "errors": [],
            "voxels_before": stats["voxels_before"],
            "voxels_after": stats["voxels_after"],
            "voxels_added": stats["voxels_added"],
        }

    except Exception as e:
        return {"processed": 0, "errors": [f"Error processing {run.name}: {e}"]}


def expand_labels_batch(
    root: "CopickRoot",
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str,
    voxel_spacing: float,
    distance: float = 10.0,
    output_user_id: str = "expand-labels",
    output_session_id: str = "0",
    is_multilabel: bool = False,
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Batch expand labels across multiple runs.

    Args:
        root: The copick root containing runs to process.
        segmentation_name: Name of the segmentation to process.
        segmentation_user_id: User ID of the segmentation to process.
        segmentation_session_id: Session ID of the segmentation to process.
        voxel_spacing: Voxel spacing in angstroms.
        distance: Distance in angstroms by which to expand labels.
        output_user_id: User ID for output segmentations.
        output_session_id: Session ID for output segmentations.
        is_multilabel: Whether the segmentation is multilabel.
        run_names: List of run names to process. If None, processes all runs.
        workers: Number of worker processes.

    Returns:
        Dictionary with processing results and statistics.
    """
    from copick.ops.run import map_runs

    runs_to_process = [run.name for run in root.runs] if run_names is None else run_names

    results = map_runs(
        callback=_expand_labels_worker,
        root=root,
        runs=runs_to_process,
        workers=workers,
        task_desc="Expanding labels",
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        segmentation_session_id=segmentation_session_id,
        voxel_spacing=voxel_spacing,
        distance=distance,
        output_user_id=output_user_id,
        output_session_id=output_session_id,
        is_multilabel=is_multilabel,
    )

    return results
