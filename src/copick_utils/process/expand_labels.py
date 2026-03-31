"""Expand labels in segmentations to fill holes and gaps."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
from copick.util.log import get_logger
from skimage.segmentation import expand_labels as skimage_expand_labels

from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickSegmentation

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
        distance: Distance in angstroms by which to expand labels.
        **kwargs: Additional keyword arguments (voxel_spacing, etc. from lazy converter).

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

        # Use the actual voxel_size from the segmentation (authoritative source)
        actual_voxel_spacing = segmentation.voxel_size

        # Convert distance from angstroms to voxels
        distance_voxels = distance / actual_voxel_spacing

        voxels_before = int(np.count_nonzero(seg_array))

        # Expand labels
        result_array = _expand_labels_single(seg_array, distance_voxels=distance_voxels)

        voxels_after = int(np.count_nonzero(result_array))

        # Create output segmentation
        output_seg = run.new_segmentation(
            name=object_name,
            user_id=user_id,
            session_id=session_id,
            is_multilabel=segmentation.is_multilabel,
            voxel_size=actual_voxel_spacing,
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


# Lazy batch converter for parallel discovery and processing
expand_labels_lazy_batch = create_lazy_batch_converter(
    converter_func=expand_labels_segmentation,
    task_description="Expanding labels",
)
