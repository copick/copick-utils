"""Expand labels in segmentations to fill holes and gaps."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
from copick.util.log import get_logger
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as ndimage_label
from skimage.segmentation import expand_labels as skimage_expand_labels

from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickSegmentation

logger = get_logger(__name__)


def _expand_labels_single(
    seg: np.ndarray,
    distance_voxels: float,
    max_hole_size_voxels: Optional[float] = None,
    connectivity_value: int = 3,
) -> np.ndarray:
    """
    Expand labels in a segmentation by a given distance in voxels.

    Uses skimage.segmentation.expand_labels to grow label regions outward
    without overlapping into neighboring regions. When max_hole_size_voxels is set,
    expansion is restricted to only fill background holes smaller than the threshold,
    preventing labels from expanding into the unbounded exterior.

    Args:
        seg: Label image (integer numpy array).
        distance_voxels: Euclidean distance in voxels by which to expand labels.
        max_hole_size_voxels: Maximum hole size in voxels to fill. Background connected
            components larger than this are protected from expansion. None means no limit.
        connectivity_value: Connectivity for background component analysis (1=6-conn, 2=18-conn, 3=26-conn).

    Returns:
        Expanded label array with the same dtype as input.
    """
    if max_hole_size_voxels is None:
        return skimage_expand_labels(seg, distance=distance_voxels)

    # Identify which background regions are small enough to fill
    original_foreground = seg > 0
    struct = generate_binary_structure(seg.ndim, connectivity_value)
    labeled_bg, num_bg = ndimage_label(~original_foreground, structure=struct)

    # Build fillable mask: background components with voxel count <= threshold
    bg_counts = np.bincount(labeled_bg.ravel())
    is_fillable = np.zeros(len(bg_counts), dtype=bool)
    for i in range(1, len(bg_counts)):
        is_fillable[i] = bg_counts[i] <= max_hole_size_voxels
    fillable_mask = is_fillable[labeled_bg]

    # Expand labels as usual
    result = skimage_expand_labels(seg, distance=distance_voxels)

    # Mask out expansion into non-fillable background regions
    newly_expanded = (result != 0) & ~original_foreground
    result[newly_expanded & ~fillable_mask] = 0

    return result


def expand_labels_segmentation(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    distance: float = 10.0,
    max_hole_size: Optional[float] = None,
    connectivity: str = "all",
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
        max_hole_size: Maximum hole volume in cubic angstroms (Å³) to fill. Background
            connected components larger than this are protected from expansion. None means no limit.
        connectivity: Connectivity for background hole analysis ("face", "face-edge", "all").
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

        # Convert max_hole_size from cubic angstroms to cubic voxels
        voxel_volume = actual_voxel_spacing**3
        max_hole_size_voxels = max_hole_size / voxel_volume if max_hole_size is not None else None

        # Convert connectivity string to integer
        connectivity_map = {"face": 1, "face-edge": 2, "all": 3}
        connectivity_value = connectivity_map.get(connectivity, 3)

        voxels_before = int(np.count_nonzero(seg_array))

        # Expand labels
        result_array = _expand_labels_single(
            seg_array,
            distance_voxels=distance_voxels,
            max_hole_size_voxels=max_hole_size_voxels,
            connectivity_value=connectivity_value,
        )

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
        msg = (
            f"Expanded labels: {stats['voxels_before']} -> {stats['voxels_after']} voxels "
            f"(+{stats['voxels_added']} voxels, distance={distance:.1f} Å"
        )
        if max_hole_size is not None:
            msg += f", max_hole_size={max_hole_size:.1f} Å³"
        msg += ")"
        logger.info(msg)
        return output_seg, stats

    except Exception as e:
        logger.error(f"Error expanding labels: {e}")
        return None


# Lazy batch converter for parallel discovery and processing
expand_labels_lazy_batch = create_lazy_batch_converter(
    converter_func=expand_labels_segmentation,
    task_description="Expanding labels",
)
