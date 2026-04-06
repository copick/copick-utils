"""Filter thin regions from segmentations by thickness."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
from copick.util.log import get_logger
from scipy.ndimage import distance_transform_edt
from skimage.morphology import ball, binary_dilation

from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickSegmentation

logger = get_logger(__name__)


def _thickness_filter_binary(
    seg: np.ndarray,
    min_thickness_voxels: float,
) -> np.ndarray:
    """
    Remove thin regions from a binary segmentation mask.

    A region is kept if it contains at least one voxel whose inscribed sphere
    diameter >= min_thickness_voxels. The thick core is dilated back to recover
    the full extent of thick structures.

    Args:
        seg: Binary mask (3-D numpy array).
        min_thickness_voxels: Minimum thickness in voxels (diameter) to survive the filter.

    Returns:
        Filtered binary mask (uint8 array, same shape).
    """
    min_radius = min_thickness_voxels / 2.0

    # EDT gives inscribed-sphere radius at each interior voxel
    dt = distance_transform_edt(seg)

    # Thick cores: voxels where an inscribed sphere of >= min_radius exists
    thick_core = dt >= min_radius

    # Dilate cores back by min_radius to recover full region extent,
    # then intersect with the original mask
    selem = ball(int(np.ceil(min_radius)))
    recovered = binary_dilation(thick_core, footprint=selem)

    return (seg & recovered).astype(np.uint8)


def _thickness_filter_multilabel(
    label_map: np.ndarray,
    min_thickness_voxels: float,
    background: int = 0,
) -> np.ndarray:
    """
    Per-label thickness filter for multi-label segmentations.

    Args:
        label_map: Multi-label segmentation (3-D numpy array).
        min_thickness_voxels: Minimum thickness in voxels (diameter) to survive the filter.
        background: Background label value to skip.

    Returns:
        Filtered multi-label segmentation (same dtype and shape).
    """
    result = np.zeros_like(label_map)
    for label_id in np.unique(label_map):
        if label_id == background:
            continue
        mask = label_map == label_id
        filtered = _thickness_filter_binary(mask, min_thickness_voxels)
        result[filtered > 0] = label_id
    return result


def thickness_filter_segmentation(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    min_thickness: float = 50.0,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Filter thin regions from a segmentation by thickness.

    Args:
        segmentation: Input CopickSegmentation object.
        run: CopickRun object.
        object_name: Name for the output segmentation.
        session_id: Session ID for the output segmentation.
        user_id: User ID for the output segmentation.
        min_thickness: Minimum thickness in angstroms (diameter) to keep.
        **kwargs: Additional keyword arguments from lazy converter.

    Returns:
        Tuple of (CopickSegmentation object, stats dict) or None if operation failed.
    """
    try:
        seg_array = segmentation.numpy()

        if seg_array is None:
            logger.error("Could not load segmentation data")
            return None

        if seg_array.size == 0:
            logger.error("Empty segmentation data")
            return None

        # Use actual voxel_size from the segmentation (authoritative source)
        actual_voxel_spacing = segmentation.voxel_size

        # Convert thickness from angstroms to voxels
        min_thickness_voxels = min_thickness / actual_voxel_spacing

        voxels_before = int(np.count_nonzero(seg_array))

        if segmentation.is_multilabel:
            result_array = _thickness_filter_multilabel(seg_array, min_thickness_voxels)
        else:
            bool_seg = seg_array.astype(bool)
            result_array = _thickness_filter_binary(bool_seg, min_thickness_voxels)

        voxels_after = int(np.count_nonzero(result_array))

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
            "voxels_removed": voxels_before - voxels_after,
        }
        logger.info(
            f"Thickness filter: {stats['voxels_before']} -> {stats['voxels_after']} voxels "
            f"({stats['voxels_removed']} removed, min_thickness={min_thickness:.1f} Å)",
        )
        return output_seg, stats

    except Exception as e:
        logger.error(f"Error filtering segmentation by thickness: {e}")
        return None


# Lazy batch converter for parallel discovery and processing
thickness_filter_lazy_batch = create_lazy_batch_converter(
    converter_func=thickness_filter_segmentation,
    task_description="Filtering by thickness",
)
