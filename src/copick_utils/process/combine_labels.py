"""Combine single-label segmentations into a multilabel segmentation."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from copick.util.log import get_logger

from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickSegmentation

logger = get_logger(__name__)


def combine_labels(
    segmentations: List["CopickSegmentation"],
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Combine multiple single-label segmentations into one multilabel segmentation.

    Each input segmentation's name is looked up in the copick config to determine
    its integer label value. Overlapping regions are resolved by lowest label priority
    (lowest label value wins).

    Args:
        segmentations: List of input CopickSegmentation objects (binary/single-label).
        run: CopickRun object.
        object_name: Name for the output multilabel segmentation.
        session_id: Session ID for the output segmentation.
        user_id: User ID for the output segmentation.
        **kwargs: Additional keyword arguments from lazy converter.

    Returns:
        Tuple of (CopickSegmentation, stats dict) or None if operation failed.
    """
    try:
        if not segmentations:
            logger.error("No segmentations provided")
            return None

        root = run.root

        # Resolve each segmentation's copick label value
        seg_labels = []
        for seg in segmentations:
            obj = root.get_object(seg.name)
            if obj is None:
                logger.warning(f"No pickable object found for '{seg.name}', skipping")
                continue
            seg_labels.append((seg, obj.label))

        if not seg_labels:
            logger.error("No valid segmentations with matching copick objects")
            return None

        # Load first segmentation to get volume shape
        first_array = seg_labels[0][0].numpy()
        if first_array is None or first_array.size == 0:
            logger.error("Could not load first segmentation")
            return None

        # Use the voxel_size from the first input
        voxel_size = seg_labels[0][0].voxel_size

        # Allocate output volume
        output = np.zeros(first_array.shape, dtype=np.uint16)

        # Count overlaps: track how many inputs are nonzero per voxel
        overlap_count = np.zeros(first_array.shape, dtype=np.uint8)

        # Sort by label value descending — paint highest first, so lowest overwrites
        seg_labels.sort(key=lambda x: x[1], reverse=True)

        for seg, label_value in seg_labels:
            mask = seg.numpy()
            if mask is None:
                logger.warning(f"Could not load segmentation '{seg.name}', skipping")
                continue

            nonzero = mask > 0
            overlap_count += nonzero.astype(np.uint8)
            output[nonzero] = label_value

        # Check for overlaps
        overlapping_voxels = int(np.sum(overlap_count > 1))
        if overlapping_voxels > 0:
            logger.warning(
                f"Detected {overlapping_voxels} overlapping voxels across inputs. "
                f"Resolved by lowest label priority.",
            )

        # Create output multilabel segmentation
        output_seg = run.new_segmentation(
            name=object_name,
            user_id=user_id,
            session_id=session_id,
            is_multilabel=True,
            voxel_size=voxel_size,
            exist_ok=True,
        )

        output_seg.from_numpy(output)

        labels_used = sorted({lv for _, lv in seg_labels})
        stats = {
            "labels_combined": len(seg_labels),
            "overlapping_voxels": overlapping_voxels,
        }
        logger.info(
            f"Combined {stats['labels_combined']} segmentations into multilabel "
            f"(labels: {labels_used}, overlaps: {overlapping_voxels})",
        )
        return output_seg, stats

    except Exception as e:
        logger.error(f"Error combining labels: {e}")
        return None


# Lazy batch converter — uses single_selector_multi_union path
# which collects all matching segmentations into a list
combine_labels_lazy_batch = create_lazy_batch_converter(
    converter_func=combine_labels,
    task_description="Combining labels",
)
