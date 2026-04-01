"""Split multilabel segmentations into individual single-class segmentations."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from copick.util.log import get_logger

from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickSegmentation

logger = get_logger(__name__)


def split_multilabel_segmentation(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    output_user_id: str = "split",
) -> List["CopickSegmentation"]:
    """
    Split a multilabel segmentation into individual single-class binary segmentations.

    For each label value in the multilabel segmentation, this function looks up the
    corresponding PickableObject and creates a binary segmentation named after that object.

    Args:
        segmentation: Input multilabel segmentation to split.
        run: CopickRun object containing the segmentation.
        output_user_id: User ID for output segmentations (default: "split").

    Returns:
        List of created CopickSegmentation objects, one per label found in the input.
    """
    volume = segmentation.numpy()
    if volume is None:
        raise ValueError("Could not load segmentation data")

    if volume.size == 0:
        raise ValueError("Empty segmentation data")

    root = run.root
    voxel_size = segmentation.voxel_size
    input_session_id = segmentation.session_id

    unique_labels = np.unique(volume)
    unique_labels = unique_labels[unique_labels > 0]

    logger.debug(f"Found {len(unique_labels)} unique labels: {unique_labels.tolist()}")

    output_segmentations = []

    for label_value in unique_labels:
        pickable_obj = next((obj for obj in root.config.pickable_objects if obj.label == label_value), None)

        if pickable_obj is None:
            logger.warning(f"No pickable object found for label {label_value}, using label value as name")
            object_name = str(label_value)
        else:
            object_name = pickable_obj.name
            logger.debug(f"Label {label_value} → object '{object_name}'")

        binary_mask = (volume == label_value).astype(np.uint8)
        voxel_count = int(np.sum(binary_mask))

        if voxel_count == 0:
            logger.warning(f"Label {label_value} has no voxels, skipping")
            continue

        try:
            output_seg = run.new_segmentation(
                name=object_name,
                user_id=output_user_id,
                session_id=input_session_id,
                is_multilabel=False,
                voxel_size=voxel_size,
                exist_ok=True,
            )

            output_seg.from_numpy(binary_mask)
            output_segmentations.append(output_seg)

        except Exception as e:
            logger.exception(f"Failed to create segmentation for label {label_value} ('{object_name}'): {e}")
            continue

    if output_segmentations:
        object_names = [seg.name for seg in output_segmentations]
        logger.info(f"Run '{run.name}': Split {len(output_segmentations)} labels → {', '.join(object_names)}")

    return output_segmentations


def split_labels_converter(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    **kwargs,
) -> Optional[Tuple[None, Dict[str, int]]]:
    """
    Lazy converter wrapper for split_multilabel_segmentation.

    The output_user_id is taken from the `user_id` parameter (which comes from
    the output URI via the lazy converter).

    Args:
        segmentation: Input multilabel CopickSegmentation object.
        run: CopickRun object.
        object_name: Unused (split determines names from copick config).
        session_id: Unused (split inherits session_id from input).
        user_id: User ID for output segmentations.
        **kwargs: Additional keyword arguments from lazy converter.

    Returns:
        Tuple of (None, stats dict) or None if operation failed.
    """
    try:
        if not segmentation.is_multilabel:
            logger.error(f"Segmentation in run {run.name} is not multilabel (is_multilabel=False)")
            return None

        output_segmentations = split_multilabel_segmentation(
            segmentation=segmentation,
            run=run,
            output_user_id=user_id,
        )

        [seg.name for seg in output_segmentations]

        stats = {
            "labels_split": len(output_segmentations),
        }

        return None, stats

    except Exception as e:
        logger.error(f"Error splitting labels in {run.name}: {e}")
        return None


# Lazy batch converter for parallel discovery and processing
split_labels_lazy_batch = create_lazy_batch_converter(
    converter_func=split_labels_converter,
    task_description="Splitting multilabel segmentations",
)
