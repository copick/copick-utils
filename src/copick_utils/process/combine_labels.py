"""Combine single-label segmentations into a multilabel segmentation."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from copick.util.log import get_logger

from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickSegmentation

logger = get_logger(__name__)


def _combine_panoptic(
    segmentations: List["CopickSegmentation"],
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    instances_uri: Optional[str] = None,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """Combine binary or multilabel segmentations (regions without instances) and per-object instance
    segmentations into one panoptic segmentation. Overlaps go to the lowest label; an instance ID survives only
    where its object's label won."""
    from copick_utils.util.segmentations import new_segmentation, resolve_segmentations, segmentation_type

    root = run.root
    layers = []  # (label value, label mask, instance IDs or None)
    for seg in segmentations:
        volume = seg.numpy()
        if volume is None:
            logger.warning(f"Could not load segmentation '{seg.name}', skipping")
            continue
        if segmentation_type(seg) == "multilabel":
            for value in np.unique(volume[volume > 0]):
                layers.append((int(value), volume == value, None))
        else:
            obj = root.get_object(seg.name)
            if obj is None:
                logger.warning(f"No pickable object found for '{seg.name}', skipping")
                continue
            layers.append((int(obj.label), volume > 0, None))
    voxel_size = segmentations[0].voxel_size if segmentations else None
    if instances_uri:
        uri = instances_uri if "instance=true" in instances_uri else instances_uri + "?instance=true"
        for seg in resolve_segmentations(uri, root, run_name=run.name):
            voxel_size = voxel_size or seg.voxel_size
            obj = root.get_object(seg.name)
            if obj is None:
                logger.warning(f"No pickable object found for instance segmentation '{seg.name}', skipping")
                continue
            ids = seg.numpy()
            layers.append((int(obj.label), ids > 0, ids))
    if not layers:
        logger.error("No segmentations to combine")
        return None

    shape = layers[0][1].shape
    label_channel = np.zeros(shape, dtype=np.uint32)
    instance_channel = np.zeros(shape, dtype=np.uint32)
    overlap_count = np.zeros(shape, dtype=np.uint8)
    # Paint the highest label first, so the lowest label wins where inputs overlap
    for label_value, mask, ids in sorted(layers, key=lambda layer: layer[0], reverse=True):
        overlap_count += mask.astype(np.uint8)
        label_channel[mask] = label_value
        instance_channel[mask] = ids[mask] if ids is not None else 0
    overlapping_voxels = int(np.sum(overlap_count > 1))
    if overlapping_voxels:
        logger.warning(f"Detected {overlapping_voxels} overlapping voxels across inputs. Resolved by lowest label.")

    output_seg = new_segmentation(run, voxel_size, object_name, session_id, user_id, "panoptic")
    output_seg.from_numpy(np.stack([label_channel, instance_channel]))
    stats = {
        "labels_combined": len({layer[0] for layer in layers}),
        "instances_combined": int(len(np.unique(instance_channel[instance_channel > 0]))),
        "overlapping_voxels": overlapping_voxels,
    }
    return output_seg, stats


def combine_labels(
    segmentations: List["CopickSegmentation"],
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    output_segmentation_type: Optional[str] = None,
    instances_uri: Optional[str] = None,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Combine multiple single-label segmentations into one multilabel segmentation.

    Each input segmentation's name is looked up in the copick config to determine
    its integer label value. Overlapping regions are resolved by lowest label priority
    (lowest label value wins).

    When the output URI names a panoptic segmentation (``?panoptic=true``), the inputs become its label channel and
    the instance segmentations selected by ``instances_uri`` add their objects' labels and instance IDs.

    Args:
        segmentations: List of input CopickSegmentation objects (binary/single-label).
        run: CopickRun object.
        object_name: Name for the output multilabel segmentation.
        session_id: Session ID for the output segmentation.
        user_id: User ID for the output segmentation.
        output_segmentation_type: The segmentation type the output URI names, if any.
        instances_uri: Instance segmentations to add to a panoptic output (URI, patterns allowed).
        **kwargs: Additional keyword arguments from lazy converter.

    Returns:
        Tuple of (CopickSegmentation, stats dict) or None if operation failed.
    """
    try:
        if output_segmentation_type == "panoptic":
            return _combine_panoptic(segmentations, run, object_name, session_id, user_id, instances_uri)
        if output_segmentation_type == "instance":
            logger.error("combine writes multilabel or panoptic segmentations, not instance ones")
            return None
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
