"""Filter connected components in segmentations by size."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
from copick.util.log import get_logger
from scipy.ndimage import generate_binary_structure, label

from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickSegmentation

logger = get_logger(__name__)


def _keep_mask(
    counts: np.ndarray,
    voxel_volume: float,
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    keep_largest: Optional[int] = None,
) -> np.ndarray:
    """Which ids (indices of ``counts``) pass the size filters. Index 0 (background) never passes; ids with no
    voxels never pass."""
    volumes = counts * voxel_volume
    keep = counts > 0
    keep[0] = False
    if min_size is not None:
        keep &= volumes >= min_size
    if max_size is not None:
        keep &= volumes <= max_size
    if keep_largest is not None:
        present = np.flatnonzero(counts[1:] > 0) + 1
        order = present[np.argsort(counts[present], kind="stable")]
        largest = np.zeros(len(counts), dtype=bool)
        largest[order[-keep_largest:] if keep_largest > 0 else []] = True
        keep &= largest
    return keep


def _apply_skeleton_length(
    labeled: np.ndarray,
    keep: np.ndarray,
    voxel_spacing: float,
    min_skeleton_length: Optional[float],
) -> np.ndarray:
    """Also drop the kept ids whose skeleton is shorter than ``min_skeleton_length`` (angstroms)."""
    if not min_skeleton_length:
        return keep
    from scipy.ndimage import find_objects

    from copick_utils.process.skeleton_graph import summarize

    keep = keep.copy()
    for value, box in enumerate(find_objects(labeled), start=1):
        if box is None or value >= len(keep) or not keep[value]:
            continue
        if summarize(labeled[box] == value).length * voxel_spacing < min_skeleton_length:
            keep[value] = False
    return keep


def _filter_components_by_size(
    seg: np.ndarray,
    voxel_spacing: float,
    connectivity: str = "all",
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    keep_largest: Optional[int] = None,
    min_skeleton_length: Optional[float] = None,
) -> Tuple[np.ndarray, int, int, list]:
    """
    Filter connected components in a segmentation by size.

    Args:
        seg: Binary mask segmentation (numpy array)
        voxel_spacing: Voxel spacing in angstroms
        connectivity: Connectivity for connected components (default: "all")
                     "face" = face connectivity (6-connected in 3D)
                     "face-edge" = face+edge connectivity (18-connected in 3D)
                     "all" = face+edge+corner connectivity (26-connected in 3D)
        min_size: Minimum component volume in cubic angstroms (Å³) to keep (None = no minimum)
        max_size: Maximum component volume in cubic angstroms (Å³) to keep (None = no maximum)
        keep_largest: If set, keep only the N largest components by voxel count (None = no limit).
            Applied in addition to (intersected with) any min/max size filter, so a component is
            kept only if it both passes the size range and ranks within the N largest.
        min_skeleton_length: Minimum skeleton length in angstroms (None = no minimum); drops specks and short
            blobs of filamentous structures.

    Returns:
        Tuple of (seg_filtered, num_kept, num_removed, component_info)
        - seg_filtered: Filtered segmentation with only components passing size criteria
        - num_kept: Number of components kept
        - num_removed: Number of components removed
        - component_info: List of dicts with info about each component
    """
    connectivity_map = {
        "face": 1,
        "face-edge": 2,
        "all": 3,
    }
    connectivity_value = connectivity_map.get(connectivity, 3)
    struct = generate_binary_structure(seg.ndim, connectivity_value)

    labeled_seg, num_components = label(seg, structure=struct)
    voxel_volume = voxel_spacing**3

    # One bincount and one lookup table: no pass over the volume per component
    counts = np.bincount(labeled_seg.ravel(), minlength=num_components + 1)
    keep = _keep_mask(counts, voxel_volume, min_size, max_size, keep_largest)
    keep = _apply_skeleton_length(labeled_seg, keep, voxel_spacing, min_skeleton_length)

    component_info = [
        {
            "component_id": component_id,
            "voxels": int(counts[component_id]),
            "volume": int(counts[component_id]) * voxel_volume,
            "kept": bool(keep[component_id]),
        }
        for component_id in range(1, num_components + 1)
    ]
    num_kept = int(keep[1:].sum())
    return keep[labeled_seg].astype(np.uint8), num_kept, num_components - num_kept, component_info


def _filter_instances(
    instances: np.ndarray,
    voxel_spacing: float,
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    keep_largest: Optional[int] = None,
    min_skeleton_length: Optional[float] = None,
) -> Tuple[np.ndarray, int, int]:
    """Drop whole instances of an instance segmentation by volume (and skeleton length); kept IDs are unchanged."""
    counts = np.bincount(instances.ravel()) if instances.size else np.zeros(1, dtype=int)
    keep = _keep_mask(counts, voxel_spacing**3, min_size, max_size, keep_largest)
    keep = _apply_skeleton_length(instances, keep, voxel_spacing, min_skeleton_length)
    present = int((counts[1:] > 0).sum())
    num_kept = int(keep.sum())
    return np.where(keep[instances], instances, 0).astype(instances.dtype), num_kept, present - num_kept


def filter_segmentation_components(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    connectivity: str = "all",
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    keep_largest: Optional[int] = None,
    min_skeleton_length: Optional[float] = None,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Filter connected components in a segmentation by size.

    A binary segmentation is filtered by its connected components, a multilabel one by the components of each label
    (labels are kept), and an instance segmentation by whole instances, whose IDs are kept so that picks and
    filaments with the same IDs still match.

    Args:
        segmentation: Input CopickSegmentation object.
        run: CopickRun object.
        object_name: Name for the output segmentation.
        session_id: Session ID for the output segmentation.
        user_id: User ID for the output segmentation.
        connectivity: Connectivity for connected components.
        min_size: Minimum component volume in cubic angstroms (Å³) to keep.
        max_size: Maximum component volume in cubic angstroms (Å³) to keep.
        keep_largest: If set, keep only the N largest components by voxel count.
        min_skeleton_length: Minimum skeleton length in angstroms to keep.
        **kwargs: Additional keyword arguments from lazy converter.

    Returns:
        Tuple of (CopickSegmentation object, stats dict) or None if operation failed.
    """
    from copick_utils.util.segmentations import new_segmentation, segmentation_type

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
        seg_type = segmentation_type(segmentation)
        filters = {
            "min_size": min_size,
            "max_size": max_size,
            "keep_largest": keep_largest,
            "min_skeleton_length": min_skeleton_length,
        }

        if seg_type == "panoptic":
            logger.error("filter-components does not filter panoptic segmentations; split them first")
            return None
        if seg_type == "instance":
            result_array, num_kept, num_removed = _filter_instances(seg_array, actual_voxel_spacing, **filters)
        elif seg_type == "multilabel":
            result_array = np.zeros_like(seg_array)
            num_kept = num_removed = 0
            for label_value in np.unique(seg_array[seg_array > 0]):
                kept, k, r, _ = _filter_components_by_size(
                    seg_array == label_value,
                    voxel_spacing=actual_voxel_spacing,
                    connectivity=connectivity,
                    **filters,
                )
                result_array[kept > 0] = label_value
                num_kept += k
                num_removed += r
        else:
            result_array, num_kept, num_removed, _ = _filter_components_by_size(
                seg_array.astype(bool),
                voxel_spacing=actual_voxel_spacing,
                connectivity=connectivity,
                **filters,
            )

        output_seg = new_segmentation(run, actual_voxel_spacing, object_name, session_id, user_id, seg_type)
        output_seg.from_numpy(result_array)

        stats = {
            "voxels_kept": int(np.count_nonzero(result_array)),
            "components_kept": num_kept,
            "components_removed": num_removed,
            "components_total": num_kept + num_removed,
        }
        logger.info(
            f"Filtered components: kept {stats['components_kept']}/{stats['components_total']}, "
            f"removed {stats['components_removed']} ({stats['voxels_kept']} voxels remaining)",
        )
        return output_seg, stats

    except Exception as e:
        logger.error(f"Error filtering segmentation components: {e}")
        return None


# Lazy batch converter for parallel discovery and processing
filter_components_lazy_batch = create_lazy_batch_converter(
    converter_func=filter_segmentation_components,
    task_description="Filtering components by size",
)
