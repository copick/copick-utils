"""Filter connected components in segmentations by size."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
from copick.util.log import get_logger
from scipy.ndimage import generate_binary_structure, label

from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickSegmentation

logger = get_logger(__name__)


def _filter_components_by_size(
    seg: np.ndarray,
    voxel_spacing: float,
    connectivity: str = "all",
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
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

    # Use bincount for O(n) counting of all components in one pass
    counts = np.bincount(labeled_seg.ravel())

    seg_filtered = np.zeros_like(seg, dtype=bool)
    component_info = []
    num_kept = 0
    num_removed = 0

    for component_id in range(1, num_components + 1):
        component_voxels = int(counts[component_id])
        component_volume = component_voxels * voxel_volume

        passes_filter = True
        if min_size is not None and component_volume < min_size:
            passes_filter = False
        if max_size is not None and component_volume > max_size:
            passes_filter = False

        info = {
            "component_id": component_id,
            "voxels": component_voxels,
            "volume": component_volume,
            "kept": passes_filter,
        }
        component_info.append(info)

        if passes_filter:
            seg_filtered = np.logical_or(seg_filtered, labeled_seg == component_id)
            num_kept += 1
        else:
            num_removed += 1

    return seg_filtered.astype(np.uint8), num_kept, num_removed, component_info


def filter_segmentation_components(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    connectivity: str = "all",
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Filter connected components in a segmentation by size.

    Args:
        segmentation: Input CopickSegmentation object.
        run: CopickRun object.
        object_name: Name for the output segmentation.
        session_id: Session ID for the output segmentation.
        user_id: User ID for the output segmentation.
        connectivity: Connectivity for connected components.
        min_size: Minimum component volume in cubic angstroms (Å³) to keep.
        max_size: Maximum component volume in cubic angstroms (Å³) to keep.
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

        bool_seg = seg_array.astype(bool)

        result_array, num_kept, num_removed, component_info = _filter_components_by_size(
            bool_seg,
            voxel_spacing=actual_voxel_spacing,
            connectivity=connectivity,
            min_size=min_size,
            max_size=max_size,
        )

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
            "voxels_kept": int(np.sum(result_array)),
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
