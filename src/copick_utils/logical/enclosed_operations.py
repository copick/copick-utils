"""Operations for finding and absorbing enclosed segmentation components."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
from copick.util.log import get_logger
from scipy.ndimage import binary_dilation, generate_binary_structure, label

from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickSegmentation

logger = get_logger(__name__)


def _find_enclosed_components(
    seg_inner: np.ndarray,
    seg_outer: np.ndarray,
    voxel_spacing: float,
    margin: int = 1,
    connectivity: str = "all",
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
) -> Tuple[np.ndarray, int, list]:
    """
    Find connected components in seg_inner that are fully surrounded by seg_outer.

    Args:
        seg_inner: Binary mask of inner segmentation (numpy array)
        seg_outer: Binary mask of outer segmentation (numpy array)
        voxel_spacing: Voxel spacing in angstroms
        margin: Number of voxels to dilate for checking surrounding (default: 1)
        connectivity: Connectivity for connected components (default: "all")
                     "face" = face connectivity (6-connected in 3D)
                     "face-edge" = face+edge connectivity (18-connected in 3D)
                     "all" = face+edge+corner connectivity (26-connected in 3D)
        min_size: Minimum component volume in cubic angstroms (Å³) to consider (None = no minimum)
        max_size: Maximum component volume in cubic angstroms (Å³) to consider (None = no maximum)

    Returns:
        Tuple of (updated_outer_seg, num_added, component_info)
        - updated_outer_seg: Updated outer segmentation with enclosed components added
        - num_added: Number of components that were added
        - component_info: List of dicts with info about each component
    """
    # Create a copy to avoid modifying the original
    seg_outer_updated = seg_outer.copy()

    # Map connectivity string to numeric value
    connectivity_map = {
        "face": 1,
        "face-edge": 2,
        "all": 3,
    }
    connectivity_value = connectivity_map.get(connectivity, 3)

    # Define connectivity structure
    struct = generate_binary_structure(seg_inner.ndim, connectivity_value)

    # Label connected components in inner segmentation
    labeled_inner, num_components = label(seg_inner, structure=struct)

    # Calculate voxel volume in cubic angstroms
    voxel_volume = voxel_spacing**3

    component_info = []
    num_added = 0

    # Check each component
    for component_id in range(1, num_components + 1):
        # Extract this component
        component_mask = labeled_inner == component_id
        component_voxels = int(np.sum(component_mask))
        component_volume = component_voxels * voxel_volume

        # Apply size filtering (in cubic angstroms)
        passes_size_filter = True
        if min_size is not None and component_volume < min_size:
            passes_size_filter = False
        if max_size is not None and component_volume > max_size:
            passes_size_filter = False

        # Dilate the component
        dilated_component = binary_dilation(component_mask, structure=struct, iterations=margin)

        # Check if dilated component is fully contained in outer segmentation
        is_surrounded = bool(np.all(dilated_component <= seg_outer))

        # Decide whether to add
        should_add = is_surrounded and passes_size_filter

        # Store information
        info = {
            "component_id": component_id,
            "voxels": component_voxels,
            "volume": component_volume,
            "is_surrounded": is_surrounded,
            "passes_size_filter": passes_size_filter,
            "added": should_add,
        }
        component_info.append(info)

        # If surrounded and passes size filter, add to outer segmentation
        if should_add:
            seg_outer_updated = np.logical_or(seg_outer_updated, component_mask)
            num_added += 1

    return seg_outer_updated.astype(np.uint8), num_added, component_info


def segmentation_enclosed(
    segmentation1: "CopickSegmentation",
    segmentation2: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    voxel_spacing: float,
    is_multilabel: bool = False,
    margin: int = 1,
    connectivity: str = "all",
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Find enclosed components in segmentation1 (inner) that are surrounded by segmentation2 (outer),
    and add them to segmentation2.

    Args:
        segmentation1: Inner CopickSegmentation object (source of enclosed components)
        segmentation2: Outer CopickSegmentation object (absorbing segmentation)
        run: CopickRun object
        object_name: Name for the output segmentation
        session_id: Session ID for the output segmentation
        user_id: User ID for the output segmentation
        voxel_spacing: Voxel spacing for the output segmentation in angstroms
        is_multilabel: Whether the segmentation is multilabel
        margin: Number of voxels to dilate for checking surrounding (default: 1)
        connectivity: Connectivity for connected components (default: "all")
                     "face" = 6-connected, "face-edge" = 18-connected, "all" = 26-connected
        min_size: Minimum component volume in cubic angstroms (Å³) to consider (None = no minimum)
        max_size: Maximum component volume in cubic angstroms (Å³) to consider (None = no maximum)
        **kwargs: Additional keyword arguments

    Returns:
        Tuple of (CopickSegmentation object, stats dict) or None if operation failed.
        Stats dict contains 'voxels_created' and 'components_added'.
    """
    try:
        # Load segmentation arrays
        seg1_array = segmentation1.numpy()
        seg2_array = segmentation2.numpy()

        if seg1_array is None or seg2_array is None:
            logger.error("Could not load segmentation data")
            return None

        if seg1_array.size == 0 or seg2_array.size == 0:
            logger.error("Empty segmentation data")
            return None

        # Ensure arrays have the same shape
        if seg1_array.shape != seg2_array.shape:
            logger.error(f"Segmentation arrays must have the same shape: {seg1_array.shape} vs {seg2_array.shape}")
            return None

        # Check that segmentations have compatible voxel spacing
        if abs(segmentation1.voxel_size - segmentation2.voxel_size) > 1e-6:
            logger.warning(
                f"Segmentations have different voxel spacing: {segmentation1.voxel_size} vs {segmentation2.voxel_size}",
            )

        # Convert to boolean arrays
        bool1 = seg1_array.astype(bool)
        bool2 = seg2_array.astype(bool)

        # Find and add enclosed components
        result_array, num_added, component_info = _find_enclosed_components(
            bool1,
            bool2,
            voxel_spacing=voxel_spacing,
            margin=margin,
            connectivity=connectivity,
            min_size=min_size,
            max_size=max_size,
        )

        # Create output segmentation
        output_seg = run.new_segmentation(
            name=object_name,
            user_id=user_id,
            session_id=session_id,
            is_multilabel=is_multilabel,
            voxel_size=voxel_spacing,
            exist_ok=True,
        )

        # Store the result
        output_seg.from_numpy(result_array)

        stats = {
            "voxels_created": int(np.sum(result_array)),
            "components_added": num_added,
            "components_evaluated": len(component_info),
        }
        logger.info(
            f"Added {stats['components_added']}/{stats['components_evaluated']} enclosed components "
            f"({stats['voxels_created']} total voxels)",
        )
        return output_seg, stats

    except Exception as e:
        logger.error(f"Error performing segmentation enclosed operation: {e}")
        return None


# Lazy batch converter for new architecture
segmentation_enclosed_lazy_batch = create_lazy_batch_converter(
    converter_func=segmentation_enclosed,
    task_description="Finding and absorbing enclosed segmentation components",
)
