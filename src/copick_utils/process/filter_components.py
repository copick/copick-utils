"""Filter connected components in segmentations by size."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from copick.util.log import get_logger
from scipy.ndimage import generate_binary_structure, label

if TYPE_CHECKING:
    from copick.models import CopickRoot, CopickRun, CopickSegmentation

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
    # Map connectivity string to numeric value
    connectivity_map = {
        "face": 1,
        "face-edge": 2,
        "all": 3,
    }
    connectivity_value = connectivity_map.get(connectivity, 3)

    # Define connectivity structure
    struct = generate_binary_structure(seg.ndim, connectivity_value)

    # Label connected components
    labeled_seg, num_components = label(seg, structure=struct)

    # Calculate voxel volume in cubic angstroms
    voxel_volume = voxel_spacing**3

    # Initialize output
    seg_filtered = np.zeros_like(seg, dtype=bool)

    component_info = []
    num_kept = 0
    num_removed = 0

    # Check each component
    for component_id in range(1, num_components + 1):
        # Extract this component
        component_mask = labeled_seg == component_id
        component_voxels = int(np.sum(component_mask))
        component_volume = component_voxels * voxel_volume

        # Apply size filtering
        passes_filter = True
        if min_size is not None and component_volume < min_size:
            passes_filter = False
        if max_size is not None and component_volume > max_size:
            passes_filter = False

        # Store information
        info = {
            "component_id": component_id,
            "voxels": component_voxels,
            "volume": component_volume,
            "kept": passes_filter,
        }
        component_info.append(info)

        # Keep or remove component
        if passes_filter:
            seg_filtered = np.logical_or(seg_filtered, component_mask)
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
    voxel_spacing: float,
    is_multilabel: bool = False,
    connectivity: str = "all",
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Filter connected components in a segmentation by size.

    Args:
        segmentation: Input CopickSegmentation object
        run: CopickRun object
        object_name: Name for the output segmentation
        session_id: Session ID for the output segmentation
        user_id: User ID for the output segmentation
        voxel_spacing: Voxel spacing for the output segmentation in angstroms
        is_multilabel: Whether the segmentation is multilabel
        connectivity: Connectivity for connected components (default: "all")
                     "face" = 6-connected, "face-edge" = 18-connected, "all" = 26-connected
        min_size: Minimum component volume in cubic angstroms (Å³) to keep (None = no minimum)
        max_size: Maximum component volume in cubic angstroms (Å³) to keep (None = no maximum)
        **kwargs: Additional keyword arguments

    Returns:
        Tuple of (CopickSegmentation object, stats dict) or None if operation failed.
        Stats dict contains 'voxels_kept', 'components_kept', 'components_removed'.
    """
    try:
        # Load segmentation array
        seg_array = segmentation.numpy()

        if seg_array is None:
            logger.error("Could not load segmentation data")
            return None

        if seg_array.size == 0:
            logger.error("Empty segmentation data")
            return None

        # Convert to boolean array
        bool_seg = seg_array.astype(bool)

        # Filter components
        result_array, num_kept, num_removed, component_info = _filter_components_by_size(
            bool_seg,
            voxel_spacing=voxel_spacing,
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


def _filter_components_worker(
    run: "CopickRun",
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str,
    voxel_spacing: float,
    connectivity: str,
    min_size: Optional[float],
    max_size: Optional[float],
    output_user_id: str,
    output_session_id: str,
    is_multilabel: bool,
    root: "CopickRoot",
) -> Dict[str, Any]:
    """Worker function for batch component filtering."""
    try:
        # Get segmentation
        segmentations = run.get_segmentations(
            name=segmentation_name,
            user_id=segmentation_user_id,
            session_id=segmentation_session_id,
            voxel_size=voxel_spacing,
        )

        if not segmentations:
            return {"processed": 0, "errors": [f"No segmentation found for {run.name}"]}

        segmentation = segmentations[0]

        # Filter components
        result = filter_segmentation_components(
            segmentation=segmentation,
            run=run,
            object_name=segmentation_name,
            session_id=output_session_id,
            user_id=output_user_id,
            voxel_spacing=voxel_spacing,
            is_multilabel=is_multilabel,
            connectivity=connectivity,
            min_size=min_size,
            max_size=max_size,
        )

        if result is None:
            return {"processed": 0, "errors": [f"Failed to filter components for {run.name}"]}

        output_seg, stats = result

        return {
            "processed": 1,
            "errors": [],
            "voxels_kept": stats["voxels_kept"],
            "components_kept": stats["components_kept"],
            "components_removed": stats["components_removed"],
            "components_total": stats["components_total"],
        }

    except Exception as e:
        return {"processed": 0, "errors": [f"Error processing {run.name}: {e}"]}


def filter_components_batch(
    root: "CopickRoot",
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str,
    voxel_spacing: float,
    connectivity: str = "all",
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    output_user_id: str = "filter-components",
    output_session_id: str = "filtered",
    is_multilabel: bool = False,
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Batch filter connected components by size across multiple runs.

    Args:
        root: The copick root containing runs to process
        segmentation_name: Name of the segmentation to process
        segmentation_user_id: User ID of the segmentation to process
        segmentation_session_id: Session ID of the segmentation to process
        voxel_spacing: Voxel spacing in angstroms
        connectivity: Connectivity for connected components (default: "all")
        min_size: Minimum component volume in Å³ to keep (None = no minimum)
        max_size: Maximum component volume in Å³ to keep (None = no maximum)
        output_user_id: User ID for output segmentations
        output_session_id: Session ID for output segmentations
        is_multilabel: Whether the segmentation is multilabel
        run_names: List of run names to process. If None, processes all runs.
        workers: Number of worker processes

    Returns:
        Dictionary with processing results and statistics
    """
    from copick.ops.run import map_runs

    runs_to_process = [run.name for run in root.runs] if run_names is None else run_names

    results = map_runs(
        callback=_filter_components_worker,
        root=root,
        runs=runs_to_process,
        workers=workers,
        task_desc="Filtering components by size",
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        segmentation_session_id=segmentation_session_id,
        voxel_spacing=voxel_spacing,
        connectivity=connectivity,
        min_size=min_size,
        max_size=max_size,
        output_user_id=output_user_id,
        output_session_id=output_session_id,
        is_multilabel=is_multilabel,
    )

    return results
