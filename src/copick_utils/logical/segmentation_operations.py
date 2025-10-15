"""Boolean operations on segmentations (union, intersection, difference, exclusion)."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from copick.util.log import get_logger

from copick_utils.converters.converter_common import (
    create_batch_converter,
    create_batch_worker,
)
from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickSegmentation

logger = get_logger(__name__)


def _perform_segmentation_boolean_operation(
    seg1_array: np.ndarray,
    seg2_array: np.ndarray,
    operation: str,
) -> np.ndarray:
    """
    Perform boolean operation between two segmentation arrays.

    Args:
        seg1_array: First segmentation array (should be binary: 0 or 1)
        seg2_array: Second segmentation array (should be binary: 0 or 1)
        operation: Type of boolean operation ('union', 'difference', 'intersection', 'exclusion')

    Returns:
        Result segmentation array
    """
    # Ensure arrays have the same shape
    if seg1_array.shape != seg2_array.shape:
        raise ValueError(f"Segmentation arrays must have the same shape: {seg1_array.shape} vs {seg2_array.shape}")

    # Convert to boolean arrays
    bool1 = seg1_array.astype(bool)
    bool2 = seg2_array.astype(bool)

    if operation == "union":
        result = np.logical_or(bool1, bool2)
    elif operation == "difference":
        result = np.logical_and(bool1, np.logical_not(bool2))
    elif operation == "intersection":
        result = np.logical_and(bool1, bool2)
    elif operation == "exclusion":
        # Exclusion = (A or B) and not (A and B)
        result = np.logical_and(np.logical_or(bool1, bool2), np.logical_not(np.logical_and(bool1, bool2)))
    else:
        raise ValueError(f"Unknown boolean operation: {operation}")

    return result.astype(np.uint8)


def segmentation_boolean_operation(
    segmentation1: "CopickSegmentation",
    segmentation2: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    operation: str,
    voxel_spacing: float,
    is_multilabel: bool = False,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Perform boolean operation between two CopickSegmentation objects.

    Args:
        segmentation1: First CopickSegmentation object
        segmentation2: Second CopickSegmentation object
        run: CopickRun object
        object_name: Name for the output segmentation
        session_id: Session ID for the output segmentation
        user_id: User ID for the output segmentation
        operation: Type of boolean operation ('union', 'difference', 'intersection', 'exclusion')
        voxel_spacing: Voxel spacing for the output segmentation
        is_multilabel: Whether the segmentation is multilabel
        **kwargs: Additional keyword arguments

    Returns:
        Tuple of (CopickSegmentation object, stats dict) or None if operation failed.
        Stats dict contains 'voxels_created'.
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

        # Check that segmentations have compatible voxel spacing
        if abs(segmentation1.voxel_size - segmentation2.voxel_size) > 1e-6:
            logger.warning(
                f"Segmentations have different voxel spacing: {segmentation1.voxel_size} vs {segmentation2.voxel_size}",
            )

        # Perform boolean operation
        result_array = _perform_segmentation_boolean_operation(seg1_array, seg2_array, operation)

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

        stats = {"voxels_created": int(np.sum(result_array))}
        logger.info(f"Created {operation} segmentation with {stats['voxels_created']} voxels")
        return output_seg, stats

    except Exception as e:
        logger.error(f"Error performing segmentation {operation}: {e}")
        return None


# Individual operation functions
def segmentation_union(
    segmentation1: "CopickSegmentation",
    segmentation2: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    voxel_spacing: float,
    is_multilabel: bool = False,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """Union of two segmentations."""
    return segmentation_boolean_operation(
        segmentation1,
        segmentation2,
        run,
        object_name,
        session_id,
        user_id,
        "union",
        voxel_spacing,
        is_multilabel,
        **kwargs,
    )


def segmentation_difference(
    segmentation1: "CopickSegmentation",
    segmentation2: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    voxel_spacing: float,
    is_multilabel: bool = False,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """Difference of two segmentations (seg1 - seg2)."""
    return segmentation_boolean_operation(
        segmentation1,
        segmentation2,
        run,
        object_name,
        session_id,
        user_id,
        "difference",
        voxel_spacing,
        is_multilabel,
        **kwargs,
    )


def segmentation_intersection(
    segmentation1: "CopickSegmentation",
    segmentation2: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    voxel_spacing: float,
    is_multilabel: bool = False,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """Intersection of two segmentations."""
    return segmentation_boolean_operation(
        segmentation1,
        segmentation2,
        run,
        object_name,
        session_id,
        user_id,
        "intersection",
        voxel_spacing,
        is_multilabel,
        **kwargs,
    )


def segmentation_exclusion(
    segmentation1: "CopickSegmentation",
    segmentation2: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    voxel_spacing: float,
    is_multilabel: bool = False,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """Exclusive or (XOR) of two segmentations."""
    return segmentation_boolean_operation(
        segmentation1,
        segmentation2,
        run,
        object_name,
        session_id,
        user_id,
        "exclusion",
        voxel_spacing,
        is_multilabel,
        **kwargs,
    )


def segmentation_multi_union(
    segmentations: List["CopickSegmentation"],
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    voxel_spacing: float,
    is_multilabel: bool = False,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Perform N-way union of multiple segmentations.

    All input segmentations are converted to binary and combined using logical OR.

    Args:
        segmentations: List of CopickSegmentation objects (N≥2)
        run: CopickRun object
        object_name: Name for output segmentation
        session_id: Session ID for output
        user_id: User ID for output
        voxel_spacing: Voxel spacing for output
        is_multilabel: Whether output is multilabel (default: False for binary union)
        **kwargs: Additional arguments

    Returns:
        Tuple of (CopickSegmentation, stats) or None if failed
    """
    try:
        if len(segmentations) < 2:
            logger.error("Need at least 2 segmentations for N-way union")
            return None

        # Load all segmentation arrays
        arrays = []
        for i, seg in enumerate(segmentations):
            arr = seg.numpy()
            if arr is None or arr.size == 0:
                logger.error(f"Could not load segmentation {i+1} (session: {seg.session_id})")
                return None
            arrays.append(arr)

        # Verify all have same shape
        ref_shape = arrays[0].shape
        for i, arr in enumerate(arrays[1:], start=2):
            if arr.shape != ref_shape:
                logger.error(f"Shape mismatch in segmentation {i}: {arr.shape} vs {ref_shape}")
                return None

        # Perform N-way union (accumulative logical OR)
        result = arrays[0].astype(bool)
        for arr in arrays[1:]:
            result = np.logical_or(result, arr.astype(bool))

        result_array = result.astype(np.uint8)

        # Create output segmentation
        output_seg = run.new_segmentation(
            name=object_name,
            user_id=user_id,
            session_id=session_id,
            is_multilabel=is_multilabel,
            voxel_size=voxel_spacing,
            exist_ok=True,
        )

        output_seg.from_numpy(result_array)

        stats = {"voxels_created": int(np.sum(result_array))}
        logger.info(f"Created {len(segmentations)}-way union with {stats['voxels_created']} voxels")
        return output_seg, stats

    except Exception as e:
        logger.error(f"Error performing N-way union: {e}")
        return None


# Create batch workers for each operation
_segmentation_union_worker = create_batch_worker(segmentation_union, "segmentation", "segmentation", min_points=0)
_segmentation_difference_worker = create_batch_worker(
    segmentation_difference,
    "segmentation",
    "segmentation",
    min_points=0,
)
_segmentation_intersection_worker = create_batch_worker(
    segmentation_intersection,
    "segmentation",
    "segmentation",
    min_points=0,
)
_segmentation_exclusion_worker = create_batch_worker(
    segmentation_exclusion,
    "segmentation",
    "segmentation",
    min_points=0,
)

# Create batch converters
segmentation_union_batch = create_batch_converter(
    segmentation_union,
    "Computing segmentation unions",
    "segmentation",
    "segmentation",
    min_points=0,
    dual_input=True,
)

segmentation_difference_batch = create_batch_converter(
    segmentation_difference,
    "Computing segmentation differences",
    "segmentation",
    "segmentation",
    min_points=0,
    dual_input=True,
)

segmentation_intersection_batch = create_batch_converter(
    segmentation_intersection,
    "Computing segmentation intersections",
    "segmentation",
    "segmentation",
    min_points=0,
    dual_input=True,
)

segmentation_exclusion_batch = create_batch_converter(
    segmentation_exclusion,
    "Computing segmentation exclusions",
    "segmentation",
    "segmentation",
    min_points=0,
    dual_input=True,
)

# Lazy batch converters for new architecture
segmentation_union_lazy_batch = create_lazy_batch_converter(
    converter_func=segmentation_union,
    task_description="Computing segmentation unions",
)

segmentation_difference_lazy_batch = create_lazy_batch_converter(
    converter_func=segmentation_difference,
    task_description="Computing segmentation differences",
)

segmentation_intersection_lazy_batch = create_lazy_batch_converter(
    converter_func=segmentation_intersection,
    task_description="Computing segmentation intersections",
)

segmentation_exclusion_lazy_batch = create_lazy_batch_converter(
    converter_func=segmentation_exclusion,
    task_description="Computing segmentation exclusions",
)

# Lazy batch converter for N-way union
segmentation_multi_union_lazy_batch = create_lazy_batch_converter(
    converter_func=segmentation_multi_union,
    task_description="Computing N-way segmentation unions",
)
