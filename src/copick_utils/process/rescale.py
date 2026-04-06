"""Rescale segmentations to a different voxel spacing."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import zarr
from copick.util.log import get_logger
from scipy.ndimage import zoom

from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickSegmentation

logger = get_logger(__name__)


def _rescale_array(
    array: np.ndarray,
    source_spacing: float,
    target_spacing: float,
    target_shape: Optional[Tuple[int, ...]] = None,
    order: int = 0,
) -> np.ndarray:
    """
    Rescale a 3D array from one voxel spacing to another.

    Args:
        array: Input 3D array (z, y, x).
        source_spacing: Source voxel spacing in angstroms.
        target_spacing: Target voxel spacing in angstroms.
        target_shape: Explicit target shape. If None, computed from the spacing ratio.
        order: Interpolation order (0=nearest-neighbor, 1=linear).

    Returns:
        Rescaled array with the target shape.
    """
    if target_shape is None:
        # Check if spacings are effectively identical
        if abs(source_spacing - target_spacing) < 1e-6:
            return array.copy()

        # Compute target shape from spacing ratio
        scale_factor = source_spacing / target_spacing
        target_shape = tuple(max(1, round(s * scale_factor)) for s in array.shape)

    zoom_factors = [t / s for t, s in zip(target_shape, array.shape)]
    return zoom(array, zoom_factors, order=order)


def _get_tomogram_shape(
    run: "CopickRun",
    target_voxel_spacing: float,
    tomo_type: str,
) -> Optional[Tuple[int, ...]]:
    """
    Get the shape of a tomogram at the target voxel spacing.

    Args:
        run: CopickRun to search.
        target_voxel_spacing: Voxel spacing to look up.
        tomo_type: Tomogram type (e.g. 'wbp').

    Returns:
        Tomogram shape as (z, y, x) tuple, or None if not found.
    """
    voxel_spacing_obj = run.get_voxel_spacing(target_voxel_spacing)
    if voxel_spacing_obj is None:
        return None

    tomogram = voxel_spacing_obj.get_tomogram(tomo_type)
    if tomogram is None:
        return None

    tomo_zarr = zarr.open(tomogram.zarr(), "r")
    return tuple(tomo_zarr["0"].shape)


def rescale_segmentation(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    target_voxel_spacing: float,
    tomo_type: Optional[str] = None,
    order: int = 0,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Rescale a segmentation to a different voxel spacing.

    Args:
        segmentation: Input CopickSegmentation object.
        run: CopickRun object.
        object_name: Name for the output segmentation.
        session_id: Session ID for the output segmentation.
        user_id: User ID for the output segmentation.
        target_voxel_spacing: Target voxel spacing in angstroms.
        tomo_type: Tomogram type for shape reference. When provided, output shape
            matches the tomogram at the target spacing exactly.
        order: Interpolation order (0=nearest-neighbor, 1=linear).
        **kwargs: Additional keyword arguments from lazy converter.

    Returns:
        Tuple of (CopickSegmentation, stats dict) or None if operation failed.
    """
    try:
        seg_array = segmentation.numpy()

        if seg_array is None:
            logger.error("Could not load segmentation data")
            return None

        if seg_array.size == 0:
            logger.error("Empty segmentation data")
            return None

        source_spacing = segmentation.voxel_size
        scale_factor = source_spacing / target_voxel_spacing

        # Determine target shape
        target_shape = None
        if tomo_type is not None:
            target_shape = _get_tomogram_shape(run, target_voxel_spacing, tomo_type)
            if target_shape is None:
                logger.warning(
                    f"No tomogram '{tomo_type}' found at {target_voxel_spacing} Å, "
                    f"computing target shape from scale factor",
                )

        # Warn about large scale factors
        if scale_factor > 10:
            logger.warning(
                f"Large upscale factor ({scale_factor:.1f}x) — output array will be "
                f"~{scale_factor**3:.0f}x larger than input",
            )

        # Rescale
        input_labels = set(np.unique(seg_array))
        result_array = _rescale_array(seg_array, source_spacing, target_voxel_spacing, target_shape, order)
        output_labels = set(np.unique(result_array))

        labels_preserved = output_labels <= input_labels
        if not labels_preserved:
            logger.warning(
                f"New label values introduced by interpolation: {output_labels - input_labels}",
            )

        # Create output segmentation
        output_seg = run.new_segmentation(
            name=object_name,
            user_id=user_id,
            session_id=session_id,
            is_multilabel=segmentation.is_multilabel,
            voxel_size=target_voxel_spacing,
            exist_ok=True,
        )

        output_seg.from_numpy(result_array)

        stats = {
            "rescaled": 1,
            "labels_preserved": 1 if labels_preserved else 0,
        }
        logger.info(
            f"Rescaled segmentation: {seg_array.shape} @ {source_spacing} Å → "
            f"{result_array.shape} @ {target_voxel_spacing} Å "
            f"(scale={scale_factor:.2f}x, labels={'preserved' if labels_preserved else 'modified'})",
        )
        return output_seg, stats

    except Exception as e:
        logger.error(f"Error rescaling segmentation: {e}")
        return None


# Lazy batch converter for parallel discovery and processing
rescale_lazy_batch = create_lazy_batch_converter(
    converter_func=rescale_segmentation,
    task_description="Rescaling segmentations",
)
