from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
from copick.util.log import get_logger
from skimage.measure import regionprops
from skimage.morphology import ball, binary_dilation, binary_erosion
from skimage.segmentation import watershed

from copick_utils.converters.converter_common import (
    create_batch_converter,
    create_batch_worker,
)
from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickPicks, CopickRun, CopickSegmentation

logger = get_logger(__name__)


def _extract_centroids_from_segmentation_array(
    segmentation: np.ndarray,
    segmentation_idx: int,
    maxima_filter_size: int,
    min_particle_size: int,
    max_particle_size: int,
    voxel_spacing: float,
) -> Optional[np.ndarray]:
    """
    Extract centroids from a segmentation array.

    Args:
        segmentation: Multilabel segmentation array.
        segmentation_idx: The specific label from the segmentation to process.
        maxima_filter_size: Size of the maximum detection filter.
        min_particle_size: Minimum size threshold for particles.
        max_particle_size: Maximum size threshold for particles.
        voxel_spacing: The voxel spacing used to scale pick locations.

    Returns:
        Array of centroid positions or None if no centroids found.
    """
    # Create a binary mask for the specific segmentation label
    binary_mask = (segmentation == segmentation_idx).astype(int)

    # Skip if the segmentation label is not present
    if np.sum(binary_mask) == 0:
        logger.warning(f"No segmentation with label {segmentation_idx} found")
        return None

    # Structuring element for erosion and dilation
    struct_elem = ball(1)
    eroded = binary_erosion(binary_mask, struct_elem)
    dilated = binary_dilation(eroded, struct_elem)

    # Distance transform and local maxima detection
    distance = ndi.distance_transform_edt(dilated)
    local_max = distance == ndi.maximum_filter(
        distance,
        footprint=np.ones((maxima_filter_size, maxima_filter_size, maxima_filter_size)),
    )

    # Watershed segmentation
    markers, _ = ndi.label(local_max)
    watershed_labels = watershed(-distance, markers, mask=dilated)

    # Extract region properties and filter based on particle size
    all_centroids = []
    for region in regionprops(watershed_labels):
        if min_particle_size <= region.area <= max_particle_size:
            all_centroids.append(region.centroid)

    if all_centroids:
        # Convert to positions (Z, Y, X) -> (X, Y, Z) and scale by voxel spacing
        positions = np.array(all_centroids)[:, [2, 1, 0]] * voxel_spacing
        return positions
    else:
        logger.warning(f"No valid centroids found for label {segmentation_idx}")
        return None


def picks_from_segmentation(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    segmentation_idx: int,
    maxima_filter_size: int = 9,
    min_particle_size: int = 1000,
    max_particle_size: int = 50000,
) -> Optional[Tuple["CopickPicks", Dict[str, int]]]:
    """
    Convert a CopickSegmentation to picks by extracting centroids.

    Args:
        segmentation: CopickSegmentation object to convert
        run: CopickRun object
        object_name: Name for the output pick object
        session_id: Session ID for the output picks
        user_id: User ID for the output picks
        segmentation_idx: The specific label from the segmentation to process
        maxima_filter_size: Size of the maximum detection filter
        min_particle_size: Minimum size threshold for particles
        max_particle_size: Maximum size threshold for particles

    Returns:
        Tuple of (CopickPicks object, stats dict) or None if creation failed.
        Stats dict contains 'points_created'.
    """
    try:
        # Load the segmentation array
        segmentation_array = segmentation.numpy()

        if segmentation_array is None or segmentation_array.size == 0:
            logger.error("Empty or invalid segmentation volume")
            return None

        # Get voxel spacing from segmentation
        voxel_spacing = segmentation.voxel_size

        # Extract centroids
        positions = _extract_centroids_from_segmentation_array(
            segmentation_array,
            segmentation_idx,
            maxima_filter_size,
            min_particle_size,
            max_particle_size,
            voxel_spacing,
        )

        if positions is None:
            logger.error("No centroids extracted from segmentation")
            return None

        # Create pick set and store positions
        pick_set = run.new_picks(object_name, session_id, user_id, exist_ok=True)
        pick_set.from_numpy(positions=positions)
        pick_set.store()

        stats = {"points_created": len(positions)}
        logger.info(f"Created {stats['points_created']} picks from segmentation")
        return pick_set, stats

    except Exception as e:
        logger.error(f"Error creating picks: {e}")
        return None


# Create worker function using common infrastructure
_picks_from_segmentation_worker = create_batch_worker(picks_from_segmentation, "picks", "segmentation", min_points=0)


# Create batch converter using common infrastructure
picks_from_segmentation_batch = create_batch_converter(
    picks_from_segmentation,
    "Converting segmentations to picks",
    "picks",
    "segmentation",
    min_points=0,
)

# Lazy batch converter for new architecture
picks_from_segmentation_lazy_batch = create_lazy_batch_converter(
    converter_func=picks_from_segmentation,
    task_description="Converting segmentations to picks",
)
