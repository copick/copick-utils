from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import zarr
from copick.util.log import get_logger

from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickObject, CopickPicks, CopickRun, CopickSegmentation

logger = get_logger(__name__)


def from_picks(
    pick: "CopickPicks",
    seg_volume: np.ndarray,
    radius: float = 10.0,
    label_value: int = 1,
    voxel_spacing: float = 10,
) -> np.ndarray:
    """Paint picks into a segmentation volume as spheres.

    Each pick location is converted from physical units to voxel indices and a
    solid sphere of `radius` is written into `seg_volume` with `label_value`.

    Args:
        pick: Copick picks whose `points` each carry a `location` with x, y, z
            coordinates in physical units.
        seg_volume: 3D segmentation volume to paint into, shape (Z, Y, X).
        radius: Sphere radius in physical units (angstroms), not voxels.
        label_value: Integer label written into the sphere regions.
        voxel_spacing: Voxel spacing of `seg_volume`, used to scale the radius.

    Returns:
        The modified segmentation volume with spheres painted at the pick
        locations.
    """

    def create_sphere(shape, center, radius, val):
        zc, yc, xc = center
        z, y, x = np.indices(shape)
        distance_sq = (x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2
        sphere = np.zeros(shape, dtype=np.float32)
        sphere[distance_sq <= radius**2] = val
        return sphere

    def get_relative_target_coordinates(center, delta, shape):
        low = max(int(np.floor(center - delta)), 0)
        high = min(int(np.ceil(center + delta + 1)), shape)
        return low, high

    # Adjust radius for voxel spacing
    radius_voxel = max(radius / voxel_spacing, 1)
    delta = int(np.ceil(radius_voxel))

    # Paint each pick as a sphere
    for point in pick.points:
        # Convert the pick's location from angstroms to voxel units
        cx, cy, cz = (
            point.location.x / voxel_spacing,
            point.location.y / voxel_spacing,
            point.location.z / voxel_spacing,
        )

        # Calculate subarray bounds
        xLow, xHigh = get_relative_target_coordinates(cx, delta, seg_volume.shape[2])
        yLow, yHigh = get_relative_target_coordinates(cy, delta, seg_volume.shape[1])
        zLow, zHigh = get_relative_target_coordinates(cz, delta, seg_volume.shape[0])

        # Subarray shape
        subarray_shape = (zHigh - zLow, yHigh - yLow, xHigh - xLow)
        if any(dim <= 0 for dim in subarray_shape):
            continue

        # Compute the local center of the sphere within the subarray
        local_center = (cz - zLow, cy - yLow, cx - xLow)
        sphere = create_sphere(subarray_shape, local_center, radius_voxel, label_value)

        # Assign Sphere to Segmentation Target Volume
        seg_volume[zLow:zHigh, yLow:yHigh, xLow:xHigh] = np.maximum(
            seg_volume[zLow:zHigh, yLow:yHigh, xLow:xHigh],
            sphere,
        )

    return seg_volume


def _create_segmentation_from_picks_legacy(
    radius: float,
    painting_segmentation_name: str,
    run: "CopickRun",
    voxel_spacing: float,
    tomo_type: str,
    pickable_object: "CopickObject",
    pick_set: "CopickPicks",
    user_id: str = "paintedPicks",
    session_id: str = "0",
) -> "CopickSegmentation":
    """
    Paints picks from a run into a multiscale segmentation array, representing them as spheres in 3D space.

    Parameters:
    -----------
    radius : float
        Radius of the spheres in physical units.
    painting_segmentation_name : str
        The name of the segmentation dataset to be created or modified.
    run : copick.Run
        The current Copick run object.
    voxel_spacing : float
        The spacing of the voxels in the tomogram data.
    tomo_type : str
        The type of tomogram to retrieve.
    pickable_object : copick.models.CopickObject
        The object that defines the label value to be used in segmentation.
    pick_set : copick.models.CopickPicks
        The set of picks containing the locations to paint spheres.
    user_id : str, optional
        The ID of the user creating the segmentation. Default is "paintedPicks".
    session_id : str, optional
        The session ID for this segmentation. Default is "0".

    Returns:
    --------
    copick.Segmentation
        The created or modified segmentation object.
    """
    # Fetch the tomogram and determine its multiscale structure
    tomogram = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type)
    if not tomogram:
        raise ValueError("Tomogram not found for the given parameters.")

    # Use copick to create a new segmentation if one does not exist
    segs = run.get_segmentations(
        user_id=user_id,
        session_id=session_id,
        is_multilabel=True,
        name=painting_segmentation_name,
        voxel_size=voxel_spacing,
    )
    if len(segs) == 0:
        seg = run.new_segmentation(voxel_spacing, painting_segmentation_name, session_id, True, user_id=user_id)
    else:
        seg = segs[0]

    # Paint the picks into a fresh full-resolution label volume.
    tomogram_zarr = zarr.open(tomogram.zarr(), "r")
    highest_res_shape = tomogram_zarr["0"].shape
    highest_res_seg = np.zeros(highest_res_shape, dtype=np.uint16)
    highest_res_seg = from_picks(pick_set, highest_res_seg, radius, pickable_object.label, voxel_spacing)

    # Write through the copick API so the OME-Zarr ``multiscales`` group attributes
    # (axes + coordinate transformations) are written alongside the data. The previous
    # implementation created the level datasets directly and omitted these attributes,
    # producing a segmentation that copick could read but OME-Zarr clients (e.g.
    # ChimeraX) could not. Written single-level, matching copick's default and every
    # other converter.
    seg.from_numpy(highest_res_seg, dtype=np.uint16)

    return seg


def segmentation_from_picks(
    picks: "CopickPicks",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    radius: float,
    voxel_spacing: float,
    tomo_type: str = "wbp",
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Convert CopickPicks to a segmentation by painting spheres.

    Args:
        picks: CopickPicks object to convert
        run: CopickRun object
        object_name: Name for the output segmentation
        session_id: Session ID for the output segmentation
        user_id: User ID for the output segmentation
        radius: Radius of the spheres in physical units
        voxel_spacing: Voxel spacing for the segmentation
        tomo_type: Type of tomogram to use as reference

    Returns:
        Tuple of (CopickSegmentation object, stats dict) or None if creation failed.
        Stats dict contains 'points_converted' and 'voxels_created'.
    """
    try:
        # Get the pickable object for label information
        root = run.root
        pickable_object = root.get_object(picks.pickable_object_name)
        if not pickable_object:
            logger.error(f"Object '{picks.pickable_object_name}' not found in config")
            return None

        if not picks.points:
            logger.error("No points found in pick set")
            return None

        # Create segmentation using the legacy function
        seg = _create_segmentation_from_picks_legacy(
            radius=radius,
            painting_segmentation_name=object_name,
            run=run,
            voxel_spacing=voxel_spacing,
            tomo_type=tomo_type,
            pickable_object=pickable_object,
            pick_set=picks,
            user_id=user_id,
            session_id=session_id,
        )

        # Calculate statistics
        # For now, we don't have easy access to the actual voxel count, so we estimate
        # based on number of spheres and their volume
        sphere_volume_voxels = (4 / 3) * np.pi * (radius / voxel_spacing) ** 3
        estimated_voxels = int(len(picks.points) * sphere_volume_voxels)

        stats = {
            "points_converted": len(picks.points),
            "voxels_created": estimated_voxels,
        }
        logger.info(f"Created segmentation from {stats['points_converted']} picks")
        return seg, stats

    except Exception as e:
        logger.error(f"Error creating segmentation: {e}")
        return None


# Lazy batch converter for new architecture
segmentation_from_picks_lazy_batch = create_lazy_batch_converter(
    converter_func=segmentation_from_picks,
    task_description="Converting picks to segmentations",
)
