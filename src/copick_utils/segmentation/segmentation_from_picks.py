import numpy as np
import zarr
from scipy.ndimage import zoom
import copick


def from_picks(pick, seg_volume, radius=10.0, label_value=1, voxel_spacing=10):
    """
    Paints picks into a segmentation volume as spheres.
    """
    def create_sphere(shape, center, radius, val):
        zc, yc, xc = center
        z, y, x = np.indices(shape)
        distance_sq = (x - xc)**2 + (y - yc)**2 + (z - zc)**2
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
        cx, cy, cz = point.location.x / voxel_spacing, point.location.y / voxel_spacing, point.location.z / voxel_spacing

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
        seg_volume[zLow:zHigh, yLow:yHigh, xLow:xHigh] = np.maximum(seg_volume[zLow:zHigh, yLow:yHigh, xLow:xHigh], sphere)

    return seg_volume


def downsample_to_exact_shape(array, target_shape):
    """
    Downsamples a 3D array to match the target shape using nearest-neighbor interpolation.
    Ensures that the resulting array has the exact target shape.
    """
    zoom_factors = [t / s for t, s in zip(target_shape, array.shape)]
    return zoom(array, zoom_factors, order=0)


def segmentation_from_picks(radius, painting_segmentation_name, run, voxel_spacing, tomo_type, pickable_object, pick_set, user_id="paintedPicks", session_id="0"):
    """
    Paints picks into the highest resolution scale and generates lower scales by downsampling.
    """
    # Fetch the tomogram and determine its multiscale structure
    tomogram = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type)
    if not tomogram:
        raise ValueError("Tomogram not found for the given parameters.")

    # Use copick to create a new segmentation if one does not exist
    segs = run.get_segmentations(user_id=user_id, session_id=session_id, is_multilabel=True, name=painting_segmentation_name, voxel_size=voxel_spacing)
    if len(segs) == 0:
        seg = run.new_segmentation(voxel_spacing, painting_segmentation_name, session_id, True, user_id=user_id)
    else:
        seg = segs[0]

    segmentation_group = zarr.open(seg.zarr(), mode="a")
    highest_res_name = "0"

    # Get the highest resolution dimensions and create a new array if necessary
    tomogram_zarr = zarr.open(tomogram.zarr(), "r")

    highest_res_shape = tomogram_zarr[highest_res_name].shape
    if highest_res_name not in segmentation_group:
        segmentation_group.create(highest_res_name, shape=highest_res_shape, dtype=np.uint16, overwrite=True)

    # Initialize or load the highest resolution array
    highest_res_seg = segmentation_group[highest_res_name][:]
    highest_res_seg.fill(0)

    # Paint picks into the highest resolution array
    highest_res_seg = from_picks(pick_set, highest_res_seg, radius, pickable_object.label, voxel_spacing)

    # Write back the highest resolution data
    segmentation_group[highest_res_name][:] = highest_res_seg

    # Downsample to create lower resolution scales
    multiscale_metadata = tomogram_zarr.attrs.get('multiscales', [{}])[0].get('datasets', [])
    for level_index, level_metadata in enumerate(multiscale_metadata):
        if level_index == 0:
            continue

        level_name = level_metadata.get("path", str(level_index))
        expected_shape = tuple(tomogram_zarr[level_name].shape)

        # Compute scaling factors relative to the highest resolution shape
        scaled_array = downsample_to_exact_shape(highest_res_seg, expected_shape)

        # Create/overwrite the Zarr array for this level
        segmentation_group.create_dataset(level_name, shape=expected_shape, data=scaled_array, dtype=np.uint16, overwrite=True)

        segmentation_group[level_name][:] = scaled_array

    return seg
