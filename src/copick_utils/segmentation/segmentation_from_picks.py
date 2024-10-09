import numpy as np
import zarr
import copick

def from_picks(pick, 
               seg_volume,
               radius: float = 10.0, 
               label_value: int = 1,
               voxel_spacing: float = 10):
    """
    Paints picks into a segmentation volume as spheres.

    Parameters:
    -----------
    pick : copick.models.CopickPicks
        Copick object containing `points`, where each point has a `location` attribute with `x`, `y`, `z` coordinates.
    seg_volume : numpy.ndarray
        3D segmentation volume (numpy array) where the spheres are painted. Shape should be (Z, Y, X).
    radius : float, optional
        The radius of the spheres to be inserted in physical units (not voxel units). Default is 10.0.
    label_value : int, optional
        The integer value used to label the sphere regions in the segmentation volume. Default is 1.
    voxel_spacing : float, optional
        The spacing of voxels in the segmentation volume, used to scale the radius of the spheres. Default is 10.

    Returns:
    --------
    numpy.ndarray
        The modified segmentation volume with spheres inserted at pick locations.
    """
        
    def create_sphere(shape, center, radius, val):
        """Creates a 3D sphere within the given shape, centered at the given coordinates."""
        zc, yc, xc = center
        z, y, x = np.indices(shape)
        
        # Compute squared distance from the center
        distance_sq = (x - xc)**2 + (y - yc)**2 + (z - zc)**2
        
        # Create a mask for points within the sphere
        sphere = np.zeros(shape, dtype=np.float32)
        sphere[distance_sq <= radius**2] = val
        return sphere

    def get_relative_target_coordinates(center, delta, shape):
        """
        Calculate the low and high index bounds for placing a sphere within a 3D volume, 
        ensuring that the indices are clamped to the valid range of the volume dimensions.
        """

        low = max(int(np.floor(center) - delta), 0)
        high = min(int(np.ceil(center) + delta + 1), shape)

        return low, high

    # Adjust radius for voxel spacing
    radius_voxel = radius / voxel_spacing
    delta = int(np.ceil(radius_voxel))

    # Get volume dimensions
    vol_shape_x, vol_shape_y, vol_shape_z = seg_volume.shape

    # Paint each pick as a sphere
    for pick in pick.points:
        
        # Adjust the pick's location for voxel spacing
        cx, cy, cz = pick.location.z / voxel_spacing, pick.location.y / voxel_spacing, pick.location.x / voxel_spacing

        # Calculate subarray bounds, clamped to the valid volume dimensions
        xLow, xHigh = get_relative_target_coordinates(cx, delta, vol_shape_x)
        yLow, yHigh = get_relative_target_coordinates(cy, delta, vol_shape_y)
        zLow, zHigh = get_relative_target_coordinates(cz, delta, vol_shape_z)

        # Subarray shape
        subarray_shape = (xHigh - xLow, yHigh - yLow, zHigh - zLow)

        # Compute the local center of the sphere within the subarray
        local_center = (cx - xLow, cy - yLow, cz - zLow)

        # Create the sphere
        sphere = create_sphere(subarray_shape, local_center, radius_voxel, label_value)

        # Assign Sphere to Segmentation Target Volume
        seg_volume[xLow:xHigh, yLow:yHigh, zLow:zHigh] = np.maximum(seg_volume[xLow:xHigh, yLow:yHigh, zLow:zHigh], sphere)

    return seg_volume

def segmentation_from_picks(radius, painting_segmentation_name, run, voxel_spacing, tomo_type, pickable_object, pick_set, user_id="paintedPicks", session_id="0"):
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
    segs = run.get_segmentations(user_id=user_id, session_id=session_id, is_multilabel=True, name=painting_segmentation_name, voxel_size=voxel_spacing)
    if len(segs) == 0:
        seg = run.new_segmentation(voxel_spacing, painting_segmentation_name, session_id, True, user_id=user_id)
    else:
        seg = segs[0]

    segmentation_group = zarr.open_group(seg.path, mode="a")
    
    # Ensure that the tomogram Zarr is available
    tomogram_group = zarr.open(tomogram.zarr(), "r")
    multiscale_levels = list(tomogram_group.array_keys())

    # Iterate through multiscale levels to paint picks as spheres
    for level in multiscale_levels:
        level_data = tomogram_group[level]
        shape = level_data.shape

        # Load or initialize the segmentation array in memory for the current level
        if level in segmentation_group:
            painting_seg_array = np.array(segmentation_group[level])
        else:
            painting_seg_array = np.zeros(shape, dtype=np.uint16)

        scale_factor = tomogram_group.attrs.get('multiscales', [{}])[0].get('datasets', [{}])[int(level)].get('scale', 1)
        scaled_radius = int(radius / scale_factor)        

        # Once all picks are painted at this level, write the array to the Zarr store
        segmentation_group[level] = from_picks(pick_set, painting_seg_array, scaled_radius, pickable_object.label, voxel_spacing)

    return seg
