import numpy as np

def from_picks(pick, 
               seg_volume,
               radius: float = 10.0, 
               label_value: int = 1,
               voxel_spacing: float = 10):
    """
    Paints picks from a run into a multiscale segmentation array using copick, representing them as spheres (balls) 
    of the given radius.
    
    Parameters:
    pick (object): Copick object containing the `points` attribute, where each point represents a pick with `location` 
                   coordinates (`x`, `y`, `z`). These coordinates are used as the center of each sphere.
    seg_volume (numpy.ndarray): The 3D segmentation volume (numpy array) where the spheres are painted. 
                                The array should have dimensions (Z, Y, X).
    radius (float, optional): The radius of the spheres to be inserted, in physical units (not voxel units). 
    label_value (int, optional): The integer value used to label the sphere regions in the segmentation volume. 
    voxel_spacing (float, optional): The spacing of voxels in the segmentation volume, used to scale the radius of the spheres. 

    Returns:
    numpy.ndarray: The modified segmentation volume (`seg_volume`) with spheres inserted at the pick locations.
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
