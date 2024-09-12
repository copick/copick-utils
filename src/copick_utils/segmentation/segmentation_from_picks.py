import numpy as np
import zarr
import copick

def segmentation_from_picks(radius, painting_segmentation_name, run, voxel_spacing, tomo_type, pickable_object, pick_set, user_id="paintedPicks", session_id="0"):
    """
    Paints picks from a run into a multiscale segmentation array using copick, representing them as spheres (balls) 
    of the given radius. The painting is done in memory using NumPy arrays, and only written to Zarr at the end for each multiscale level.
    """
    def create_ball(center, radius):
        """Create a 3D ball with the specified radius."""
        zc, yc, xc = center
        shape = (2 * radius + 1, 2 * radius + 1, 2 * radius + 1)
        ball = np.zeros(shape, dtype=np.uint8)

        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    if np.linalg.norm(np.array([z, y, x]) - np.array([radius, radius, radius])) <= radius:
                        ball[z, y, x] = 1
        return ball

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
        
        # Paint each pick
        for pick in pick_set.points:
            z, y, x = int(pick.location.z / voxel_spacing / scale_factor), int(pick.location.y / voxel_spacing / scale_factor), int(pick.location.x / voxel_spacing / scale_factor)

            # Create the spherical ball with the scaled radius
            ball = create_ball((scaled_radius, scaled_radius, scaled_radius), scaled_radius)

            # Determine where the ball should be placed in the segmentation array
            z_min, z_max = max(0, z - scaled_radius), min(painting_seg_array.shape[0], z + scaled_radius + 1)
            y_min, y_max = max(0, y - scaled_radius), min(painting_seg_array.shape[1], y + scaled_radius + 1)
            x_min, x_max = max(0, x - scaled_radius), min(painting_seg_array.shape[2], x + scaled_radius + 1)

            # Skip if any of the ranges have a size of 0
            if z_min >= z_max or y_min >= y_max or x_min >= x_max:
                continue

            # Calculate the actual size of the region we are painting in the segmentation array
            z_size = z_max - z_min
            y_size = y_max - y_min
            x_size = x_max - x_min

            # Adjust the ball dimensions to match the size of the region in the segmentation array
            z_ball_min, z_ball_max = scaled_radius - (z - z_min), scaled_radius + (z_max - z)
            y_ball_min, y_ball_max = scaled_radius - (y - y_min), scaled_radius + (y_max - y)
            x_ball_min, x_ball_max = scaled_radius - (x - x_min), scaled_radius + (x_max - x)

            # Ensure the mask dimensions match the segmentation array subarray
            ball_subarray = ball[z_ball_min:z_ball_max, y_ball_min:y_ball_max, x_ball_min:x_ball_max]
            mask = ball_subarray == 1

            if painting_seg_array[z_min:z_max, y_min:y_max, x_min:x_max].shape != mask.shape:
                raise ValueError(f"Shape mismatch between segmentation array {painting_seg_array[z_min:z_max, y_min:y_max, x_min:x_max].shape} and mask {mask.shape}")

            # Apply the ball to the segmentation array in memory
            painting_seg_array[z_min:z_max, y_min:y_max, x_min:x_max][mask] = pickable_object.label

        # Once all picks are painted at this level, write the array to the Zarr store
        segmentation_group[level] = painting_seg_array

    return seg
