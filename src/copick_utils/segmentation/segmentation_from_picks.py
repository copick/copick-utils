import numpy as np
import zarr
import copick

def segmentation_from_picks(radius, painting_segmentation_name, run, voxel_spacing, tomo_type, pickable_object, pick_set, user_id="paintedPicks", session_id=0):
    """
    Paints picks from a run into a multiscale Zarr segmentation array, representing them as spheres (balls) of the given radius.

    Parameters:
    radius (int): The radius of the spherical ball to paint.
    painting_segmentation_name (str): Name of the segmentation layer to create or use.
    run (object): The run object containing picks.
    voxel_spacing (float): The voxel spacing for scaling pick locations.
    tomo_type (str): Type of tomogram to use for segmentation (e.g., denoised).
    pickable_object (object): The pickable object type from the Copick configuration.
    pick_set (list): A list of picks containing the object type and location.
    user_id (str): The user ID for the segmentation session.
    session_id (str): The session ID for the segmentation session.

    Returns:
    zarr.Group: The multiscale painted segmentation array.
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

    tomogram_group = zarr.open(tomogram.zarr(), "r")
    multiscale_levels = list(tomogram_group.array_keys())
    
    # Create or open the segmentation group
    segmentation_group = zarr.open_group(f"{run.name}_{painting_segmentation_name}.zarr", mode="a")

    # Paint each pick as a ball at each multiscale level
    for level in multiscale_levels:
        level_data = tomogram_group[level]
        shape = level_data.shape

        if level not in segmentation_group:
            segmentation_group.create_dataset(level, shape=shape, dtype=np.uint16, fill_value=0)
        painting_seg_array = segmentation_group[level]

        scale_factor = tomogram_group.attrs.get('multiscales', [{}])[0].get('datasets', [{}])[int(level)].get('scale', 1)
        scaled_radius = int(radius / scale_factor)
        
        for pick in pick_set.points:
            z, y, x = int(pick.location.z / voxel_spacing / scale_factor), int(pick.location.y / voxel_spacing / scale_factor), int(pick.location.x / voxel_spacing / scale_factor)

            # Create the spherical ball with the scaled radius
            ball = create_ball((scaled_radius, scaled_radius, scaled_radius), scaled_radius)

            # Determine where the ball should be placed in the segmentation array
            z_min, z_max = max(0, z - scaled_radius), min(painting_seg_array.shape[0], z + scaled_radius + 1)
            y_min, y_max = max(0, y - scaled_radius), min(painting_seg_array.shape[1], y + scaled_radius + 1)
            x_min, x_max = max(0, x - scaled_radius), min(painting_seg_array.shape[2], x + scaled_radius + 1)

            # Determine the portion of the ball to apply based on the array bounds
            z_ball_min, z_ball_max = max(0, scaled_radius - z), min(2 * scaled_radius + 1, scaled_radius + painting_seg_array.shape[0] - z)
            y_ball_min, y_ball_max = max(0, scaled_radius - y), min(2 * scaled_radius + 1, scaled_radius + painting_seg_array.shape[1] - y)
            x_ball_min, x_ball_max = max(0, scaled_radius - x), min(2 * scaled_radius + 1, scaled_radius + painting_seg_array.shape[2] - x)

            # Apply the ball to the segmentation array
            mask = ball[z_ball_min:z_ball_max, y_ball_min:y_ball_max, x_ball_min:x_ball_max] == 1
            painting_seg_array[z_min:z_max, y_min:y_max, x_min:x_max][mask] = pickable_object.label

    return segmentation_group
