import numpy as np
import zarr
from copick.models import CopickPoint


def grid_picker(pickable_obj, run, tomogram, grid_spacing_factor, session_id="0", user_id="gridPicker"):
    """
    Creates a grid of picks for a pickable object based on a tomogram and grid spacing factor.

    Args:
        pickable_obj: The pickable object (particle).
        run: The Copick run.
        tomogram: The tomogram data.
        grid_spacing_factor: Factor to multiply the particle radius by to determine grid spacing.
        session_id: The session ID for the segmentation.
        user_id: The user ID for segmentation creation.
    """
    # Ensure it's a pickable particle object
    if not pickable_obj.is_particle:
        print(f"Object {pickable_obj.name} is not a particle.")
        return

    obj_name = pickable_obj.name
    radius = pickable_obj.radius
    if not radius:
        print(f"Object {obj_name} does not have a valid radius.")
        return

    grid_spacing = radius * grid_spacing_factor

    # Open the highest resolution of the tomogram
    image = zarr.open(tomogram.zarr(), mode="r")["0"]

    # Create a grid of points
    points = []
    for z in np.arange(0, image.shape[0], grid_spacing):
        for y in np.arange(0, image.shape[1], grid_spacing):
            for x in np.arange(0, image.shape[2], grid_spacing):
                points.append(CopickPoint(location={"x": x, "y": y, "z": z}))

    # Save the picks
    pick_set = run.new_picks(obj_name, session_id, user_id)
    pick_set.points = points
    pick_set.store()

    print(f"Saved {len(points)} grid points for object {obj_name}.")
    return pick_set


if __name__ == "__main__":
    import copick

    copick_config_path = "path/to/copick_config.json"
    grid_spacing_factor = 1.5
    tomo_type = "your_tomo_type"
    voxel_spacing = 1.0
    session_id = "example_session"
    user_id = "example_user"
    run_name = "example_run"

    # Load the Copick root and the run
    root = copick.from_file(copick_config_path)
    run = root.get_run(run_name)

    # Get the tomogram and pickable object
    voxel_spacing_obj = run.get_voxel_spacing(voxel_spacing)
    tomogram = voxel_spacing_obj.get_tomogram(tomo_type)

    for pickable_obj in root.pickable_objects:
        grid_picker(pickable_obj, run, tomogram, grid_spacing_factor, session_id, user_id)
