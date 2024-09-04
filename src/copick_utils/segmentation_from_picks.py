import numpy as np
import zarr
import copick

class SegmentationFromPicks:
    """
    A class to handle the process of painting picks from a Copick project into a segmentation layer in Zarr format.
    
    Parameters:
    copick_config_path (str): Path to the Copick configuration JSON file.
    painting_segmentation_name (str, optional): Name of the segmentation layer to create or use. Defaults to 'paintingsegmentation'.
    """

    def __init__(self, copick_config_path, painting_segmentation_name=None):
        self.root = copick.from_file(copick_config_path)
        self.painting_segmentation_name = painting_segmentation_name or "paintingsegmentation"

    @staticmethod
    def create_ball(center, radius):
        """
        Create a spherical 3D array (ball) of given radius.

        Parameters:
        center (tuple): The center of the ball.
        radius (int): Radius of the ball.

        Returns:
        np.ndarray: A 3D binary array representing the ball shape.
        """
        zc, yc, xc = center
        shape = (2 * radius + 1, 2 * radius + 1, 2 * radius + 1)
        ball = np.zeros(shape, dtype=np.uint8)

        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    if np.linalg.norm(np.array([z, y, x]) - np.array([radius, radius, radius])) <= radius:
                        ball[z, y, x] = 1
        return ball

    def get_painting_segmentation(self, run, user_id, session_id, voxel_spacing, tomo_type):
        """
        Retrieve or create a segmentation layer for painting.

        Parameters:
        run (object): The run object to retrieve the segmentation from.
        user_id (str): The user ID performing the segmentation.
        session_id (str): The session ID used for the segmentation.
        voxel_spacing (float): The voxel spacing for scaling pick locations.
        tomo_type (str): Type of tomogram to use (e.g., denoised).

        Returns:
        tuple: (painting_seg_array, shape) where `painting_seg_array` is a Zarr dataset and `shape` is the shape of the segmentation.
        """
        segs = run.get_segmentations(
            user_id=user_id, session_id=session_id, is_multilabel=True, name=self.painting_segmentation_name, voxel_size=voxel_spacing
        )
        tomogram = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type)
        if not tomogram:
            return None, None
        elif len(segs) == 0:
            seg = run.new_segmentation(voxel_spacing, self.painting_segmentation_name, session_id, True, user_id=user_id)
            shape = zarr.open(tomogram.zarr(), "r")["0"].shape
            group = zarr.group(seg.path)
            group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        else:
            seg = segs[0]
            group = zarr.open_group(seg.path, mode="a")
            if 'data' not in group:
                shape = zarr.open(tomogram.zarr(), "r")["0"].shape
                group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        return group['data'], shape

    def paint_picks_as_balls(self, painting_seg_array, pick_location, segmentation_id, radius):
        """
        Paint a pick location into the segmentation array as a spherical ball.

        Parameters:
        painting_seg_array (np.ndarray): The segmentation array.
        pick_location (tuple): The (z, y, x) coordinates of the pick.
        segmentation_id (int): The ID of the segmentation label.
        radius (int): The radius of the ball to be painted.
        """
        z, y, x = pick_location
        ball = self.create_ball((radius, radius, radius), radius)

        z_min = max(0, z - radius)
        z_max = min(painting_seg_array.shape[0], z + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(painting_seg_array.shape[1], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(painting_seg_array.shape[2], x + radius + 1)

        z_ball_min = max(0, radius - z)
        z_ball_max = min(2 * radius + 1, radius + painting_seg_array.shape[0] - z)
        y_ball_min = max(0, radius - y)
        y_ball_max = min(2 * radius + 1, radius + painting_seg_array.shape[1] - y)
        x_ball_min = max(0, radius - x)
        x_ball_max = min(2 * radius + 1, radius + painting_seg_array.shape[2] - x)

        mask = ball[z_ball_min:z_ball_max, y_ball_min:y_ball_max, x_ball_min:x_ball_max] == 1
        painting_seg_array[z_min:z_max, y_min:y_max, x_min:x_max][mask] = segmentation_id

    def paint_picks(self, run, painting_seg_array, picks, segmentation_mapping, voxel_spacing, ball_radius_factor):
        """
        Paint multiple picks into the segmentation array.

        Parameters:
        run (object): The run object containing picks.
        painting_seg_array (np.ndarray): The segmentation array.
        picks (list): List of picks with object_type and location.
        segmentation_mapping (dict): Mapping from object_type to segmentation ID.
        voxel_spacing (float): Voxel spacing for pick scaling.
        ball_radius_factor (float): Factor to adjust the ball radius.
        """
        for pick in picks:
            pick_location = pick['location']
            pick_name = pick['object_type']
            segmentation_id = segmentation_mapping.get(pick_name)

            if segmentation_id is None:
                print(f"Skipping unknown object type: {pick_name}")
                continue

            z, y, x = pick_location
            z = int(z / voxel_spacing)
            y = int(y / voxel_spacing)
            x = int(x / voxel_spacing)

            particle_radius = next(obj.radius for obj in self.root.config.pickable_objects if obj.name == pick_name)
            ball_radius = int(particle_radius * ball_radius_factor / voxel_spacing)

            self.paint_picks_as_balls(painting_seg_array, (z, y, x), segmentation_id, ball_radius)

    def process_run(self, run, user_id, session_id, voxel_spacing, ball_radius_factor, allowlist_user_ids, tomo_type):
        """
        Process a single run by painting all picks into a segmentation layer.

        Parameters:
        run (object): The run to process.
        user_id (str): The user ID for segmentation.
        session_id (str): The session ID for segmentation.
        voxel_spacing (float): Voxel spacing for scaling picks.
        ball_radius_factor (float): Factor to adjust the ball radius based on pick size.
        allowlist_user_ids (list): List of allowed user IDs for segmentation.
        tomo_type (str): Type of tomogram to use for painting.
        """
        painting_seg, shape = self.get_painting_segmentation(run, user_id, session_id, voxel_spacing, tomo_type)

        if painting_seg is None:
            raise ValueError(f"Unable to obtain or create painting segmentation for run '{run.name}'.")

        segmentation_mapping = {obj.name: obj.label for obj in self.root.config.pickable_objects}
        painting_seg_array = np.zeros(shape, dtype=np.uint16)

        for obj in self.root.config.pickable_objects:
            for pick_set in run.get_picks(obj.name):
                if pick_set and pick_set.points and (not allowlist_user_ids or pick_set.user_id in allowlist_user_ids):
                    picks = [{'object_type': obj.name, 'location': (point.location.z, point.location.y, point.location.x)}
                             for point in pick_set.points]
                    self.paint_picks(run, painting_seg_array, picks, segmentation_mapping, voxel_spacing, ball_radius_factor)

        painting_seg[:] = painting_seg_array

    def process_all_runs(self, user_id, session_id, voxel_spacing, ball_radius_factor, allowlist_user_ids=None, run_name=None, tomo_type=None):
        """
        Process all runs or a specific run by painting picks into segmentation layers.

        Parameters:
        user_id (str): The user ID for segmentation.
        session_id (str): The session ID for segmentation.
        voxel_spacing (float): Voxel spacing for scaling picks.
        ball_radius_factor (float): Factor to adjust the ball radius based on pick size.
        allowlist_user_ids (list, optional): List of allowed user IDs for segmentation. Defaults to None.
        run_name (str, optional): Name of the run to process. If not provided, all runs will be processed.
        tomo_type (str, optional): Type of tomogram to use. Defaults to None.
        """
        if run_name:
            run = self.root.get_run(run_name)
            if not run:
                raise ValueError(f"Run with name '{run_name}' not found.")
            self.process_run(run, user_id, session_id, voxel_spacing, ball_radius_factor, allowlist_user_ids, tomo_type)
        else:
            for run in self.root.runs:
                self.process_run(run, user_id, session_id, voxel_spacing, ball_radius_factor, allowlist_user_ids, tomo_type)
