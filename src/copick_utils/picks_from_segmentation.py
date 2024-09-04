import os
import numpy as np
import zarr
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.morphology import binary_erosion, binary_dilation, ball


class PicksFromSegmentation:
    """
    Class for processing multilabel segmentations and extracting centroids using Copick.
    """

    def __init__(self, copick_root, run_name, voxel_spacing, segmentation_idx_offset=0):
        """
        Initialize the processor with the necessary Copick root and run parameters.

        Args:
            copick_root: A CopickRootFSSpec object for managing filesystem interactions.
            run_name (str): The name of the run to process in Copick.
            voxel_spacing (int): The voxel spacing used to scale pick locations.
            segmentation_idx_offset (int): Offset applied to the segmentation indices (default 0).
        """
        self.root = copick_root
        self.run = self.root.get_run(run_name)
        self.voxel_spacing = voxel_spacing
        self.segmentation_idx_offset = segmentation_idx_offset

    def get_painting_segmentation(self, user_id, session_id, painting_segmentation_name):
        """
        Get or create a multilabel segmentation for painting. Creates a segmentation if it doesn't exist.

        Args:
            user_id (str): The ID of the user for whom the segmentation is created.
            session_id (str): The session ID.
            painting_segmentation_name (str): Name of the painting segmentation.

        Returns:
            zarr.Dataset: The segmentation dataset.
        """
        segs = self.run.get_segmentations(user_id=user_id, session_id=session_id, is_multilabel=True, name=painting_segmentation_name, voxel_size=self.voxel_spacing)
        if not self.run.get_voxel_spacing(self.voxel_spacing).get_tomogram("denoised"):
            return None
        elif len(segs) == 0:
            seg = self.run.new_segmentation(
                self.voxel_spacing, painting_segmentation_name, session_id, True, user_id=user_id
            )
            shape = zarr.open(self.run.get_voxel_spacing(self.voxel_spacing).get_tomogram("denoised").zarr(), "r")["0"].shape
            group = zarr.group(seg.path)
            group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        else:
            seg = segs[0]
            group = zarr.open_group(seg.path, mode="a")
            if 'data' not in group:
                if not self.run.get_voxel_spacing(self.voxel_spacing).get_tomogram("denoised"):
                    return None
                shape = zarr.open(self.run.get_voxel_spacing(self.voxel_spacing).get_tomogram("denoised").zarr(), "r")["0"].shape
                group.create_dataset('data', shape=shape, dtype=np.uint16, fill_value=0)
        return group['data']

    @staticmethod
    def load_multilabel_segmentation(segmentation_dir, segmentation_name, segmentation_idx_offset=0):
        """
        Load a multilabel segmentation from a Zarr file.

        Args:
            segmentation_dir (str): Directory containing the segmentation files.
            segmentation_name (str): Name of the segmentation to load.
            segmentation_idx_offset (int): Offset applied to segmentation indices (default 0).

        Returns:
            np.ndarray: The loaded segmentation array with offset applied.
        """
        segmentation_file = [f for f in os.listdir(segmentation_dir) if f.endswith('.zarr') and segmentation_name in f]
        if not segmentation_file:
            raise FileNotFoundError(f"No segmentation file found with name: {segmentation_name}")
        seg_path = os.path.join(segmentation_dir, segmentation_file[0])
        return (zarr.open(seg_path, mode='r')['data'][:] + segmentation_idx_offset)

    @staticmethod
    def detect_local_maxima(distance, maxima_filter_size=9):
        """
        Detect local maxima in the distance transform.

        Args:
            distance (np.ndarray): Distance transform of the binary mask.
            maxima_filter_size (int): Size of the maximum detection filter (default 9).

        Returns:
            np.ndarray: A binary array indicating the location of local maxima.
        """
        footprint = np.ones((maxima_filter_size, maxima_filter_size, maxima_filter_size))
        local_max = (distance == ndi.maximum_filter(distance, footprint=footprint))
        return local_max

    def get_centroids_and_save(self, segmentation, labels_to_process, user_id, session_id, min_particle_size, max_particle_size, maxima_filter_size=9):
        """
        Extract centroids from the multilabel segmentation and save them for each label.

        Args:
            segmentation (np.ndarray): Multilabel segmentation array.
            labels_to_process (list): List of labels to process.
            user_id (str): User ID for pick saving.
            session_id (str): Session ID for pick saving.
            min_particle_size (int): Minimum size threshold for particles.
            max_particle_size (int): Maximum size threshold for particles.
            maxima_filter_size (int): Size of the maximum detection filter (default 9).

        Returns:
            dict: A dictionary mapping labels to their centroids.
        """
        all_centroids = {}

        # Structuring element for erosion and dilation
        struct_elem = ball(1)  # Adjust the size of the ball as needed

        # Create a binary mask where particles are detected based on labels_to_process
        binary_mask = np.isin(segmentation, labels_to_process).astype(int)
        eroded = binary_erosion(binary_mask, struct_elem)
        dilated = binary_dilation(eroded, struct_elem)

        # Distance transform and local maxima detection
        distance = ndi.distance_transform_edt(dilated)
        local_maxi = self.detect_local_maxima(distance, maxima_filter_size=maxima_filter_size)

        # Watershed segmentation
        markers, _ = ndi.label(local_maxi)
        watershed_labels = watershed(-distance, markers, mask=dilated)

        # Compute region properties and filter based on size
        props_list = regionprops(watershed_labels)
        for region in props_list:
            label_num = region.label
            if label_num == 0:
                continue  # Skip background
            
            region_mask = watershed_labels == label_num
            original_labels_in_region = segmentation[region_mask]

            if len(original_labels_in_region) == 0:
                continue  # Skip empty regions

            unique_labels, counts = np.unique(original_labels_in_region, return_counts=True)
            dominant_label = unique_labels[np.argmax(counts)]
            
            # Use centroid of the region to assign a pick
            centroid = region.centroid
            if min_particle_size <= region.area <= max_particle_size:
                if dominant_label in all_centroids:
                    all_centroids[dominant_label].append(centroid)
                else:
                    all_centroids[dominant_label] = [centroid]

        return all_centroids

    def save_centroids_as_picks(self, all_centroids, user_id, session_id):
        """
        Save the extracted centroids as picks using Copick.

        Args:
            all_centroids (dict): Dictionary mapping labels to their centroids.
            user_id (str): User ID for pick saving.
            session_id (str): Session ID for pick saving.
        """
        for label_num, centroids in all_centroids.items():
            object_name = [obj.name for obj in self.root.pickable_objects if obj.label == label_num]
            if not object_name:
                raise ValueError(f"Label {label_num} does not correspond to any object name in pickable objects.")
            object_name = object_name[0]
            pick_set = self.run.new_picks(object_name, session_id, user_id)
            pick_set.points = [CopickPoint(location={'x': c[2] * self.voxel_spacing, 'y': c[1] * self.voxel_spacing, 'z': c[0] * self.voxel_spacing}) for c in centroids]
            pick_set.store()


def process_segmentation(copick_root, run_name, voxel_spacing, segmentation_dir, painting_segmentation_name, session_id, user_id, labels_to_process, min_particle_size=1000, max_particle_size=50000, maxima_filter_size=9, segmentation_idx_offset=0):
    """
    High-level function to process segmentation, extract centroids, and save them as picks.

    Args:
        copick_root: A CopickRootFSSpec object for managing filesystem interactions.
        run_name (str): The name of the run to process in Copick.
        voxel_spacing (int): The voxel spacing used to scale pick locations.
        segmentation_dir (str): Directory containing the multilabel segmentation.
        painting_segmentation_name (str): Name of the painting segmentation.
        session_id (str): Session ID for pick saving.
        user_id (str): User ID for pick saving.
        labels_to_process (list): List of segmentation labels to process.
        min_particle_size (int): Minimum size threshold for particles (default 1000).
        max_particle_size (int): Maximum size threshold for particles (default 50000).
        maxima_filter_size (int): Size of the maximum detection filter (default 9).
        segmentation_idx_offset (int): Offset applied to segmentation indices (default 0).
    """
    processor = PicksFromSegmentation(copick_root, run_name, voxel_spacing, segmentation_idx_offset)
    
    segmentation = processor.load_multilabel_segmentation(segmentation_dir, painting_segmentation_name, segmentation_idx_offset)
    centroids = processor.get_centroids_and_save(segmentation, labels_to_process, user_id, session_id, min_particle_size, max_particle_size, maxima_filter_size=maxima_filter_size)
    
    processor.save_centroids_as_picks(centroids, user_id, session_id)
    print("Centroid extraction and saving complete.")