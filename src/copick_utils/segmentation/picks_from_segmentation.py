import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.morphology import binary_erosion, binary_dilation, ball


def picks_from_segmentation(segmentation, segmentation_idx, maxima_filter_size, min_particle_size, max_particle_size, session_id, user_id, pickable_object, run, voxel_spacing=1):
    """
    Process a specific label in the segmentation, extract centroids, and save them as picks.

    Args:
        segmentation (np.ndarray): Multilabel segmentation array.
        segmentation_idx (int): The specific label from the segmentation to process.
        maxima_filter_size (int): Size of the maximum detection filter.
        min_particle_size (int): Minimum size threshold for particles.
        max_particle_size (int): Maximum size threshold for particles.
        session_id (str): Session ID for pick saving.
        user_id (str): User ID for pick saving.
        pickable_object (str): The name of the object to save picks for.
        run: A Copick run object that manages pick saving.
        voxel_spacing (int): The voxel spacing used to scale pick locations (default 1).
    """
    # Create a binary mask for the specific segmentation label
    binary_mask = (segmentation == segmentation_idx).astype(int)

    # Skip if the segmentation label is not present
    if np.sum(binary_mask) == 0:
        print(f"No segmentation with label {segmentation_idx} found.")
        return

    # Structuring element for erosion and dilation
    struct_elem = ball(1)
    eroded = binary_erosion(binary_mask, struct_elem)
    dilated = binary_dilation(eroded, struct_elem)

    # Distance transform and local maxima detection
    distance = ndi.distance_transform_edt(dilated)
    local_max = (distance == ndi.maximum_filter(distance, footprint=np.ones((maxima_filter_size, maxima_filter_size, maxima_filter_size))))

    # Watershed segmentation
    markers, _ = ndi.label(local_max)
    watershed_labels = watershed(-distance, markers, mask=dilated)

    # Extract region properties and filter based on particle size
    all_centroids = []
    for region in regionprops(watershed_labels):
        if min_particle_size <= region.area <= max_particle_size:
            all_centroids.append(region.centroid)

    # Save centroids as picks
    if all_centroids:
        pick_set = run.new_picks(pickable_object, session_id, user_id)
        pick_set.points = [{'x': c[2] * voxel_spacing, 'y': c[1] * voxel_spacing, 'z': c[0] * voxel_spacing} for c in all_centroids]
        pick_set.store()
        print(f"Centroids for label {segmentation_idx} saved successfully.")
        return pick_set
    else:
        print(f"No valid centroids found for label {segmentation_idx}.")
        return None


# Example call to the function
# picks_from_segmentation(segmentation_array, label_id, 9, 1000, 50000, session_id, user_id, pickable_object_name, run_object)
