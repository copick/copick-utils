from typing import TYPE_CHECKING, Optional, Dict, Any, List, Union

import numpy as np
import scipy.ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import ball, binary_dilation, binary_erosion
from skimage.segmentation import watershed

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickPicks, CopickRoot


def picks_from_segmentation(
    segmentation: np.ndarray,
    segmentation_idx: int,
    maxima_filter_size: int,
    min_particle_size: int,
    max_particle_size: int,
    session_id: str,
    user_id: str,
    pickable_object: str,
    run: "CopickRun",
    voxel_spacing: float = 1,
) -> Optional["CopickPicks"]:
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
    local_max = distance == ndi.maximum_filter(
        distance,
        footprint=np.ones((maxima_filter_size, maxima_filter_size, maxima_filter_size)),
    )

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

        positions = np.array(all_centroids)[:, [2, 1, 0]] * voxel_spacing
        pick_set.from_numpy(positions=positions)
        pick_set.store()

        print(f"Centroids for label {segmentation_idx} saved successfully.")
        return pick_set
    else:
        print(f"No valid centroids found for label {segmentation_idx}.")
        return None


def _picks_from_segmentation_worker(
    run: "CopickRun",
    object_name: str,
    seg_user_id: str,
    seg_session_id: str,
    segmentation_name: str,
    segmentation_idx: int,
    maxima_filter_size: int,
    min_particle_size: int,
    max_particle_size: int,
    pick_user_id: str,
    pick_session_id: str,
    voxel_spacing: float,
    root: "CopickRoot",
) -> Dict[str, Any]:
    """Worker function for batch conversion of segmentations to picks."""
    try:
        pickable_object = root.get_object(object_name)
        if not pickable_object:
            return {"processed": 0, "errors": [f"Object '{object_name}' not found in config"]}

        segs = run.get_segmentations(
            name=segmentation_name,
            user_id=seg_user_id,
            session_id=seg_session_id,
            voxel_size=voxel_spacing,
        )
        
        if not segs:
            return {"processed": 0, "errors": [f"No segmentations found for {run.name}"]}
            
        seg = segs[0]
        segmentation_array = seg.numpy()
        
        if segmentation_array is None:
            return {"processed": 0, "errors": [f"Could not load segmentation data for {run.name}"]}
        
        pick_set = picks_from_segmentation(
            segmentation=segmentation_array,
            segmentation_idx=segmentation_idx,
            maxima_filter_size=maxima_filter_size,
            min_particle_size=min_particle_size,
            max_particle_size=max_particle_size,
            session_id=pick_session_id,
            user_id=pick_user_id,
            pickable_object=object_name,
            run=run,
            voxel_spacing=voxel_spacing,
        )
        
        if pick_set and pick_set.points:
            return {"processed": 1, "errors": [], "result": pick_set, "points_created": len(pick_set.points)}
        else:
            return {"processed": 0, "errors": [f"No picks generated for {run.name}"]}
        
    except Exception as e:
        return {"processed": 0, "errors": [f"Error processing {run.name}: {e}"]}


def picks_from_segmentation_batch(
    root: "CopickRoot",
    object_name: str,
    seg_user_id: str,
    seg_session_id: str,
    segmentation_name: str,
    segmentation_idx: int,
    maxima_filter_size: int,
    min_particle_size: int,
    max_particle_size: int,
    pick_user_id: str,
    pick_session_id: str,
    voxel_spacing: float,
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Batch convert segmentations to picks across multiple runs.

    Parameters:
    -----------
    root : copick.Root
        The copick root containing runs to process.
    object_name : str
        Name of the object to process segmentations for.
    seg_user_id : str
        User ID of the segmentations to convert.
    seg_session_id : str
        Session ID of the segmentations to convert.
    segmentation_name : str
        Name of the segmentation to process.
    segmentation_idx : int
        The specific label from the segmentation to process.
    maxima_filter_size : int
        Size of the maximum detection filter.
    min_particle_size : int
        Minimum size threshold for particles.
    max_particle_size : int
        Maximum size threshold for particles.
    pick_user_id : str
        User ID for the created picks.
    pick_session_id : str
        Session ID for the created picks.
    voxel_spacing : float
        Voxel spacing for scaling pick locations.
    run_names : list, optional
        List of run names to process. If None, processes all runs.
    workers : int, optional
        Number of worker processes. Default is 8.

    Returns:
    --------
    dict
        Dictionary with processing results and statistics.
    """
    from copick.ops.run import map_runs
    
    runs_to_process = [run.name for run in root.runs] if run_names is None else run_names
    
    results = map_runs(
        callback=_picks_from_segmentation_worker,
        root=root,
        runs=runs_to_process,
        workers=workers,
        task_desc="Converting segmentations to picks",
        object_name=object_name,
        seg_user_id=seg_user_id,
        seg_session_id=seg_session_id,
        segmentation_name=segmentation_name,
        segmentation_idx=segmentation_idx,
        maxima_filter_size=maxima_filter_size,
        min_particle_size=min_particle_size,
        max_particle_size=max_particle_size,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        voxel_spacing=voxel_spacing,
    )
    
    return results


# Example call to the function
# picks_from_segmentation(segmentation_array, label_id, 9, 1000, 50000, session_id, user_id, pickable_object_name, run_object)
