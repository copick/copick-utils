import os
import logging
import numpy as np
import zarr
from scipy.spatial import KDTree
from tqdm import tqdm
from copick.models import CopickPoint
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sample_background_points_far_from_picks(
    tomo_shape: Tuple[int, int, int],
    pick_points: np.ndarray,
    num_points: int,
    min_distance: float,
    box_half_size: int,
    max_attempts: int = 10000
) -> np.ndarray:
    """
    Sample random points that are far from existing picks and within bounds of the tomogram.
    
    Args:
        tomo_shape: Shape of the tomogram (z, y, x)
        pick_points: Array of existing pick points with shape (n, 3)
        num_points: Number of background points to sample
        min_distance: Minimum distance from any existing pick
        box_half_size: Half the box size to ensure points are within extraction bounds
        max_attempts: Maximum number of attempts to find valid background points
        
    Returns:
        numpy.ndarray: Array of background points with shape (m, 3), where m <= num_points
    """
    # Create KDTree for efficient nearest neighbor queries
    if len(pick_points) > 0:
        kdtree = KDTree(pick_points)
    else:
        # If no picks, just sample random points within bounds
        logger.warning("No existing picks found for background point reference")
        return sample_random_points_within_bounds(tomo_shape, num_points, box_half_size)
    
    shape_z, shape_y, shape_x = tomo_shape
    background_points = []
    attempts = 0
    
    # Progress bar for sampling
    pbar = tqdm(total=num_points, desc="Sampling background points")
    
    while len(background_points) < num_points and attempts < max_attempts:
        # Sample a batch of candidate points to reduce loop iterations
        batch_size = min(1000, max_attempts - attempts)
        
        # Generate random coordinates within bounds
        x_coords = np.random.randint(box_half_size, shape_x - box_half_size, batch_size)
        y_coords = np.random.randint(box_half_size, shape_y - box_half_size, batch_size)
        z_coords = np.random.randint(box_half_size, shape_z - box_half_size, batch_size)
        
        candidates = np.column_stack([z_coords, y_coords, x_coords])
        
        # Query distances to nearest picks
        distances, _ = kdtree.query(candidates)
        
        # Filter points that are far enough from existing picks
        valid_indices = np.where(distances >= min_distance)[0]
        valid_candidates = candidates[valid_indices]
        
        # Add valid points to our collection
        points_needed = num_points - len(background_points)
        points_to_add = valid_candidates[:points_needed]
        
        background_points.extend(points_to_add)
        attempts += batch_size
        
        # Update progress bar
        new_progress = min(len(background_points), num_points)
        pbar.update(new_progress - pbar.n)
    
    pbar.close()
    
    if len(background_points) < num_points:
        logger.warning(f"Could only sample {len(background_points)}/{num_points} background points " 
                      f"after {attempts} attempts. Consider reducing min_distance ({min_distance}).")
    
    return np.array(background_points)

def sample_random_points_within_bounds(tomo_shape: Tuple[int, int, int], 
                                      num_points: int, 
                                      box_half_size: int) -> np.ndarray:
    """
    Sample random points within the bounds of the tomogram.
    
    Args:
        tomo_shape: Shape of the tomogram (z, y, x)
        num_points: Number of background points to sample
        box_half_size: Half the box size to ensure points are within extraction bounds
        
    Returns:
        numpy.ndarray: Array of background points with shape (num_points, 3)
    """
    shape_z, shape_y, shape_x = tomo_shape
    
    # Generate random coordinates within bounds
    x_coords = np.random.randint(box_half_size, shape_x - box_half_size, num_points)
    y_coords = np.random.randint(box_half_size, shape_y - box_half_size, num_points)
    z_coords = np.random.randint(box_half_size, shape_z - box_half_size, num_points)
    
    return np.column_stack([z_coords, y_coords, x_coords])

def convert_to_copick_points(points: np.ndarray) -> List[CopickPoint]:
    """
    Convert numpy array of points to list of CopickPoint objects.
    
    Args:
        points: Array of points with shape (n, 3) in order (z, y, x)
        
    Returns:
        List[CopickPoint]: List of CopickPoint objects
    """
    copick_points = []
    for point in points:
        z, y, x = point
        copick_points.append(CopickPoint(location={'x': x, 'y': y, 'z': z}))
    return copick_points

def get_all_pick_points(run, voxel_spacing: float, filter_user_ids: Optional[List[str]] = None) -> np.ndarray:
    """
    Get all pick points from a run.
    
    Args:
        run: The Copick run object
        voxel_spacing: Voxel spacing for coordinate scaling
        filter_user_ids: List of user IDs to include (None means include all)
        
    Returns:
        numpy.ndarray: Array of pick points with shape (n, 3) in (z, y, x) order
    """
    all_pick_points = []
    
    for picks in run.picks:
        if not picks.from_tool:
            continue

        # Filter by user ID if specified
        if filter_user_ids is not None and picks.user_id not in filter_user_ids:
            continue
        
        # Get pick coordinates and rescale
        points, _ = picks.numpy()
        points = points / voxel_spacing
        all_pick_points.extend(points)
    
    return np.array(all_pick_points) if all_pick_points else np.empty((0, 3))

def background_picker(
    run,
    tomogram,
    bg_sample_ratio: float = 0.2,
    min_distance: float = 50.0,
    box_size: int = 48,
    max_attempts: int = 10000,
    voxel_spacing: float = 1.0,
    filter_user_ids: Optional[List[str]] = None,
    session_id: str = "0",
    user_id: str = "backgroundPicker"
) -> Optional[List[CopickPoint]]:
    """
    Creates background picks far from existing picks in a tomogram.
    
    Args:
        run: The Copick run object
        tomogram: The tomogram object
        bg_sample_ratio: Ratio of background points to sample relative to total picks
        min_distance: Minimum distance for background points from any existing pick
        box_size: Size of box to extract around each pick
        max_attempts: Maximum number of attempts to sample valid background points
        voxel_spacing: Voxel spacing to be used for coordinate scaling
        filter_user_ids: List of user IDs to include when collecting reference picks
        session_id: The session ID for the segmentation
        user_id: The user ID for segmentation creation
        
    Returns:
        Optional[List[CopickPoint]]: List of CopickPoint objects or None if failed
    """
    try:
        # Get zarr path and validate
        zarr_path = tomogram.zarr()
        if not zarr_path:
            logger.warning(f"Empty zarr path for tomogram, skipping")
            return None
        
        # Load tomogram data
        try:
            tomo_data = zarr.open(zarr_path, mode='r')['0']
            logger.info(f"Loaded tomogram with shape: {tomo_data.shape}")
        except Exception as e:
            logger.warning(f"Failed to open tomogram zarr file: {str(e)}")
            return None
        
        # If no filter_user_ids provided, use default
        if filter_user_ids is None:
            filter_user_ids = ["curation"]
        
        # Get all pick points for background sampling reference
        logger.info("Collecting pick coordinates")
        all_pick_points = get_all_pick_points(run, voxel_spacing, filter_user_ids)
        
        if len(all_pick_points) == 0:
            logger.warning("No existing picks found, using random sampling")
            # Calculate number of points based on tomogram size
            num_background_points = int(np.prod(tomo_data.shape) / (box_size**3) * 0.0001)  # Arbitrary small fraction
            logger.info(f"Sampling {num_background_points} random background points")
            background_points = sample_random_points_within_bounds(
                tomo_data.shape, 
                num_background_points, 
                box_size // 2
            )
        else:
            # Calculate number of points based on existing picks
            num_background_points = int(bg_sample_ratio * len(all_pick_points))
            logger.info(f"Sampling {num_background_points} background points far from existing {len(all_pick_points)} picks")
            
            half_box = box_size // 2
            background_points = sample_background_points_far_from_picks(
                tomo_data.shape,
                all_pick_points,
                num_background_points,
                min_distance,
                half_box,
                max_attempts
            )
        
        logger.info(f"Successfully sampled {len(background_points)} background points")
        
        # Convert to CopickPoint objects
        copick_points = convert_to_copick_points(background_points)
        
        # Save the picks
        pick_set = run.new_picks("background", session_id, user_id)
        pick_set.points = copick_points
        pick_set.store()
        
        logger.info(f"Saved {len(copick_points)} background points for run {run.name}")
        return copick_points
        
    except Exception as e:
        logger.error(f"Error in background_picker: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    import copick
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample background points far from existing picks")
    
    parser.add_argument("--copick_config", type=str, required=True,
                      help="Path to Copick config file")
    parser.add_argument("--run_name", type=str, required=True,
                      help="Name of run to process")
    parser.add_argument("--tomo_type", type=str, default="wbp",
                      help="Type of tomogram to process (default: wbp)")
    parser.add_argument("--voxel_spacing", type=float, default=10.0,
                      help="Voxel spacing to be used (default: 10.0)")
    parser.add_argument("--bg_sample_ratio", type=float, default=0.2,
                      help="Ratio of background points to sample relative to total picks (default: 0.2)")
    parser.add_argument("--min_distance", type=float, default=50.0,
                      help="Minimum distance for background points from any existing pick (default: 50.0)")
    parser.add_argument("--box_size", type=int, default=48,
                      help="Size of box to extract around each pick (default: 48)")
    parser.add_argument("--max_attempts", type=int, default=10000,
                      help="Maximum number of attempts to sample valid background points (default: 10000)")
    parser.add_argument("--session_id", type=str, default="0",
                      help="Session ID for background picks (default: 0)")
    parser.add_argument("--user_id", type=str, default="backgroundPicker",
                      help="User ID for background picks (default: backgroundPicker)")
    parser.add_argument("--filter_user_ids", type=str, nargs="+", default=["curation"],
                      help="List of user IDs to include when collecting reference picks (default: ['curation'])")
    
    args = parser.parse_args()
    
    # Load the Copick root and the run
    root = copick.from_file(args.copick_config)
    run = root.get_run(args.run_name)
    
    if run is None:
        logger.error(f"Run {args.run_name} not found")
        exit(1)
    
    # Get voxel spacing object
    spacing_obj = run.get_voxel_spacing(args.voxel_spacing)
    if spacing_obj is None:
        logger.error(f"Voxel spacing {args.voxel_spacing} not found in run {args.run_name}")
        exit(1)
    
    # Get tomograms
    tomograms = spacing_obj.get_tomograms(args.tomo_type)
    if not tomograms:
        logger.error(f"Tomogram type '{args.tomo_type}' not found for voxel spacing {args.voxel_spacing} in run {args.run_name}")
        exit(1)
    
    tomogram = tomograms[0]
    
    # Run background picker
    background_picker(
        run=run,
        tomogram=tomogram,
        bg_sample_ratio=args.bg_sample_ratio,
        min_distance=args.min_distance,
        box_size=args.box_size,
        max_attempts=args.max_attempts,
        voxel_spacing=args.voxel_spacing,
        filter_user_ids=args.filter_user_ids,
        session_id=args.session_id,
        user_id=args.user_id
    )