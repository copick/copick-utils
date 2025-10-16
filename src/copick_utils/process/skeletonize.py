"""3D skeletonization processing for segmentation volumes."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from copick.util.uri import get_copick_objects_by_type
from scipy import ndimage
from skimage import morphology
from skimage.morphology import remove_small_objects, skeletonize

if TYPE_CHECKING:
    from copick.models import CopickRoot, CopickRun, CopickSegmentation


class TubeSkeletonizer3D:
    """3D tube skeletonization class based on scikit-image."""

    def __init__(self):
        self.original_volume = None
        self.skeleton = None
        self.skeleton_coords = None

    def load_volume(self, volume_array: np.ndarray):
        """
        Load 3D volume from array.

        Args:
            volume_array: 3D binary array where tube is 1, background is 0
        """
        self.original_volume = volume_array.astype(bool)

    def preprocess_volume(self, remove_noise: bool = True, min_object_size: int = 100):
        """
        Preprocess the volume before skeletonization.

        Args:
            remove_noise: Whether to remove small objects (noise)
            min_object_size: Minimum size of objects to keep
        """
        if remove_noise and np.any(self.original_volume):
            self.original_volume = remove_small_objects(self.original_volume, min_size=min_object_size)

    def skeletonize(self, method: str = "skimage"):
        """
        Perform 3D skeletonization.

        Args:
            method: Method to use ('skimage', 'distance_transform')
        """
        if not np.any(self.original_volume):
            print("Warning: Volume is empty, creating empty skeleton")
            self.skeleton = np.zeros_like(self.original_volume, dtype=bool)
            self.skeleton_coords = np.array([]).reshape(0, 3)
            return

        if method == "skimage":
            # Use scikit-image's 3D skeletonization
            self.skeleton = skeletonize(self.original_volume)

        elif method == "distance_transform":
            # Alternative method using distance transform
            # Compute distance transform
            distance = ndimage.distance_transform_edt(self.original_volume)

            # Find local maxima of distance transform
            local_maxima = morphology.local_maxima(distance)

            # Clean up the skeleton
            self.skeleton = local_maxima & self.original_volume

        else:
            raise ValueError(f"Unknown skeletonization method: {method}")

        # Get skeleton coordinates
        self.skeleton_coords = np.array(np.where(self.skeleton)).T

    def post_process_skeleton(self, remove_short_branches: bool = True, min_branch_length: int = 5):
        """
        Post-process the skeleton to remove artifacts.

        Args:
            remove_short_branches: Whether to remove short branches
            min_branch_length: Minimum length of branches to keep
        """
        if remove_short_branches and len(self.skeleton_coords) > 0:
            # Remove small objects from skeleton
            cleaned_skeleton = remove_small_objects(self.skeleton, min_size=min_branch_length)
            self.skeleton = cleaned_skeleton
            self.skeleton_coords = np.array(np.where(self.skeleton)).T

    def get_skeleton_properties(self) -> Dict[str, Any]:
        """
        Calculate properties of the skeleton.

        Returns:
            Dict of skeleton properties
        """
        if self.skeleton_coords is None or len(self.skeleton_coords) == 0:
            return {"n_voxels": 0, "bounding_box": {"min": None, "max": None}}

        properties = {
            "n_voxels": len(self.skeleton_coords),
            "bounding_box": {
                "min": np.min(self.skeleton_coords, axis=0).tolist(),
                "max": np.max(self.skeleton_coords, axis=0).tolist(),
            },
        }

        return properties


def skeletonize_segmentation(
    segmentation: "CopickSegmentation",
    method: str = "skimage",
    remove_noise: bool = True,
    min_object_size: int = 50,
    remove_short_branches: bool = True,
    min_branch_length: int = 5,
    output_session_id: Optional[str] = None,
    output_user_id: str = "skel",
) -> Optional["CopickSegmentation"]:
    """
    Skeletonize a segmentation volume.

    Args:
        segmentation: Input segmentation to skeletonize
        method: Skeletonization method ('skimage', 'distance_transform')
        remove_noise: Whether to remove small objects before skeletonization
        min_object_size: Minimum size of objects to keep during preprocessing
        remove_short_branches: Whether to remove short branches from skeleton
        min_branch_length: Minimum length of branches to keep
        output_session_id: Session ID for output segmentation (default: same as input)
        output_user_id: User ID for output segmentation

    Returns:
        Created skeleton segmentation or None if failed
    """
    # Get the segmentation volume
    volume = segmentation.numpy()
    if volume is None:
        print(f"Error: Could not load segmentation data for {segmentation.run.name}")
        return None

    run = segmentation.run
    voxel_size = segmentation.voxel_size
    name = segmentation.name

    # Use input session_id if no output session_id specified
    if output_session_id is None:
        output_session_id = segmentation.session_id

    print(f"Skeletonizing segmentation {segmentation.session_id} in run {run.name}")

    # Initialize skeletonizer
    skeletonizer = TubeSkeletonizer3D()

    # Load volume
    skeletonizer.load_volume(volume)

    # Preprocess
    skeletonizer.preprocess_volume(remove_noise=remove_noise, min_object_size=min_object_size)

    # Skeletonize
    skeletonizer.skeletonize(method=method)

    # Post-process
    skeletonizer.post_process_skeleton(remove_short_branches=remove_short_branches, min_branch_length=min_branch_length)

    # Get properties
    properties = skeletonizer.get_skeleton_properties()
    print(f"Skeleton properties: {properties['n_voxels']} voxels")

    # Create output segmentation
    try:
        output_seg = run.new_segmentation(
            voxel_size=voxel_size,
            name=name,
            session_id=output_session_id,
            is_multilabel=False,
            user_id=output_user_id,
            exist_ok=True,
        )

        # Store the skeleton volume
        output_seg.from_numpy(skeletonizer.skeleton.astype(np.uint8))

        print(f"Created skeleton segmentation with session_id: {output_session_id}")
        return output_seg

    except Exception as e:
        print(f"Error creating skeleton segmentation: {e}")
        return None


def _skeletonize_worker(
    run: "CopickRun",
    segmentation_name: str,
    segmentation_user_id: str,
    session_id_pattern: str,
    method: str,
    remove_noise: bool,
    min_object_size: int,
    remove_short_branches: bool,
    min_branch_length: int,
    output_session_id_template: Optional[str],
    output_user_id: str,
) -> Dict[str, Any]:
    """Worker function for batch skeletonization."""
    try:
        # Find matching segmentations using copick's official URI resolution
        matching_segmentations = get_copick_objects_by_type(
            root=run.root,
            object_type="segmentation",
            run_name=run.name,
            name=segmentation_name,
            user_id=segmentation_user_id,
            session_id=session_id_pattern,
            pattern_type="glob",
        )

        if not matching_segmentations:
            return {
                "processed": 0,
                "errors": [f"No segmentations found matching pattern '{session_id_pattern}' in {run.name}"],
                "skeletons_created": 0,
            }

        skeletons_created = 0
        errors = []

        for segmentation in matching_segmentations:
            # Determine output session ID
            if output_session_id_template:
                # Replace placeholders in template
                output_session_id = output_session_id_template.replace("{input_session_id}", segmentation.session_id)
            else:
                output_session_id = segmentation.session_id

            # Skeletonize
            skeleton_seg = skeletonize_segmentation(
                segmentation=segmentation,
                method=method,
                remove_noise=remove_noise,
                min_object_size=min_object_size,
                remove_short_branches=remove_short_branches,
                min_branch_length=min_branch_length,
                output_session_id=output_session_id,
                output_user_id=output_user_id,
            )

            if skeleton_seg:
                skeletons_created += 1
            else:
                errors.append(f"Failed to skeletonize {segmentation.session_id}")

        return {
            "processed": 1,
            "errors": errors,
            "skeletons_created": skeletons_created,
            "segmentations_processed": len(matching_segmentations),
        }

    except Exception as e:
        return {"processed": 0, "errors": [f"Error processing {run.name}: {e}"], "skeletons_created": 0}


def skeletonize_batch(
    root: "CopickRoot",
    segmentation_name: str,
    segmentation_user_id: str,
    session_id_pattern: str,
    method: str = "skimage",
    remove_noise: bool = True,
    min_object_size: int = 50,
    remove_short_branches: bool = True,
    min_branch_length: int = 5,
    output_session_id_template: Optional[str] = None,
    output_user_id: str = "skel",
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Batch skeletonize segmentations across multiple runs.

    Args:
        root: The copick root containing runs to process
        segmentation_name: Name of the segmentations to process
        segmentation_user_id: User ID of the segmentations to process
        session_id_pattern: Regex pattern or exact session ID to match segmentations
        method: Skeletonization method ('skimage', 'distance_transform'). Default is 'skimage'.
        remove_noise: Whether to remove small objects before skeletonization. Default is True.
        min_object_size: Minimum size of objects to keep during preprocessing. Default is 50.
        remove_short_branches: Whether to remove short branches from skeleton. Default is True.
        min_branch_length: Minimum length of branches to keep. Default is 5.
        output_session_id_template: Template for output session IDs. Use {input_session_id} as placeholder.
            If None, uses the same session ID as input.
        output_user_id: User ID for output segmentations. Default is "skel".
        run_names: List of run names to process. If None, processes all runs.
        workers: Number of worker processes. Default is 8.

    Returns:
        Dictionary with processing results and statistics
    """
    from copick.ops.run import map_runs

    runs_to_process = [run.name for run in root.runs] if run_names is None else run_names

    results = map_runs(
        callback=_skeletonize_worker,
        root=root,
        runs=runs_to_process,
        workers=workers,
        task_desc="Skeletonizing segmentations",
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        session_id_pattern=session_id_pattern,
        method=method,
        remove_noise=remove_noise,
        min_object_size=min_object_size,
        remove_short_branches=remove_short_branches,
        min_branch_length=min_branch_length,
        output_session_id_template=output_session_id_template,
        output_user_id=output_user_id,
    )

    return results
