"""3D skeletonization processing for segmentation volumes."""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
from copick.util.log import get_logger
from scipy import ndimage
from skimage import morphology
from skimage.morphology import remove_small_objects, skeletonize

from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickSegmentation

logger = get_logger(__name__)


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


def skeletonize_converter(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    method: str = "skimage",
    remove_noise: bool = True,
    min_object_size: int = 50,
    remove_short_branches: bool = True,
    min_branch_length: int = 5,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Lazy converter wrapper for skeletonize_segmentation.

    Matches the lazy converter signature:
        (segmentation, run, object_name, session_id, user_id, **tool_kwargs)

    Args:
        segmentation: Input CopickSegmentation object.
        run: CopickRun object.
        object_name: Name for the output segmentation.
        session_id: Session ID for the output segmentation.
        user_id: User ID for the output segmentation.
        method: Skeletonization method ('skimage', 'distance_transform').
        remove_noise: Whether to remove small objects before skeletonization.
        min_object_size: Minimum size of objects to keep during preprocessing.
        remove_short_branches: Whether to remove short branches from skeleton.
        min_branch_length: Minimum length of branches to keep.
        **kwargs: Additional keyword arguments from lazy converter (ignored).

    Returns:
        Tuple of (CopickSegmentation object, stats dict) or None if operation failed.
    """
    output_seg = skeletonize_segmentation(
        segmentation=segmentation,
        method=method,
        remove_noise=remove_noise,
        min_object_size=min_object_size,
        remove_short_branches=remove_short_branches,
        min_branch_length=min_branch_length,
        output_session_id=session_id,
        output_user_id=user_id,
    )

    if output_seg is None:
        return None

    return output_seg, {"skeletons_created": 1}


# Lazy batch converter for parallel discovery and processing
skeletonize_lazy_batch = create_lazy_batch_converter(
    converter_func=skeletonize_converter,
    task_description="Skeletonizing segmentations",
)
