"""Split multilabel segmentations into individual single-class segmentations."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from copick.util.log import get_logger

if TYPE_CHECKING:
    from copick.models import CopickRoot, CopickRun, CopickSegmentation

logger = get_logger(__name__)


def split_multilabel_segmentation(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    output_user_id: str = "split",
) -> List["CopickSegmentation"]:
    """
    Split a multilabel segmentation into individual single-class binary segmentations.

    For each label value in the multilabel segmentation, this function looks up the
    corresponding PickableObject and creates a binary segmentation named after that object.

    Args:
        segmentation: Input multilabel segmentation to split
        run: CopickRun object containing the segmentation
        output_user_id: User ID for output segmentations (default: "split")

    Returns:
        List of created CopickSegmentation objects, one per label found in the input
    """
    # Load segmentation volume
    volume = segmentation.numpy()
    if volume is None:
        raise ValueError("Could not load segmentation data")

    if volume.size == 0:
        raise ValueError("Empty segmentation data")

    # Get root to access pickable objects configuration
    root = run.root
    voxel_size = segmentation.voxel_size
    input_session_id = segmentation.session_id

    # Find all unique non-zero labels
    unique_labels = np.unique(volume)
    unique_labels = unique_labels[unique_labels > 0]  # Skip background (0)

    logger.debug(f"Found {len(unique_labels)} unique labels: {unique_labels.tolist()}")

    output_segmentations = []

    # Process each label
    for label_value in unique_labels:
        # Look up the PickableObject with this label
        pickable_obj = next((obj for obj in root.config.pickable_objects if obj.label == label_value), None)

        if pickable_obj is None:
            logger.warning(f"No pickable object found for label {label_value}, using label value as name")
            object_name = str(label_value)
        else:
            object_name = pickable_obj.name
            logger.debug(f"Label {label_value} → object '{object_name}'")

        # Create binary mask for this label
        binary_mask = (volume == label_value).astype(np.uint8)
        voxel_count = int(np.sum(binary_mask))

        if voxel_count == 0:
            logger.warning(f"Label {label_value} has no voxels, skipping")
            continue

        logger.debug(f"Creating segmentation for '{object_name}' with {voxel_count} voxels")

        # Create output segmentation
        try:
            output_seg = run.new_segmentation(
                name=object_name,
                user_id=output_user_id,
                session_id=input_session_id,
                is_multilabel=False,
                voxel_size=voxel_size,
                exist_ok=True,
            )

            # Store the binary mask
            output_seg.from_numpy(binary_mask)
            output_segmentations.append(output_seg)

            logger.debug(f"Successfully created segmentation '{object_name}:{output_user_id}/{input_session_id}'")

        except Exception as e:
            logger.exception(f"Failed to create segmentation for label {label_value} ('{object_name}'): {e}")
            continue

    # Log single-line summary
    if output_segmentations:
        object_names = [seg.name for seg in output_segmentations]
        logger.info(f"Run '{run.name}': Split {len(output_segmentations)} labels → {', '.join(object_names)}")

    return output_segmentations


def _split_labels_worker(
    run: "CopickRun",
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str,
    voxel_spacing: float,
    output_user_id: str,
) -> Dict[str, Any]:
    """
    Worker function for batch splitting of multilabel segmentations.

    Args:
        run: CopickRun to process
        segmentation_name: Name of the input segmentation
        segmentation_user_id: User ID of the input segmentation
        segmentation_session_id: Session ID of the input segmentation
        voxel_spacing: Voxel spacing of the segmentation
        output_user_id: User ID for output segmentations

    Returns:
        Dictionary with processing results and statistics
    """
    try:
        # Get the input segmentation
        segmentations = run.get_segmentations(
            name=segmentation_name,
            user_id=segmentation_user_id,
            session_id=segmentation_session_id,
            voxel_size=voxel_spacing,
            is_multilabel=True,
        )

        if not segmentations:
            return {"processed": 0, "errors": [f"No multilabel segmentation found for run {run.name}"]}

        segmentation = segmentations[0]

        # Verify it's multilabel
        if not segmentation.is_multilabel:
            return {
                "processed": 0,
                "errors": [f"Segmentation in run {run.name} is not multilabel (is_multilabel=False)"],
            }

        # Split the segmentation
        output_segmentations = split_multilabel_segmentation(
            segmentation=segmentation,
            run=run,
            output_user_id=output_user_id,
        )

        # Collect object names created
        object_names = [seg.name for seg in output_segmentations]

        return {
            "processed": 1,
            "errors": [],
            "labels_split": len(output_segmentations),
            "object_names": object_names,
        }

    except Exception as e:
        logger.exception(f"Error processing run {run.name}: {e}")
        return {"processed": 0, "errors": [f"Error processing run {run.name}: {e}"]}


def split_labels_batch(
    root: "CopickRoot",
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str,
    voxel_spacing: float,
    output_user_id: str = "split",
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Batch split multilabel segmentations across multiple runs.

    Args:
        root: The copick root containing runs to process
        segmentation_name: Name of the input segmentation
        segmentation_user_id: User ID of the input segmentation
        segmentation_session_id: Session ID of the input segmentation
        voxel_spacing: Voxel spacing in angstroms
        output_user_id: User ID for output segmentations (default: "split")
        run_names: List of run names to process. If None, processes all runs.
        workers: Number of worker processes (default: 8)

    Returns:
        Dictionary with processing results and statistics per run
    """
    from copick.ops.run import map_runs

    runs_to_process = [run.name for run in root.runs] if run_names is None else run_names

    results = map_runs(
        callback=_split_labels_worker,
        root=root,
        runs=runs_to_process,
        workers=workers,
        task_desc="Splitting multilabel segmentations",
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        segmentation_session_id=segmentation_session_id,
        voxel_spacing=voxel_spacing,
        output_user_id=output_user_id,
    )

    return results
