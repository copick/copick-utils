"""Connected components processing for segmentation volumes."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage
from skimage import measure

if TYPE_CHECKING:
    from copick.models import CopickRoot, CopickRun, CopickSegmentation


def separate_connected_components_3d(
    volume: np.ndarray,
    voxel_spacing: float,
    connectivity: Union[int, str] = "all",
    min_size: Optional[float] = None,
) -> Tuple[np.ndarray, int, Dict[int, Dict[str, Any]]]:
    """
    Separate connected components in a 3D binary or labeled volume.

    Args:
        volume: 3D binary or labeled segmentation volume
        voxel_spacing: Voxel spacing in angstroms
        connectivity: Connectivity for connected components (default: "all")
            String format: "face" (6-connected), "face-edge" (18-connected), "all" (26-connected)
            Legacy int format: 6, 18, or 26 (for backward compatibility)
        min_size: Minimum component volume in cubic angstroms (Å³) to keep (None = keep all)

    Returns:
        Tuple of (labeled_volume, num_components, component_info):
            - labeled_volume: Volume with each connected component labeled with unique integer
            - num_components: Number of connected components found
            - component_info: Dictionary with information about each component
    """
    # Convert to binary if not already
    binary_volume = volume > 0 if volume.dtype != bool else volume.copy()

    # Map connectivity to integer (support both string and legacy int format)
    if isinstance(connectivity, str):
        connectivity_map = {
            "face": 6,
            "face-edge": 18,
            "all": 26,
        }
        connectivity_int = connectivity_map.get(connectivity, 26)
    else:
        connectivity_int = connectivity

    # Define connectivity structure
    if connectivity_int == 6:
        structure = ndimage.generate_binary_structure(3, 1)  # faces only
    elif connectivity_int == 18:
        structure = ndimage.generate_binary_structure(3, 2)  # faces + edges
    elif connectivity_int == 26:
        structure = ndimage.generate_binary_structure(3, 3)  # all neighbors
    else:
        raise ValueError("Connectivity must be 6, 18, or 26 (or 'face', 'face-edge', 'all')")

    # Label connected components
    labeled_volume, num_components = ndimage.label(binary_volume, structure=structure)

    print(f"Found {num_components} connected components")

    # Get component properties
    component_info = {}
    props = measure.regionprops(labeled_volume)

    print(f"Found {len(props)} connected components")

    # Calculate voxel volume in cubic angstroms
    voxel_volume = voxel_spacing**3

    # Filter by size if specified
    if min_size is not None and min_size > 0:
        for prop in props:
            component_volume = prop.area * voxel_volume
            if component_volume < min_size:
                labeled_volume[labeled_volume == prop.label] = 0

        # Relabel after filtering
        labeled_volume, num_components = ndimage.label(labeled_volume > 0, structure=structure)
        props = measure.regionprops(labeled_volume)
        print(f"After filtering by size (min={min_size} Å³): {num_components} components")

    # Store component information
    for _i, prop in enumerate(props, 1):
        component_info[prop.label] = {
            "volume": prop.area,  # number of voxels
            "centroid": prop.centroid,
            "bbox": prop.bbox,  # (min_z, min_y, min_x, max_z, max_y, max_x)
            "extent": prop.extent,  # ratio of component area to bounding box area
        }

    return labeled_volume, num_components, component_info


def extract_individual_components(labeled_volume: np.ndarray) -> List[np.ndarray]:
    """
    Extract each connected component as a separate binary volume.

    Args:
        labeled_volume: Volume with labeled connected components

    Returns:
        List of binary volumes, each containing one component
    """
    unique_labels = np.unique(labeled_volume)
    unique_labels = unique_labels[unique_labels > 0]  # exclude background (0)

    components = []
    for label in unique_labels:
        component = (labeled_volume == label).astype(np.uint8)
        components.append(component)

    return components


def print_component_stats(component_info: Dict[int, Dict[str, Any]]) -> None:
    """Print statistics about connected components."""
    print("\nComponent Statistics:")
    print("-" * 60)
    print(f"{'Label':<8} {'Volume':<10} {'Centroid (z,y,x)':<25} {'Extent':<10}")
    print("-" * 60)

    for label, info in component_info.items():
        centroid_str = f"({info['centroid'][0]:.1f},{info['centroid'][1]:.1f},{info['centroid'][2]:.1f})"
        print(f"{label:<8} {info['volume']:<10} {centroid_str:<25} {info['extent']:<10.3f}")


def separate_segmentation_components(
    segmentation: "CopickSegmentation",
    connectivity: Union[int, str] = "all",
    min_size: Optional[float] = None,
    session_id_template: str = "inst-{instance_id}",
    output_user_id: str = "components",
    multilabel: bool = True,
    session_id_prefix: str = None,  # Deprecated, kept for backward compatibility
) -> List["CopickSegmentation"]:
    """
    Separate connected components in a segmentation into individual segmentations.

    Args:
        segmentation: Input segmentation to process
        connectivity: Connectivity for connected components (default: "all")
            String format: "face" (6-connected), "face-edge" (18-connected), "all" (26-connected)
            Legacy int format: 6, 18, or 26 (for backward compatibility)
        min_size: Minimum component volume in cubic angstroms (Å³) to keep (None = keep all)
        session_id_template: Template for output session IDs with {instance_id} placeholder
        output_user_id: User ID for output segmentations
        multilabel: Whether to treat input as multilabel segmentation
        session_id_prefix: Deprecated. Use session_id_template instead.

    Returns:
        List of created segmentations, one per component
    """
    # Handle deprecated session_id_prefix parameter
    if session_id_prefix is not None:
        session_id_template = f"{session_id_prefix}{{instance_id}}"
    # Get the segmentation volume
    volume = segmentation.numpy()
    if volume is None:
        raise ValueError("Could not load segmentation data")

    run = segmentation.run
    voxel_size = segmentation.voxel_size
    name = segmentation.name

    output_segmentations = []
    component_count = 0

    if multilabel:
        # Process each label separately
        unique_labels = np.unique(volume)
        unique_labels = unique_labels[unique_labels > 0]  # skip background

        print(f"Processing multilabel segmentation with {len(unique_labels)} labels")

        for label_value in unique_labels:
            print(f"Processing label {label_value}")

            # Extract binary volume for this label
            binary_vol = volume == label_value

            # Separate connected components
            labeled_vol, n_components, component_info = separate_connected_components_3d(
                binary_vol,
                voxel_spacing=voxel_size,
                connectivity=connectivity,
                min_size=min_size,
            )

            # Extract individual components
            individual_components = extract_individual_components(labeled_vol)

            # Create segmentations for each component
            for component_vol in individual_components:
                session_id = session_id_template.replace("{instance_id}", str(component_count))

                # Create new segmentation
                output_seg = run.new_segmentation(
                    voxel_size=voxel_size,
                    name=name,
                    session_id=session_id,
                    is_multilabel=False,
                    user_id=output_user_id,
                    exist_ok=True,
                )

                # Store the component volume
                output_seg.from_numpy(component_vol)
                output_segmentations.append(output_seg)
                component_count += 1

    else:
        # Process as binary segmentation
        print("Processing binary segmentation")

        # Separate connected components
        labeled_vol, n_components, component_info = separate_connected_components_3d(
            volume,
            voxel_spacing=voxel_size,
            connectivity=connectivity,
            min_size=min_size,
        )

        # Extract individual components
        individual_components = extract_individual_components(labeled_vol)

        # Create segmentations for each component
        for component_vol in individual_components:
            session_id = session_id_template.replace("{instance_id}", str(component_count))

            # Create new segmentation
            output_seg = run.new_segmentation(
                voxel_size=voxel_size,
                name=name,
                session_id=session_id,
                is_multilabel=False,
                user_id=output_user_id,
                exist_ok=True,
            )

            # Store the component volume
            output_seg.from_numpy(component_vol)
            output_segmentations.append(output_seg)
            component_count += 1

    print(f"Created {len(output_segmentations)} component segmentations")
    return output_segmentations


def _separate_components_worker(
    run: "CopickRun",
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str,
    connectivity: Union[int, str],
    min_size: Optional[float],
    session_id_template: str,
    output_user_id: str,
    multilabel: bool,
    root: "CopickRoot",
) -> Dict[str, Any]:
    """Worker function for batch connected components separation."""
    try:
        # Get segmentation
        segmentations = run.get_segmentations(
            name=segmentation_name,
            user_id=segmentation_user_id,
            session_id=segmentation_session_id,
        )

        if not segmentations:
            return {"processed": 0, "errors": [f"No segmentation found for {run.name}"]}

        segmentation = segmentations[0]

        # Separate components
        output_segmentations = separate_segmentation_components(
            segmentation=segmentation,
            connectivity=connectivity,
            min_size=min_size,
            session_id_template=session_id_template,
            output_user_id=output_user_id,
            multilabel=multilabel,
        )

        return {
            "processed": 1,
            "errors": [],
            "components_created": len(output_segmentations),
            "segmentations": output_segmentations,
        }

    except Exception as e:
        return {"processed": 0, "errors": [f"Error processing {run.name}: {e}"]}


def separate_components_batch(
    root: "CopickRoot",
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str,
    connectivity: Union[int, str] = "all",
    min_size: Optional[float] = None,
    session_id_template: str = "inst-{instance_id}",
    output_user_id: str = "components",
    multilabel: bool = True,
    run_names: Optional[List[str]] = None,
    workers: int = 8,
    session_id_prefix: str = None,  # Deprecated, kept for backward compatibility
) -> Dict[str, Any]:
    """
    Batch separate connected components across multiple runs.

    Args:
        root: The copick root containing runs to process
        segmentation_name: Name of the segmentation to process
        segmentation_user_id: User ID of the segmentation to process
        segmentation_session_id: Session ID of the segmentation to process
        connectivity: Connectivity for connected components (default: "all")
            String format: "face" (6-connected), "face-edge" (18-connected), "all" (26-connected)
            Legacy int format: 6, 18, or 26 (for backward compatibility)
        min_size: Minimum component volume in cubic angstroms (Å³) to keep (None = keep all)
        session_id_template: Template for output session IDs with {instance_id} placeholder. Default is "inst-{instance_id}".
        output_user_id: User ID for output segmentations. Default is "components".
        multilabel: Whether to treat input as multilabel segmentation. Default is True.
        run_names: List of run names to process. If None, processes all runs.
        workers: Number of worker processes. Default is 8.
        session_id_prefix: Deprecated. Use session_id_template instead.

    Returns:
        Dictionary with processing results and statistics
    """
    from copick.ops.run import map_runs

    # Handle deprecated session_id_prefix parameter
    if session_id_prefix is not None:
        session_id_template = f"{session_id_prefix}{{instance_id}}"

    runs_to_process = [run.name for run in root.runs] if run_names is None else run_names

    results = map_runs(
        callback=_separate_components_worker,
        root=root,
        runs=runs_to_process,
        workers=workers,
        task_desc="Separating connected components",
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        segmentation_session_id=segmentation_session_id,
        connectivity=connectivity,
        min_size=min_size,
        session_id_template=session_id_template,
        output_user_id=output_user_id,
        multilabel=multilabel,
    )

    return results
