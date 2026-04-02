"""Connected components processing for segmentation volumes."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from copick.util.log import get_logger
from scipy import ndimage
from skimage import measure

from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickSegmentation


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


def separate_components_converter(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    connectivity: Union[int, str] = "all",
    min_size: Optional[float] = None,
    multilabel: bool = True,
    **kwargs,
) -> Optional[Tuple[None, Dict[str, int]]]:
    """
    Lazy converter wrapper for separate_segmentation_components.

    The session_id from the output URI is used as a template with {instance_id}.

    Args:
        segmentation: Input CopickSegmentation object.
        run: CopickRun object.
        object_name: Unused (components keep the input segmentation name).
        session_id: Session ID template (should contain {instance_id}).
        user_id: User ID for output segmentations.
        connectivity: Connectivity for connected components.
        min_size: Minimum component volume in cubic angstroms (Å³) to keep.
        multilabel: Whether to treat input as multilabel segmentation.
        **kwargs: Additional keyword arguments from lazy converter.

    Returns:
        Tuple of (None, stats dict) or None if operation failed.
    """
    try:
        output_segmentations = separate_segmentation_components(
            segmentation=segmentation,
            connectivity=connectivity,
            min_size=min_size,
            session_id_template=session_id,
            output_user_id=user_id,
            multilabel=multilabel,
        )

        stats = {
            "components_created": len(output_segmentations),
        }

        return None, stats

    except Exception as e:
        get_logger(__name__).error(f"Error separating components in {run.name}: {e}")
        return None


# Lazy batch converter for parallel discovery and processing
separate_components_lazy_batch = create_lazy_batch_converter(
    converter_func=separate_components_converter,
    task_description="Separating connected components",
)
