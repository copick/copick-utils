"""Connected components processing for segmentation volumes."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from copick.util.log import get_logger
from scipy import ndimage
from skimage import measure

from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickSegmentation

logger = get_logger(__name__)


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
    logger.debug(f"Found {num_components} connected components")

    # Filter by size if specified: one lookup table instead of one pass over the volume per component
    if min_size is not None and min_size > 0 and num_components > 0:
        sizes = np.bincount(labeled_volume.ravel(), minlength=num_components + 1)
        keep = sizes * voxel_spacing**3 >= min_size
        keep[0] = False
        relabel = np.zeros(num_components + 1, dtype=labeled_volume.dtype)
        relabel[keep] = np.arange(1, int(keep.sum()) + 1, dtype=labeled_volume.dtype)
        labeled_volume = relabel[labeled_volume]
        num_components = int(keep.sum())
        logger.debug(f"After filtering by size (min={min_size} Å³): {num_components} components")

    # Get component properties
    component_info = {}
    props = measure.regionprops(labeled_volume)

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


def components_as_instances(
    volume: np.ndarray,
    voxel_spacing: float,
    connectivity: Union[int, str] = "all",
    min_size: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """
    Label the connected components of a binary volume as instances, numbered 1..K by size (largest first).

    Args:
        volume: 3D binary volume (any non-zero voxel is foreground).
        voxel_spacing: Voxel spacing in angstroms.
        connectivity: "face", "face-edge" or "all" (or 6, 18, 26).
        min_size: Minimum component volume in cubic angstroms (Å³) to keep (None = keep all).

    Returns:
        Tuple of (instance volume, K). The volume holds 0 for background and 1..K for the components.
    """
    labeled, n, _ = separate_connected_components_3d(volume, voxel_spacing, connectivity, min_size)
    if n == 0:
        return np.zeros(volume.shape, dtype=np.uint16), 0
    sizes = np.bincount(labeled.ravel(), minlength=n + 1)[1:]
    order = np.argsort(-sizes, kind="stable")  # largest first; ties keep their scan order
    dtype = np.uint16 if n < 2**16 else np.uint32
    rank = np.zeros(n + 1, dtype=dtype)
    rank[order + 1] = np.arange(1, n + 1, dtype=dtype)
    return rank[labeled], n


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

    volume = segmentation.numpy()
    if volume is None:
        raise ValueError("Could not load segmentation data")

    run = segmentation.run
    voxel_size = segmentation.voxel_size
    name = segmentation.name

    if multilabel:
        unique_labels = np.unique(volume)
        masks = [volume == label_value for label_value in unique_labels[unique_labels > 0]]
        logger.info(f"Processing multilabel segmentation with {len(masks)} labels")
    else:
        masks = [volume > 0]

    output_segmentations = []
    component_count = 0
    for mask in masks:
        labeled_vol, n_components, _ = separate_connected_components_3d(
            mask,
            voxel_spacing=voxel_size,
            connectivity=connectivity,
            min_size=min_size,
        )
        # One component volume at a time, so K components never occupy K full volumes at once
        for component in range(1, n_components + 1):
            output_seg = run.new_segmentation(
                voxel_size=voxel_size,
                name=name,
                session_id=session_id_template.replace("{instance_id}", str(component_count)),
                is_multilabel=False,
                user_id=output_user_id,
                exist_ok=True,
            )
            output_seg.from_numpy((labeled_vol == component).astype(np.uint8))
            output_segmentations.append(output_seg)
            component_count += 1

    logger.info(f"Created {len(output_segmentations)} component segmentations")
    return output_segmentations


def separate_components_as_instances(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    output_type: str,
    connectivity: Union[int, str] = "all",
    min_size: Optional[float] = None,
) -> Tuple["CopickSegmentation", Dict[str, int]]:
    """
    Write the connected components of a segmentation as one instance or panoptic segmentation.

    An instance segmentation holds one object, so its input is a binary segmentation (or a multilabel one with a
    single label); each component becomes an instance, numbered 1..K by size. A panoptic output keeps every label of
    a multilabel input (or the object's label for a binary input) and numbers the components of each label 1..K by
    size.

    Args:
        segmentation: Input binary or multilabel segmentation.
        run: CopickRun object.
        object_name: Output name (the object, for an instance segmentation).
        session_id: Output session ID.
        user_id: Output user ID.
        output_type: "instance" or "panoptic".
        connectivity: Connectivity for connected components.
        min_size: Minimum component volume in cubic angstroms (Å³) to keep.

    Returns:
        Tuple of (output segmentation, stats dict).
    """
    from copick_utils.util.segmentations import new_segmentation, segmentation_type

    volume = segmentation.numpy()
    if volume is None:
        raise ValueError("Could not load segmentation data")
    input_type = segmentation_type(segmentation)
    if input_type not in ("binary", "multilabel"):
        raise ValueError(f"Components are separated from binary or multilabel segmentations, not {input_type} ones.")
    voxel_size = segmentation.voxel_size
    labels = np.unique(volume)
    labels = labels[labels > 0]

    if output_type == "instance":
        if len(labels) > 1:
            raise ValueError(
                f"An instance segmentation holds one object, but {segmentation.name} has {len(labels)} labels; "
                "write a panoptic segmentation (?panoptic=true) instead.",
            )
        instances, n = components_as_instances(volume, voxel_size, connectivity, min_size)
        output = new_segmentation(run, voxel_size, object_name, session_id, user_id, "instance")
        output.from_numpy(instances)
        return output, {"components_created": n}

    if output_type != "panoptic":
        raise ValueError(f"Unknown output type {output_type!r}.")
    if input_type == "binary":
        obj = run.root.get_object(segmentation.name)
        if obj is None or obj.label is None:
            raise ValueError(f"{segmentation.name} is not a pickable object with a label.")
        label_of = {int(v): int(obj.label) for v in labels}
    else:
        label_of = {int(v): int(v) for v in labels}
    label_channel = np.zeros(volume.shape, dtype=np.uint32)
    instance_channel = np.zeros(volume.shape, dtype=np.uint32)
    total = 0
    for value in labels:
        instances, n = components_as_instances(volume == value, voxel_size, connectivity, min_size)
        kept = instances > 0
        label_channel[kept] = label_of[int(value)]
        instance_channel[kept] = instances[kept]
        total += n
    output = new_segmentation(run, voxel_size, object_name, session_id, user_id, "panoptic")
    output.from_numpy(np.stack([label_channel, instance_channel]))
    return output, {"components_created": total}


def separate_components_converter(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    connectivity: Union[int, str] = "all",
    min_size: Optional[float] = None,
    multilabel: bool = True,
    output_segmentation_type: Optional[str] = None,
    **kwargs,
) -> Optional[Tuple[Optional["CopickSegmentation"], Dict[str, int]]]:
    """
    Lazy converter wrapper for separate_segmentation_components.

    By default each component becomes its own segmentation, and the session_id from the output URI is used as a
    template with {instance_id}. When the output URI names an instance (``?instance=true``) or panoptic
    (``?panoptic=true``) segmentation, the components are written as the instances of that one segmentation instead.

    Args:
        segmentation: Input CopickSegmentation object.
        run: CopickRun object.
        object_name: Output name (unused for per-component outputs, which keep the input segmentation name).
        session_id: Session ID (a template with {instance_id} for per-component outputs).
        user_id: User ID for output segmentations.
        connectivity: Connectivity for connected components.
        min_size: Minimum component volume in cubic angstroms (Å³) to keep.
        multilabel: Whether to treat input as multilabel segmentation.
        output_segmentation_type: The segmentation type the output URI names, if any.
        **kwargs: Additional keyword arguments from lazy converter.

    Returns:
        Tuple of (output segmentation or None, stats dict) or None if operation failed.
    """
    try:
        if output_segmentation_type in ("instance", "panoptic"):
            return separate_components_as_instances(
                segmentation,
                run,
                object_name,
                session_id,
                user_id,
                output_segmentation_type,
                connectivity=connectivity,
                min_size=min_size,
            )
        output_segmentations = separate_segmentation_components(
            segmentation=segmentation,
            connectivity=connectivity,
            min_size=min_size,
            session_id_template=session_id,
            output_user_id=user_id,
            multilabel=multilabel,
        )
        return None, {"components_created": len(output_segmentations)}

    except Exception as e:
        logger.error(f"Error separating components in {run.name}: {e}")
        return None


# Lazy batch converter for parallel discovery and processing
separate_components_lazy_batch = create_lazy_batch_converter(
    converter_func=separate_components_converter,
    task_description="Separating connected components",
)
