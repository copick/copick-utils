"""Distance-based limiting operations for meshes, segmentations, and picks."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger

from copick_utils.converters.converter_common import (
    create_batch_converter,
    create_batch_worker,
    store_mesh_with_stats,
)
from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickPicks, CopickRun, CopickSegmentation

logger = get_logger(__name__)


def _create_distance_field_from_segmentation(segmentation_array: np.ndarray, voxel_spacing: float) -> np.ndarray:
    """
    Create Euclidean distance field from reference segmentation using distance transform.

    Args:
        segmentation_array: Binary reference segmentation array
        voxel_spacing: Voxel spacing of the segmentation

    Returns:
        Distance field array with exact Euclidean distances in physical units
    """
    from scipy import ndimage

    # Convert reference to binary
    binary_ref = (segmentation_array > 0).astype(bool)

    # Compute distance transform (distances to nearest foreground voxel)
    # We want distances FROM the segmentation, so use the inverse
    distance_field_voxels = ndimage.distance_transform_edt(~binary_ref)

    # Convert from voxel units to physical units
    distance_field = distance_field_voxels * voxel_spacing

    return distance_field


def _create_distance_field_from_mesh(
    mesh: tm.Trimesh,
    target_shape: tuple,
    target_voxel_spacing: float,
    mesh_voxel_spacing: float = None,
) -> np.ndarray:
    """
    Create Euclidean distance field from reference mesh using voxelization and distance transform.

    Args:
        mesh: Reference trimesh object
        target_shape: Shape of target array
        target_voxel_spacing: Voxel spacing of target
        mesh_voxel_spacing: Voxel spacing for mesh voxelization (defaults to target_voxel_spacing)

    Returns:
        Distance field array with exact Euclidean distances in physical units
    """
    if mesh_voxel_spacing is None:
        mesh_voxel_spacing = target_voxel_spacing

    # Calculate voxelization grid size based on target shape and spacing
    physical_size = np.array(target_shape) * target_voxel_spacing
    voxel_grid_shape = np.ceil(physical_size / mesh_voxel_spacing).astype(int)

    # Voxelize the mesh
    try:
        # Use trimesh's voxelization
        voxel_grid = mesh.voxelized(pitch=mesh_voxel_spacing)
        voxelized_array = voxel_grid.matrix
    except Exception as e:
        logger.warning(f"Trimesh voxelization failed: {e}. Using fallback method.")
        # Fallback: create a simple voxelization by checking mesh bounds
        bounds = mesh.bounds
        origin = bounds[0]

        # Create coordinate grids
        x_coords = np.arange(voxel_grid_shape[0]) * mesh_voxel_spacing + origin[0]
        y_coords = np.arange(voxel_grid_shape[1]) * mesh_voxel_spacing + origin[1]
        z_coords = np.arange(voxel_grid_shape[2]) * mesh_voxel_spacing + origin[2]

        xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        # Check which points are inside the mesh
        inside = mesh.contains(points)
        voxelized_array = inside.reshape(voxel_grid_shape)

    # Resample to target resolution if needed
    if mesh_voxel_spacing != target_voxel_spacing:
        from scipy.ndimage import zoom

        zoom_factor = mesh_voxel_spacing / target_voxel_spacing
        voxelized_array = zoom(voxelized_array.astype(float), zoom_factor, order=0) > 0.5

    # Ensure shape matches target
    if voxelized_array.shape != target_shape:
        # Crop or pad to match target shape
        result = np.zeros(target_shape, dtype=bool)

        # Calculate valid regions for copying
        copy_shape = np.minimum(voxelized_array.shape, target_shape)
        slices_src = tuple(slice(0, s) for s in copy_shape)
        slices_dst = tuple(slice(0, s) for s in copy_shape)

        result[slices_dst] = voxelized_array[slices_src]
        voxelized_array = result

    # Create distance field using distance transform
    return _create_distance_field_from_segmentation(voxelized_array.astype(np.uint8), target_voxel_spacing)


def limit_mesh_by_distance(
    mesh: "CopickMesh",
    run: "CopickRun",
    output_object_name: str,
    output_session_id: str,
    output_user_id: str,
    reference_mesh: Optional["CopickMesh"] = None,
    reference_segmentation: Optional["CopickSegmentation"] = None,
    max_distance: float = 100.0,
    mesh_voxel_spacing: float = None,
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """
    Limit a mesh to vertices within a certain distance of a reference surface.

    Args:
        mesh: CopickMesh to limit
        reference_mesh: Reference CopickMesh (either this or reference_segmentation must be provided)
        reference_segmentation: Reference CopickSegmentation
        run: CopickRun object
        output_object_name: Name for the output mesh
        output_session_id: Session ID for the output mesh
        output_user_id: User ID for the output mesh
        max_distance: Maximum distance from reference surface
        mesh_voxel_spacing: Voxel spacing for mesh voxelization (defaults to 10.0)
        **kwargs: Additional keyword arguments

    Returns:
        Tuple of (CopickMesh object, stats dict) or None if operation failed.
        Stats dict contains 'vertices_created' and 'faces_created'.
    """
    try:
        if reference_mesh is None and reference_segmentation is None:
            raise ValueError("Either reference_mesh or reference_segmentation must be provided")

        # Get the target mesh
        target_mesh = mesh.mesh
        if target_mesh is None:
            logger.error("Could not load target mesh data")
            return None

        # Handle Scene objects
        if isinstance(target_mesh, tm.Scene):
            if len(target_mesh.geometry) == 0:
                logger.error("Target mesh is empty")
                return None
            target_mesh = tm.util.concatenate(list(target_mesh.geometry.values()))

        # Create distance field from reference
        # Use mesh bounds to define coordinate space
        mesh_bounds = np.array([target_mesh.vertices.min(axis=0), target_mesh.vertices.max(axis=0)])

        # Add padding for max_distance
        padding = max_distance * 1.1
        mesh_bounds[0] -= padding
        mesh_bounds[1] += padding

        field_voxel_spacing = mesh_voxel_spacing if mesh_voxel_spacing is not None else 10.0
        field_size = mesh_bounds[1] - mesh_bounds[0]
        field_shape = np.ceil(field_size / field_voxel_spacing).astype(int)

        if reference_mesh is not None:
            ref_mesh = reference_mesh.mesh
            if ref_mesh is None:
                logger.error("Could not load reference mesh data")
                return None

            if isinstance(ref_mesh, tm.Scene):
                if len(ref_mesh.geometry) == 0:
                    logger.error("Reference mesh is empty")
                    return None
                ref_mesh = tm.util.concatenate(list(ref_mesh.geometry.values()))

            # Create distance field from mesh
            distance_field = _create_distance_field_from_mesh(
                ref_mesh,
                field_shape,
                field_voxel_spacing,
                mesh_voxel_spacing,
            )

        else:  # reference_segmentation is not None
            ref_seg_array = reference_segmentation.numpy()
            if ref_seg_array is None or ref_seg_array.size == 0:
                logger.error("Could not load reference segmentation data")
                return None

            # Convert segmentation to field coordinate space
            seg_indices = np.array(np.where(ref_seg_array > 0)).T
            seg_physical = seg_indices * reference_segmentation.voxel_size
            field_coords = np.floor((seg_physical - mesh_bounds[0]) / field_voxel_spacing).astype(int)

            # Create voxelized reference in field space
            voxelized_ref = np.zeros(field_shape, dtype=bool)
            valid_coords = (field_coords >= 0).all(axis=1) & (field_coords < field_shape).all(axis=1)
            if np.any(valid_coords):
                valid_field_coords = field_coords[valid_coords]
                voxelized_ref[valid_field_coords[:, 0], valid_field_coords[:, 1], valid_field_coords[:, 2]] = True

            distance_field = _create_distance_field_from_segmentation(
                voxelized_ref.astype(np.uint8),
                field_voxel_spacing,
            )

        # Convert mesh vertex coordinates to field indices
        vertex_field_coords = (target_mesh.vertices - mesh_bounds[0]) / field_voxel_spacing
        vertex_field_indices = np.floor(vertex_field_coords).astype(int)

        # Check which vertices are within field bounds
        valid_vertices = (vertex_field_indices >= 0).all(axis=1) & (vertex_field_indices < field_shape).all(axis=1)

        if not np.any(valid_vertices):
            logger.warning("No mesh vertices within distance field bounds")
            return None

        # Get distances for valid vertices
        valid_indices = vertex_field_indices[valid_vertices]
        vertex_distances = distance_field[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]

        # Find vertices within distance threshold
        distance_valid = vertex_distances <= max_distance

        # Combine bounds validity and distance validity
        final_valid = np.zeros(len(target_mesh.vertices), dtype=bool)
        final_valid[valid_vertices] = distance_valid

        if not np.any(final_valid):
            logger.warning(f"No vertices within {max_distance} units of reference surface")
            return None

        # Create a new mesh with only valid vertices and their faces
        valid_vertex_indices = np.where(final_valid)[0]

        # Create a mapping from old vertex indices to new ones
        vertex_mapping = {}
        new_vertices = []
        for new_idx, old_idx in enumerate(valid_vertex_indices):
            vertex_mapping[old_idx] = new_idx
            new_vertices.append(target_mesh.vertices[old_idx])

        new_vertices = np.array(new_vertices)

        # Filter faces to only include those with all vertices in the valid set
        valid_faces = []
        for face in target_mesh.faces:
            if all(vertex in vertex_mapping for vertex in face):
                new_face = [vertex_mapping[vertex] for vertex in face]
                valid_faces.append(new_face)

        if len(valid_faces) == 0:
            logger.warning("No valid faces after distance filtering")
            return None

        new_faces = np.array(valid_faces)

        # Create the limited mesh
        limited_mesh = tm.Trimesh(vertices=new_vertices, faces=new_faces)

        # Store the result
        copick_mesh, stats = store_mesh_with_stats(
            run=run,
            mesh=limited_mesh,
            object_name=output_object_name,
            session_id=output_session_id,
            user_id=output_user_id,
            shape_name="distance-limited mesh",
        )

        logger.info(f"Limited mesh to {stats['vertices_created']} vertices within {max_distance} units")
        return copick_mesh, stats

    except Exception as e:
        logger.error(f"Error limiting mesh by distance: {e}")
        return None


def limit_segmentation_by_distance(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    output_object_name: str,
    output_session_id: str,
    output_user_id: str,
    reference_mesh: Optional["CopickMesh"] = None,
    reference_segmentation: Optional["CopickSegmentation"] = None,
    max_distance: float = 100.0,
    voxel_spacing: float = 10.0,
    mesh_voxel_spacing: float = None,
    is_multilabel: bool = False,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Limit a segmentation to voxels within a certain distance of a reference surface.

    Args:
        segmentation: CopickSegmentation to limit
        reference_mesh: Reference CopickMesh (either this or reference_segmentation must be provided)
        reference_segmentation: Reference CopickSegmentation
        run: CopickRun object
        output_object_name: Name for the output segmentation
        output_session_id: Session ID for the output segmentation
        output_user_id: User ID for the output segmentation
        max_distance: Maximum distance from reference surface
        voxel_spacing: Voxel spacing for the output segmentation
        mesh_voxel_spacing: Voxel spacing for mesh voxelization (defaults to target voxel spacing)
        is_multilabel: Whether the segmentation is multilabel
        **kwargs: Additional keyword arguments

    Returns:
        Tuple of (CopickSegmentation object, stats dict) or None if operation failed.
        Stats dict contains 'voxels_created'.
    """
    try:
        if reference_mesh is None and reference_segmentation is None:
            raise ValueError("Either reference_mesh or reference_segmentation must be provided")

        # Load target segmentation
        seg_array = segmentation.numpy()
        if seg_array is None or seg_array.size == 0:
            logger.error("Could not load target segmentation data")
            return None

        # Create distance field from reference
        if reference_mesh is not None:
            ref_mesh = reference_mesh.mesh
            if ref_mesh is None:
                logger.error("Could not load reference mesh data")
                return None

            if isinstance(ref_mesh, tm.Scene):
                if len(ref_mesh.geometry) == 0:
                    logger.error("Reference mesh is empty")
                    return None
                ref_mesh = tm.util.concatenate(list(ref_mesh.geometry.values()))

            # Create distance field from mesh
            distance_field = _create_distance_field_from_mesh(
                ref_mesh,
                seg_array.shape,
                segmentation.voxel_size,
                mesh_voxel_spacing,
            )

        else:  # reference_segmentation is not None
            ref_seg_array = reference_segmentation.numpy()
            if ref_seg_array is None or ref_seg_array.size == 0:
                logger.error("Could not load reference segmentation data")
                return None

            # Handle different voxel spacings between reference and target
            if abs(reference_segmentation.voxel_size - segmentation.voxel_size) > 1e-6:
                # Resample reference segmentation to match target
                from scipy.ndimage import zoom

                zoom_factor = reference_segmentation.voxel_size / segmentation.voxel_size
                ref_seg_array = zoom(ref_seg_array.astype(float), zoom_factor, order=0) > 0.5

                # Crop or pad to match target shape
                if ref_seg_array.shape != seg_array.shape:
                    result = np.zeros(seg_array.shape, dtype=bool)
                    copy_shape = np.minimum(ref_seg_array.shape, seg_array.shape)
                    slices = tuple(slice(0, s) for s in copy_shape)
                    result[slices] = ref_seg_array[slices]
                    ref_seg_array = result

            # Create distance field from segmentation
            distance_field = _create_distance_field_from_segmentation(ref_seg_array, segmentation.voxel_size)

        # Apply distance threshold to create mask
        distance_mask = distance_field <= max_distance

        # Apply mask to target segmentation
        output_array = seg_array * distance_mask

        if np.sum(output_array > 0) == 0:
            logger.warning(f"No voxels within {max_distance} units of reference surface")
            return None

        # Create output segmentation
        output_seg = run.new_segmentation(
            name=output_object_name,
            user_id=output_user_id,
            session_id=output_session_id,
            is_multilabel=is_multilabel,
            voxel_size=voxel_spacing,
            exist_ok=True,
        )

        # Store the result
        output_seg.from_numpy(output_array)

        stats = {"voxels_created": int(np.sum(output_array > 0))}
        logger.info(f"Limited segmentation to {stats['voxels_created']} voxels within {max_distance} units")
        return output_seg, stats

    except Exception as e:
        logger.error(f"Error limiting segmentation by distance: {e}")
        return None


def limit_picks_by_distance(
    picks: "CopickPicks",
    run: "CopickRun",
    pick_object_name: str,
    pick_session_id: str,
    pick_user_id: str,
    reference_mesh: Optional["CopickMesh"] = None,
    reference_segmentation: Optional["CopickSegmentation"] = None,
    max_distance: float = 100.0,
    mesh_voxel_spacing: float = None,
    **kwargs,
) -> Optional[Tuple["CopickPicks", Dict[str, int]]]:
    """
    Limit picks to those within a certain distance of a reference surface.

    Args:
        picks: CopickPicks to limit
        reference_mesh: Reference CopickMesh (either this or reference_segmentation must be provided)
        reference_segmentation: Reference CopickSegmentation
        run: CopickRun object
        pick_object_name: Name for the output picks
        pick_session_id: Session ID for the output picks
        pick_user_id: User ID for the output picks
        max_distance: Maximum distance from reference surface
        mesh_voxel_spacing: Voxel spacing for mesh voxelization (defaults to 10.0)
        **kwargs: Additional keyword arguments

    Returns:
        Tuple of (CopickPicks object, stats dict) or None if operation failed.
        Stats dict contains 'points_created'.
    """
    try:
        if reference_mesh is None and reference_segmentation is None:
            raise ValueError("Either reference_mesh or reference_segmentation must be provided")

        # Load pick data
        points, transforms = picks.numpy()
        if points is None or len(points) == 0:
            logger.error("Could not load pick data")
            return None

        pick_positions = points[:, :3]  # Use only x, y, z coordinates

        # We need a coordinate space to create the distance field
        # Use the reference segmentation's coordinate space, or create one for mesh references
        if reference_segmentation is not None:
            ref_seg_array = reference_segmentation.numpy()
            if ref_seg_array is None or ref_seg_array.size == 0:
                logger.error("Could not load reference segmentation data")
                return None

            # Use reference segmentation's coordinate space
            field_voxel_spacing = reference_segmentation.voxel_size
            distance_field = _create_distance_field_from_segmentation(ref_seg_array, field_voxel_spacing)

            # Convert pick coordinates to voxel indices in reference segmentation space
            pick_voxel_coords = pick_positions / field_voxel_spacing
            pick_voxel_indices = np.floor(pick_voxel_coords).astype(int)

        else:  # reference_mesh is not None
            ref_mesh = reference_mesh.mesh
            if ref_mesh is None:
                logger.error("Could not load reference mesh data")
                return None

            if isinstance(ref_mesh, tm.Scene):
                if len(ref_mesh.geometry) == 0:
                    logger.error("Reference mesh is empty")
                    return None
                ref_mesh = tm.util.concatenate(list(ref_mesh.geometry.values()))

            # Define coordinate space based on mesh bounds and pick positions
            all_coords = np.vstack([ref_mesh.vertices, pick_positions])
            coord_bounds = np.array([all_coords.min(axis=0), all_coords.max(axis=0)])

            # Add padding for max_distance
            padding = max_distance * 1.1
            coord_bounds[0] -= padding
            coord_bounds[1] += padding

            field_voxel_spacing = mesh_voxel_spacing if mesh_voxel_spacing is not None else 10.0
            field_size = coord_bounds[1] - coord_bounds[0]
            field_shape = np.ceil(field_size / field_voxel_spacing).astype(int)

            # Create distance field from mesh in this coordinate space
            distance_field = _create_distance_field_from_mesh(
                ref_mesh,
                field_shape,
                field_voxel_spacing,
                mesh_voxel_spacing,
            )

            # Convert pick coordinates to voxel indices in this field space
            pick_voxel_coords = (pick_positions - coord_bounds[0]) / field_voxel_spacing
            pick_voxel_indices = np.floor(pick_voxel_coords).astype(int)

        # Check which picks are within field bounds
        valid_picks = (pick_voxel_indices >= 0).all(axis=1) & (pick_voxel_indices < distance_field.shape).all(axis=1)

        if not np.any(valid_picks):
            logger.warning("No picks within distance field bounds")
            return None

        # Get distances for valid picks
        valid_indices = pick_voxel_indices[valid_picks]
        pick_distances = distance_field[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]

        # Find picks within distance threshold
        distance_valid = pick_distances <= max_distance

        # Combine bounds validity and distance validity
        final_valid = np.zeros(len(points), dtype=bool)
        final_valid[valid_picks] = distance_valid

        if not np.any(final_valid):
            logger.warning(f"No picks within {max_distance} units of reference surface")
            return None

        # Filter picks
        valid_points = points[final_valid]
        valid_transforms = transforms[final_valid] if transforms is not None else None

        # Create output picks
        output_picks = run.new_picks(pick_object_name, pick_session_id, pick_user_id, exist_ok=True)
        output_picks.from_numpy(positions=valid_points, transforms=valid_transforms)
        output_picks.store()

        stats = {"points_created": len(valid_points)}
        logger.info(f"Limited picks to {stats['points_created']} points within {max_distance} units")
        return output_picks, stats

    except Exception as e:
        logger.error(f"Error limiting picks by distance: {e}")
        return None


# Create batch workers
_limit_mesh_by_distance_worker = create_batch_worker(limit_mesh_by_distance, "mesh", "mesh", min_points=0)
_limit_segmentation_by_distance_worker = create_batch_worker(
    limit_segmentation_by_distance,
    "segmentation",
    "segmentation",
    min_points=0,
)
_limit_picks_by_distance_worker = create_batch_worker(limit_picks_by_distance, "picks", "picks", min_points=1)

# Create batch converters
limit_mesh_by_distance_batch = create_batch_converter(
    limit_mesh_by_distance,
    "Limiting meshes by distance",
    "mesh",
    "mesh",
    min_points=0,
)

limit_segmentation_by_distance_batch = create_batch_converter(
    limit_segmentation_by_distance,
    "Limiting segmentations by distance",
    "segmentation",
    "segmentation",
    min_points=0,
)

limit_picks_by_distance_batch = create_batch_converter(
    limit_picks_by_distance,
    "Limiting picks by distance",
    "picks",
    "picks",
    min_points=1,
)

# Lazy batch converters for new architecture
limit_segmentation_by_distance_lazy_batch = create_lazy_batch_converter(
    converter_func=limit_segmentation_by_distance,
    task_description="Limiting segmentations by distance",
)

limit_picks_by_distance_lazy_batch = create_lazy_batch_converter(
    converter_func=limit_picks_by_distance,
    task_description="Limiting picks by distance",
)

limit_mesh_by_distance_lazy_batch = create_lazy_batch_converter(
    converter_func=limit_mesh_by_distance,
    task_description="Limiting meshes by distance",
)
