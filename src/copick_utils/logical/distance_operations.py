"""Distance-based limiting operations for meshes, segmentations, and picks."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger
from scipy.spatial.distance import cdist

from copick_utils.converters.converter_common import (
    create_batch_converter,
    create_batch_worker,
    store_mesh_with_stats,
)

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickPicks, CopickRun, CopickSegmentation

logger = get_logger(__name__)


def _get_mesh_surface_points(mesh: tm.Trimesh, sampling_density: float = 1.0) -> np.ndarray:
    """
    Get surface points from a mesh for distance calculations.

    Args:
        mesh: Input mesh
        sampling_density: Density of surface sampling (points per unit area)

    Returns:
        Array of surface points (N, 3)
    """
    # Calculate number of sample points based on surface area
    surface_area = mesh.area
    n_points = max(int(surface_area * sampling_density), 1000)  # Minimum 1000 points

    # Sample points uniformly on the surface
    surface_points, _ = tm.sample.sample_surface(mesh, n_points)
    return surface_points


def _get_segmentation_surface_points(
    segmentation_array: np.ndarray,
    voxel_spacing: float,
    sampling_density: float = 1.0,
) -> np.ndarray:
    """
    Get surface points from a segmentation for distance calculations.

    Args:
        segmentation_array: Binary segmentation array
        voxel_spacing: Spacing between voxels
        sampling_density: Density of surface sampling

    Returns:
        Array of surface points (N, 3) in physical coordinates
    """
    from skimage.measure import marching_cubes

    # Use marching cubes to extract surface
    try:
        vertices, faces, _, _ = marching_cubes(segmentation_array.astype(float), level=0.5)

        # Convert to physical coordinates
        vertices = vertices * voxel_spacing

        # Create a mesh and sample surface points
        surface_mesh = tm.Trimesh(vertices=vertices, faces=faces)
        return _get_mesh_surface_points(surface_mesh, sampling_density)

    except Exception as e:
        logger.warning(f"Could not extract surface from segmentation: {e}")
        # Fallback: find boundary voxels
        from scipy import ndimage

        # Find edges using morphological gradient
        structure = ndimage.generate_binary_structure(3, 1)  # 6-connected
        boundary = ndimage.binary_dilation(segmentation_array, structure) ^ segmentation_array

        # Get boundary voxel coordinates
        boundary_coords = np.array(np.where(boundary)).T

        # Convert to physical coordinates (voxel centers)
        boundary_points = boundary_coords * voxel_spacing

        return boundary_points


def limit_mesh_by_distance(
    mesh: "CopickMesh",
    run: "CopickRun",
    mesh_object_name: str,
    mesh_session_id: str,
    mesh_user_id: str,
    reference_mesh: Optional["CopickMesh"] = None,
    reference_segmentation: Optional["CopickSegmentation"] = None,
    max_distance: float = 100.0,
    sampling_density: float = 1.0,
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """
    Limit a mesh to vertices within a certain distance of a reference surface.

    Args:
        mesh: CopickMesh to limit
        reference_mesh: Reference CopickMesh (either this or reference_segmentation must be provided)
        reference_segmentation: Reference CopickSegmentation
        run: CopickRun object
        mesh_object_name: Name for the output mesh
        mesh_session_id: Session ID for the output mesh
        mesh_user_id: User ID for the output mesh
        max_distance: Maximum distance from reference surface
        sampling_density: Density of surface sampling for distance calculations
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

        # Get reference surface points
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

            reference_points = _get_mesh_surface_points(ref_mesh, sampling_density)

        else:  # reference_segmentation is not None
            ref_seg_array = reference_segmentation.numpy()
            if ref_seg_array is None or ref_seg_array.size == 0:
                logger.error("Could not load reference segmentation data")
                return None

            reference_points = _get_segmentation_surface_points(
                ref_seg_array,
                reference_segmentation.voxel_size,
                sampling_density,
            )

        if len(reference_points) == 0:
            logger.error("No reference surface points found")
            return None

        # Calculate distances from mesh vertices to reference surface
        distances = cdist(target_mesh.vertices, reference_points)
        min_distances = np.min(distances, axis=1)

        # Find vertices within distance threshold
        valid_vertex_mask = min_distances <= max_distance

        if not np.any(valid_vertex_mask):
            logger.warning(f"No vertices within {max_distance} units of reference surface")
            return None

        # Create a new mesh with only valid vertices and their faces
        valid_vertex_indices = np.where(valid_vertex_mask)[0]

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
            object_name=mesh_object_name,
            session_id=mesh_session_id,
            user_id=mesh_user_id,
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
    segmentation_object_name: str,
    segmentation_session_id: str,
    segmentation_user_id: str,
    reference_mesh: Optional["CopickMesh"] = None,
    reference_segmentation: Optional["CopickSegmentation"] = None,
    max_distance: float = 100.0,
    voxel_spacing: float = 10.0,
    sampling_density: float = 1.0,
    tomo_type: str = "wbp",
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
        segmentation_object_name: Name for the output segmentation
        segmentation_session_id: Session ID for the output segmentation
        segmentation_user_id: User ID for the output segmentation
        max_distance: Maximum distance from reference surface
        voxel_spacing: Voxel spacing for the output segmentation
        sampling_density: Density of surface sampling for distance calculations
        tomo_type: Type of tomogram to use for reference dimensions
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

        # Get reference surface points
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

            reference_points = _get_mesh_surface_points(ref_mesh, sampling_density)

        else:  # reference_segmentation is not None
            ref_seg_array = reference_segmentation.numpy()
            if ref_seg_array is None or ref_seg_array.size == 0:
                logger.error("Could not load reference segmentation data")
                return None

            reference_points = _get_segmentation_surface_points(
                ref_seg_array,
                reference_segmentation.voxel_size,
                sampling_density,
            )

        if len(reference_points) == 0:
            logger.error("No reference surface points found")
            return None

        # Get coordinates of all voxels in the segmentation
        seg_coords = np.array(np.where(seg_array > 0)).T  # Only consider non-zero voxels

        if len(seg_coords) == 0:
            logger.warning("No non-zero voxels in segmentation")
            return None

        # Convert voxel coordinates to physical coordinates
        seg_points = seg_coords * segmentation.voxel_size

        # Calculate distances from segmentation voxels to reference surface
        distances = cdist(seg_points, reference_points)
        min_distances = np.min(distances, axis=1)

        # Find voxels within distance threshold
        valid_voxel_mask = min_distances <= max_distance

        if not np.any(valid_voxel_mask):
            logger.warning(f"No voxels within {max_distance} units of reference surface")
            return None

        # Create output segmentation array
        output_array = np.zeros_like(seg_array)

        # Set valid voxels to their original values
        valid_coords = seg_coords[valid_voxel_mask]
        for coord in valid_coords:
            output_array[coord[0], coord[1], coord[2]] = seg_array[coord[0], coord[1], coord[2]]

        # Create output segmentation
        output_seg = run.new_segmentation(
            name=segmentation_object_name,
            user_id=segmentation_user_id,
            session_id=segmentation_session_id,
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
    sampling_density: float = 1.0,
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
        sampling_density: Density of surface sampling for distance calculations
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

        # Get reference surface points
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

            reference_points = _get_mesh_surface_points(ref_mesh, sampling_density)

        else:  # reference_segmentation is not None
            ref_seg_array = reference_segmentation.numpy()
            if ref_seg_array is None or ref_seg_array.size == 0:
                logger.error("Could not load reference segmentation data")
                return None

            reference_points = _get_segmentation_surface_points(
                ref_seg_array,
                reference_segmentation.voxel_size,
                sampling_density,
            )

        if len(reference_points) == 0:
            logger.error("No reference surface points found")
            return None

        # Calculate distances from picks to reference surface
        pick_positions = points[:, :3]  # Use only x, y, z coordinates
        distances = cdist(pick_positions, reference_points)
        min_distances = np.min(distances, axis=1)

        # Find picks within distance threshold
        valid_pick_mask = min_distances <= max_distance

        if not np.any(valid_pick_mask):
            logger.warning(f"No picks within {max_distance} units of reference surface")
            return None

        # Filter picks
        valid_points = points[valid_pick_mask]
        valid_transforms = transforms[valid_pick_mask] if transforms is not None else None

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
