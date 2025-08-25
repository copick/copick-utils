from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger
from sklearn.decomposition import PCA

from copick_utils.converters.converter_common import (
    create_batch_converter,
    create_batch_worker,
    handle_clustering_workflow,
    store_mesh_with_stats,
    validate_points,
)

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickRun

logger = get_logger(__name__)


def fit_plane_to_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a plane to a set of 3D points using PCA.

    Args:
        points: Nx3 array of points.

    Returns:
        Tuple of (center, normal_vector).
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane")

    # Center the points
    center = np.mean(points, axis=0)
    centered_points = points - center

    # Use PCA to find the plane
    pca = PCA(n_components=3)
    pca.fit(centered_points)

    # The normal vector is the component with the smallest variance (last component)
    normal = pca.components_[-1]  # Last principal component (smallest variance)

    # Ensure consistent normal direction (pointing upward in z if possible)
    if normal[2] < 0:
        normal = -normal

    return center, normal


def create_plane_mesh(center: np.ndarray, normal: np.ndarray, points: np.ndarray, padding: float = 1.0) -> tm.Trimesh:
    """Create a plane mesh that encompasses the given points.

    Args:
        center: Plane center point.
        normal: Plane normal vector.
        points: Original points to determine plane size.
        padding: Extra padding factor for plane size.

    Returns:
        Trimesh plane object.
    """
    # Create two orthogonal vectors in the plane
    # Find a vector that's not parallel to the normal
    if abs(normal[2]) < 0.9:
        temp_vector = np.array([0, 0, 1])
    else:
        temp_vector = np.array([1, 0, 0])

    # Create two orthogonal vectors in the plane
    u = np.cross(normal, temp_vector)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    # Project points onto the plane to determine size
    centered_points = points - center
    u_coords = np.dot(centered_points, u)
    v_coords = np.dot(centered_points, v)

    # Determine plane extents with padding
    u_min, u_max = np.min(u_coords), np.max(u_coords)
    v_min, v_max = np.min(v_coords), np.max(v_coords)

    u_range = (u_max - u_min) * padding
    v_range = (v_max - v_min) * padding

    u_center = (u_min + u_max) / 2
    v_center = (v_min + v_max) / 2

    # Create plane vertices
    vertices = [
        center + (u_center - u_range / 2) * u + (v_center - v_range / 2) * v,
        center + (u_center + u_range / 2) * u + (v_center - v_range / 2) * v,
        center + (u_center + u_range / 2) * u + (v_center + v_range / 2) * v,
        center + (u_center - u_range / 2) * u + (v_center + v_range / 2) * v,
    ]

    # Create two triangular faces for the plane
    faces = [[0, 1, 2], [0, 2, 3]]

    return tm.Trimesh(vertices=vertices, faces=faces)


def plane_from_picks(
    points: np.ndarray,
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    use_clustering: bool = False,
    clustering_method: str = "dbscan",
    clustering_params: Optional[Dict[str, Any]] = None,
    padding: float = 1.2,
    all_clusters: bool = True,
    individual_meshes: bool = False,
    session_id_template: Optional[str] = None,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """Create plane mesh(es) from pick points.

    Args:
        points: Nx3 array of pick positions.
        run: Copick run object.
        object_name: Name of the mesh object.
        session_id: Session ID for the mesh.
        user_id: User ID for the mesh.
        use_clustering: Whether to cluster points first.
        clustering_method: Clustering method ('dbscan', 'kmeans').
        clustering_params: Parameters for clustering.
        padding: Padding factor for plane size (1.0 = exact fit, >1.0 = larger plane).
        all_clusters: If True, use all clusters; if False, use only the largest cluster.
        individual_meshes: If True, create separate mesh objects for each plane.
        session_id_template: Template for individual mesh session IDs.

    Returns:
        Tuple of (CopickMesh object, stats dict) or None if creation failed.
        Stats dict contains 'vertices_created' and 'faces_created' totals.
    """
    if not validate_points(points, 3, "plane"):
        return None

    if clustering_params is None:
        clustering_params = {}

    # Define plane creation function for clustering workflow
    def create_plane_from_points(cluster_points):
        center, normal = fit_plane_to_points(cluster_points)
        return create_plane_mesh(center, normal, cluster_points, padding)

    # Handle clustering workflow
    combined_mesh, points_used = handle_clustering_workflow(
        points=points,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        all_clusters=all_clusters,
        min_points_per_cluster=3,
        shape_creation_func=create_plane_from_points,
        shape_name="plane",
    )

    if combined_mesh is None:
        return None

    # Store mesh and return stats
    try:
        return store_mesh_with_stats(run, combined_mesh, object_name, session_id, user_id, "plane")
    except Exception as e:
        logger.critical(f"Error creating mesh: {e}")
        return None


# Create worker function using common infrastructure
_plane_from_picks_worker = create_batch_worker(plane_from_picks, "plane", min_points=3)


# Create batch converter using common infrastructure
plane_from_picks_batch = create_batch_converter(
    _plane_from_picks_worker,
    "Converting picks to plane meshes",
)
