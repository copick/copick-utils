from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger

from copick_utils.converters.converter_common import (
    cluster,
    create_batch_converter,
    create_batch_worker,
    store_mesh_with_stats,
    validate_points,
)
from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickRun

logger = get_logger(__name__)


def convex_hull_mesh(points: np.ndarray) -> tm.Trimesh:
    """Create a convex hull mesh from points.

    Args:
        points: Nx3 array of points.

    Returns:
        Trimesh object representing the convex hull.
    """
    if len(points) < 4:
        raise ValueError("Need at least 4 points to create a convex hull")

    # Use trimesh's convex hull function instead of scipy
    hull_mesh = tm.convex.convex_hull(points)
    return hull_mesh


def alpha_shape_mesh(points: np.ndarray, alpha: float) -> tm.Trimesh:
    """Create an alpha shape mesh from points.

    Args:
        points: Nx3 array of points.
        alpha: Alpha parameter controlling the shape detail.

    Returns:
        Trimesh object representing the alpha shape.
    """
    logger.warning("Alpha shape mode is currently disabled, falling back to convex hull")
    return convex_hull_mesh(points)


def mesh_from_picks(
    points: np.ndarray,
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    mesh_type: str = "convex_hull",
    alpha: Optional[float] = None,
    use_clustering: bool = False,
    clustering_method: str = "dbscan",
    clustering_params: Optional[Dict[str, Any]] = None,
    all_clusters: bool = False,
    individual_meshes: bool = False,
    session_id_template: Optional[str] = None,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """Create mesh(es) from pick points.

    Args:
        points: Nx3 array of pick positions.
        run: Copick run object.
        object_name: Name of the mesh object.
        session_id: Session ID for the mesh.
        user_id: User ID for the mesh.
        mesh_type: Type of mesh to create ('convex_hull', 'alpha_shape').
        alpha: Alpha parameter for alpha shapes (required if mesh_type='alpha_shape').
        use_clustering: Whether to cluster points first.
        clustering_method: Clustering method ('dbscan', 'kmeans').
        clustering_params: Parameters for clustering.
            e.g.
                - {'eps': 5.0, 'min_samples': 3} for DBSCAN
                - {'n_clusters': 3} for KMeans
        all_clusters: If True, use all clusters; if False, use only the largest cluster.
        individual_meshes: If True, create separate mesh objects for each mesh.
        session_id_template: Template for individual mesh session IDs.

    Returns:
        Tuple of (CopickMesh object, stats dict) or None if creation failed.
        Stats dict contains 'vertices_created' and 'faces_created' totals.
    """
    if not validate_points(points, 4, "mesh"):
        return None

    if clustering_params is None:
        clustering_params = {}

    # Define mesh creation function
    def create_mesh_from_points(cluster_points):
        if mesh_type == "convex_hull":
            return convex_hull_mesh(cluster_points)
        elif mesh_type == "alpha_shape":
            if alpha is None:
                raise ValueError("Alpha parameter is required for alpha shapes")
            return alpha_shape_mesh(cluster_points, alpha)
        else:
            raise ValueError(f"Unknown mesh type: {mesh_type}")

    # Handle clustering workflow with special mesh logic
    if use_clustering:
        point_clusters = cluster(
            points,
            clustering_method,
            min_points_per_cluster=4,  # Meshes need at least 4 points
            **clustering_params,
        )

        if not point_clusters:
            logger.warning("No valid clusters found")
            return None

        logger.info(f"Found {len(point_clusters)} clusters")

        if all_clusters and len(point_clusters) > 1:
            if individual_meshes:
                # Create separate mesh objects for each mesh
                created_meshes = []
                total_vertices = 0
                total_faces = 0

                for i, cluster_points in enumerate(point_clusters):
                    try:
                        cluster_mesh = create_mesh_from_points(cluster_points)

                        # Generate session ID using template if provided
                        if session_id_template:
                            mesh_session_id = session_id_template.format(
                                base_session_id=session_id,
                                instance_id=i,
                            )
                        else:
                            mesh_session_id = f"{session_id}-{i:03d}"

                        copick_mesh = run.new_mesh(object_name, mesh_session_id, user_id, exist_ok=True)
                        copick_mesh.mesh = cluster_mesh
                        copick_mesh.store()
                        created_meshes.append(copick_mesh)
                        total_vertices += len(cluster_mesh.vertices)
                        total_faces += len(cluster_mesh.faces)
                        logger.info(
                            f"Created individual mesh {i} with {len(cluster_mesh.vertices)} vertices",
                        )
                    except Exception as e:
                        logger.error(f"Failed to create mesh {i}: {e}")
                        continue

                # Return the first mesh and total stats
                if created_meshes:
                    stats = {"vertices_created": total_vertices, "faces_created": total_faces}
                    return created_meshes[0], stats
                else:
                    return None
            else:
                # Create meshes from all clusters and combine them
                all_meshes = []
                for cluster_points in point_clusters:
                    cluster_mesh = create_mesh_from_points(cluster_points)
                    all_meshes.append(cluster_mesh)

                # Combine all meshes
                combined_mesh = tm.util.concatenate(all_meshes)
        else:
            # Use largest cluster
            cluster_sizes = [len(cluster) for cluster in point_clusters]
            largest_cluster_idx = np.argmax(cluster_sizes)
            points_to_use = point_clusters[largest_cluster_idx]
            logger.info(f"Using largest cluster with {len(points_to_use)} points")

            combined_mesh = create_mesh_from_points(points_to_use)
    else:
        # Use all points without clustering
        combined_mesh = create_mesh_from_points(points)

    # Store mesh and return stats
    try:
        return store_mesh_with_stats(run, combined_mesh, object_name, session_id, user_id, "mesh")
    except Exception as e:
        logger.critical(f"Error creating mesh: {e}")
        return None


# Create worker function using common infrastructure
_mesh_from_picks_worker = create_batch_worker(mesh_from_picks, "mesh", "picks", min_points=4)


# Create batch converter using common infrastructure
mesh_from_picks_batch = create_batch_converter(
    mesh_from_picks,
    "Converting picks to meshes",
    "mesh",
    "picks",
    min_points=4,
)

# Lazy batch converter for new architecture
mesh_from_picks_lazy_batch = create_lazy_batch_converter(
    converter_func=mesh_from_picks,
    task_description="Converting picks to meshes",
)
