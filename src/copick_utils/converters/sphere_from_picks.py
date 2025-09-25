from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger
from scipy.optimize import minimize

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


def fit_sphere_to_points(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit a sphere to a set of 3D points using least squares.

    Args:
        points: Nx3 array of points.

    Returns:
        Tuple of (center, radius).
    """
    if len(points) < 4:
        raise ValueError("Need at least 4 points to fit a sphere")

    def sphere_residuals(params, points):
        """Calculate residuals for sphere fitting."""
        cx, cy, cz, r = params
        center = np.array([cx, cy, cz])
        distances = np.linalg.norm(points - center, axis=1)
        return distances - r

    # Initial guess: center at centroid, radius as average distance to centroid
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    initial_radius = np.mean(distances)

    initial_params = [centroid[0], centroid[1], centroid[2], initial_radius]

    # Fit sphere using least squares
    result = minimize(lambda params: np.sum(sphere_residuals(params, points) ** 2), initial_params, method="L-BFGS-B")

    if result.success:
        cx, cy, cz, r = result.x
        center = np.array([cx, cy, cz])
        radius = abs(r)  # Ensure positive radius
        return center, radius
    else:
        # Fallback to simple centroid and average distance
        radius = np.mean(np.linalg.norm(points - centroid, axis=1))
        return centroid, radius


def deduplicate_spheres(
    spheres: List[Tuple[np.ndarray, float]],
    min_distance: float = None,
) -> List[Tuple[np.ndarray, float]]:
    """Merge spheres that are too close to each other.

    Args:
        spheres: List of (center, radius) tuples.
        min_distance: Minimum distance between sphere centers. If None, uses average radius.

    Returns:
        List of deduplicated (center, radius) tuples.
    """
    if len(spheres) <= 1:
        return spheres

    if min_distance is None:
        # Use average radius as minimum distance
        avg_radius = np.mean([radius for _, radius in spheres])
        min_distance = avg_radius * 0.5

    deduplicated = []
    used = set()

    for i, (center1, radius1) in enumerate(spheres):
        if i in used:
            continue

        # Find all spheres close to this one
        close_spheres = [(center1, radius1)]
        used.add(i)

        for j, (center2, radius2) in enumerate(spheres):
            if j in used or i == j:
                continue

            distance = np.linalg.norm(center1 - center2)
            if distance <= min_distance:
                close_spheres.append((center2, radius2))
                used.add(j)

        if len(close_spheres) == 1:
            # Single sphere, keep as is
            deduplicated.append((center1, radius1))
        else:
            # Merge multiple close spheres
            centers = np.array([center for center, _ in close_spheres])
            radii = np.array([radius for _, radius in close_spheres])

            # Use weighted average for center (weight by volume)
            volumes = (4 / 3) * np.pi * radii**3
            weights = volumes / np.sum(volumes)
            merged_center = np.average(centers, axis=0, weights=weights)

            # Use volume-weighted average for radius
            merged_radius = np.average(radii, weights=weights)

            deduplicated.append((merged_center, merged_radius))
            logger.info(f"Merged {len(close_spheres)} overlapping spheres into one")

    return deduplicated


def create_sphere_mesh(center: np.ndarray, radius: float, subdivisions: int = 2) -> tm.Trimesh:
    """Create a sphere mesh with given center and radius.

    Args:
        center: 3D center point.
        radius: Sphere radius.
        subdivisions: Number of subdivisions for sphere resolution.

    Returns:
        Trimesh sphere object.
    """
    # Create unit sphere and scale/translate
    sphere = tm.creation.icosphere(subdivisions=subdivisions, radius=radius)
    sphere.apply_translation(center)
    return sphere


def sphere_from_picks(
    points: np.ndarray,
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    use_clustering: bool = False,
    clustering_method: str = "dbscan",
    clustering_params: Optional[Dict[str, Any]] = None,
    subdivisions: int = 2,
    all_clusters: bool = False,
    deduplicate_spheres_flag: bool = True,
    min_sphere_distance: Optional[float] = None,
    individual_meshes: bool = False,
    session_id_template: Optional[str] = None,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """Create sphere mesh(es) from pick points.

    Args:
        points: Nx3 array of pick positions.
        run: Copick run object.
        object_name: Name of the mesh object.
        session_id: Session ID for the mesh.
        user_id: User ID for the mesh.
        use_clustering: Whether to cluster points first.
        clustering_method: Clustering method ('dbscan', 'kmeans').
        clustering_params: Parameters for clustering.
            e.g.
                - {'eps': 5.0, 'min_samples': 3} for DBSCAN
                - {'n_clusters': 3} for KMeans
        subdivisions: Number of subdivisions for sphere resolution.
        all_clusters: If True and clustering is used, use all clusters. If False, use only largest cluster.
        deduplicate_spheres_flag: Whether to merge overlapping spheres.
        min_sphere_distance: Minimum distance between sphere centers for deduplication.
        individual_meshes: If True, create separate mesh objects for each sphere.
        session_id_template: Template for individual mesh session IDs.

    Returns:
        Tuple of (CopickMesh object, stats dict) or None if creation failed.
        Stats dict contains 'vertices_created' and 'faces_created' totals.
    """
    if not validate_points(points, 4, "sphere"):
        return None

    if clustering_params is None:
        clustering_params = {}

    # Handle clustering workflow with special sphere logic
    if use_clustering:
        point_clusters = cluster(points, clustering_method, 4, **clustering_params)

        if not point_clusters:
            logger.warning("No valid clusters found")
            return None

        logger.info(f"Found {len(point_clusters)} clusters")

        if all_clusters and len(point_clusters) > 1:
            # Create sphere parameters from all clusters
            sphere_params = []
            for i, cluster_points in enumerate(point_clusters):
                try:
                    center, radius = fit_sphere_to_points(cluster_points)
                    sphere_params.append((center, radius))
                    logger.info(f"Cluster {i}: sphere at {center} with radius {radius:.2f}")
                except Exception as e:
                    logger.critical(f"Failed to fit sphere to cluster {i}: {e}")
                    continue

            if not sphere_params:
                logger.warning("No valid spheres created from clusters")
                return None

            # Deduplicate overlapping spheres if requested
            if deduplicate_spheres_flag:
                final_spheres = deduplicate_spheres(sphere_params, min_sphere_distance)
            else:
                final_spheres = sphere_params

            if individual_meshes:
                # Create separate mesh objects for each sphere
                created_meshes = []
                total_vertices = 0
                total_faces = 0

                for i, (center, radius) in enumerate(final_spheres):
                    sphere_mesh = create_sphere_mesh(center, radius, subdivisions)

                    # Generate session ID using template if provided
                    if session_id_template:
                        sphere_session_id = session_id_template.format(
                            base_session_id=session_id,
                            instance_id=i,
                        )
                    else:
                        sphere_session_id = f"{session_id}-{i:03d}"

                    try:
                        copick_mesh = run.new_mesh(object_name, sphere_session_id, user_id, exist_ok=True)
                        copick_mesh.mesh = sphere_mesh
                        copick_mesh.store()
                        created_meshes.append(copick_mesh)
                        total_vertices += len(sphere_mesh.vertices)
                        total_faces += len(sphere_mesh.faces)
                        logger.info(f"Created individual sphere mesh {i} with {len(sphere_mesh.vertices)} vertices")
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
                # Create meshes from final spheres and combine them
                all_meshes = []
                for center, radius in final_spheres:
                    sphere_mesh = create_sphere_mesh(center, radius, subdivisions)
                    all_meshes.append(sphere_mesh)

                # Combine all meshes
                combined_mesh = tm.util.concatenate(all_meshes)
        else:
            # Use largest cluster
            cluster_sizes = [len(cluster) for cluster in point_clusters]
            largest_cluster_idx = np.argmax(cluster_sizes)
            points_to_use = point_clusters[largest_cluster_idx]
            logger.info(f"Using largest cluster with {len(points_to_use)} points")

            center, radius = fit_sphere_to_points(points_to_use)
            combined_mesh = create_sphere_mesh(center, radius, subdivisions)
    else:
        # Fit single sphere to all points
        center, radius = fit_sphere_to_points(points)
        combined_mesh = create_sphere_mesh(center, radius, subdivisions)
        logger.info(f"Fitted sphere at {center} with radius {radius:.2f}")

    # Store mesh and return stats
    try:
        return store_mesh_with_stats(run, combined_mesh, object_name, session_id, user_id, "sphere")
    except Exception as e:
        logger.critical(f"Error creating mesh: {e}")
        return None


# Create worker function using common infrastructure
_sphere_from_picks_worker = create_batch_worker(sphere_from_picks, "sphere", min_points=4)


# Create batch converter using common infrastructure
sphere_from_picks_batch = create_batch_converter(
    sphere_from_picks,
    "Converting picks to sphere meshes",
    "sphere",
    min_points=4,
)

# Lazy batch converter for new architecture
sphere_from_picks_lazy_batch = create_lazy_batch_converter(
    converter_func=sphere_from_picks,
    task_description="Converting picks to sphere meshes",
)
