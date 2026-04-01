from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger
from sklearn.cluster import DBSCAN, KMeans

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickRun

logger = get_logger(__name__)


def validate_points(points: np.ndarray, min_count: int, shape_name: str) -> bool:
    """Validate that we have enough points for the given shape type.

    Args:
        points: Nx3 array of points.
        min_count: Minimum number of points required.
        shape_name: Name of the shape for error messages.

    Returns:
        True if valid, False otherwise.
    """
    if len(points) < min_count:
        logger.warning(f"Need at least {min_count} points to fit a {shape_name}, got {len(points)}")
        return False
    return True


def cluster(
    points: np.ndarray,
    method: str = "dbscan",
    min_points_per_cluster: int = 3,
    **kwargs,
) -> List[np.ndarray]:
    """Cluster points using the specified method.

    Args:
        points: Nx3 array of points.
        method: Clustering method ('dbscan', 'kmeans').
        min_points_per_cluster: Minimum points required per cluster.
        **kwargs: Additional parameters for clustering.

    Returns:
        List of point arrays, one per cluster.
    """
    if method == "dbscan":
        eps = kwargs.get("eps", 1.0)
        min_samples = kwargs.get("min_samples", 3)

        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(points)

        # Group points by cluster label (excluding noise points labeled as -1)
        clusters = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label != -1:  # Skip noise points
                cluster_points = points[labels == label]
                if len(cluster_points) >= min_points_per_cluster:
                    clusters.append(cluster_points)

        return clusters

    elif method == "kmeans":
        n_clusters = kwargs.get("n_clusters", 1)

        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clustering.fit_predict(points)

        clusters = []
        for i in range(n_clusters):
            cluster_points = points[labels == i]
            if len(cluster_points) >= min_points_per_cluster:
                clusters.append(cluster_points)

        return clusters

    else:
        raise ValueError(f"Unknown clustering method: {method}")


def store_mesh_with_stats(
    run: "CopickRun",
    mesh: tm.Trimesh,
    object_name: str,
    session_id: str,
    user_id: str,
    shape_name: str,
) -> Tuple["CopickMesh", Dict[str, int]]:
    """Store a mesh and return statistics.

    Args:
        run: Copick run object.
        mesh: Trimesh object to store.
        object_name: Name of the mesh object.
        session_id: Session ID for the mesh.
        user_id: User ID for the mesh.
        shape_name: Name of the shape for logging.

    Returns:
        Tuple of (CopickMesh object, stats dict).

    Raises:
        Exception: If mesh creation fails.
    """
    copick_mesh = run.new_mesh(object_name, session_id, user_id, exist_ok=True)
    copick_mesh.mesh = mesh
    copick_mesh.store()

    stats = {
        "vertices_created": len(mesh.vertices),
        "faces_created": len(mesh.faces),
    }
    logger.info(
        f"Created {shape_name} mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces",
    )
    return copick_mesh, stats
