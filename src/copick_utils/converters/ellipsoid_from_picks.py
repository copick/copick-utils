from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickRoot, CopickRun

logger = get_logger(__name__)


def fit_ellipsoid_to_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an ellipsoid to a set of 3D points using PCA and least squares.

    Args:
        points: Nx3 array of points.

    Returns:
        Tuple of (center, semi_axes, rotation_matrix).
    """
    if len(points) < 6:
        raise ValueError("Need at least 6 points to fit an ellipsoid")
    
    # Center the points
    center = np.mean(points, axis=0)
    centered_points = points - center
    
    # Use PCA to find principal axes
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    
    # Transform to PCA coordinates
    transformed_points = pca.transform(centered_points)
    
    # Estimate semi-axes lengths from the spread in each direction
    semi_axes = np.sqrt(np.var(transformed_points, axis=0)) * 2  # 2 standard deviations
    
    # Ensure positive and reasonable semi-axes
    semi_axes = np.maximum(semi_axes, 0.1)
    
    # Sort semi-axes in descending order and reorder components
    sorted_indices = np.argsort(semi_axes)[::-1]
    semi_axes = semi_axes[sorted_indices]
    rotation_matrix = pca.components_[sorted_indices]
    
    return center, semi_axes, rotation_matrix


def deduplicate_ellipsoids(
    ellipsoids: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    min_distance: float = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Merge ellipsoids that are too close to each other.

    Args:
        ellipsoids: List of (center, semi_axes, rotation_matrix) tuples.
        min_distance: Minimum distance between ellipsoid centers. If None, uses average of largest semi-axes.

    Returns:
        List of deduplicated (center, semi_axes, rotation_matrix) tuples.
    """
    if len(ellipsoids) <= 1:
        return ellipsoids

    if min_distance is None:
        # Use average of largest semi-axes as minimum distance
        avg_major_axis = np.mean([semi_axes[0] for _, semi_axes, _ in ellipsoids])
        min_distance = avg_major_axis * 0.5

    deduplicated = []
    used = set()

    for i, (center1, semi_axes1, rotation1) in enumerate(ellipsoids):
        if i in used:
            continue

        # Find all ellipsoids close to this one
        close_ellipsoids = [(center1, semi_axes1, rotation1)]
        used.add(i)

        for j, (center2, semi_axes2, rotation2) in enumerate(ellipsoids):
            if j in used or i == j:
                continue

            distance = np.linalg.norm(center1 - center2)
            if distance <= min_distance:
                close_ellipsoids.append((center2, semi_axes2, rotation2))
                used.add(j)

        if len(close_ellipsoids) == 1:
            # Single ellipsoid, keep as is
            deduplicated.append((center1, semi_axes1, rotation1))
        else:
            # Merge multiple close ellipsoids
            centers = np.array([center for center, _, _ in close_ellipsoids])
            all_semi_axes = np.array([semi_axes for _, semi_axes, _ in close_ellipsoids])

            # Use volume-weighted average for center and semi-axes
            volumes = (4 / 3) * np.pi * np.prod(all_semi_axes, axis=1)
            weights = volumes / np.sum(volumes)
            merged_center = np.average(centers, axis=0, weights=weights)
            merged_semi_axes = np.average(all_semi_axes, axis=0, weights=weights)

            # Use first rotation matrix (could be improved with proper rotation averaging)
            merged_rotation = close_ellipsoids[0][2]

            deduplicated.append((merged_center, merged_semi_axes, merged_rotation))
            logger.info(f"Merged {len(close_ellipsoids)} overlapping ellipsoids into one")

    return deduplicated


def create_ellipsoid_mesh(center: np.ndarray, semi_axes: np.ndarray, 
                         rotation_matrix: np.ndarray, subdivisions: int = 2) -> tm.Trimesh:
    """Create an ellipsoid mesh with given center, semi-axes, and orientation.

    Args:
        center: 3D center point.
        semi_axes: Three semi-axis lengths [a, b, c].
        rotation_matrix: 3x3 rotation matrix.
        subdivisions: Number of subdivisions for ellipsoid resolution.

    Returns:
        Trimesh ellipsoid object.
    """
    # Create unit sphere
    sphere = tm.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    
    # Scale by semi-axes to create ellipsoid
    scale_matrix = np.diag([semi_axes[0], semi_axes[1], semi_axes[2]])
    
    # Apply scaling
    ellipsoid_vertices = sphere.vertices @ scale_matrix.T
    
    # Apply rotation
    ellipsoid_vertices = ellipsoid_vertices @ rotation_matrix
    
    # Translate to center
    ellipsoid_vertices += center
    
    # Create new mesh
    ellipsoid = tm.Trimesh(vertices=ellipsoid_vertices, faces=sphere.faces)
    
    return ellipsoid


def cluster(points: np.ndarray, method: str = "dbscan", **kwargs) -> List[np.ndarray]:
    """Cluster points using the specified method.

    Args:
        points: Nx3 array of points.
        method: Clustering method ('dbscan', 'kmeans').
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
                if len(cluster_points) >= 6:  # Need at least 6 points for ellipsoid fitting
                    clusters.append(cluster_points)
        
        return clusters
        
    elif method == "kmeans":
        n_clusters = kwargs.get("n_clusters", 1)
        
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clustering.fit_predict(points)
        
        clusters = []
        for i in range(n_clusters):
            cluster_points = points[labels == i]
            if len(cluster_points) >= 6:  # Need at least 6 points for ellipsoid fitting
                clusters.append(cluster_points)
        
        return clusters
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def ellipsoid_from_picks(
    points: np.ndarray,
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    use_clustering: bool = False,
    clustering_method: str = "dbscan",
    clustering_params: Optional[Dict[str, Any]] = None,
    subdivisions: int = 2,
    create_multiple: bool = False,
    deduplicate_ellipsoids_flag: bool = True,
    min_ellipsoid_distance: Optional[float] = None,
    individual_meshes: bool = False,
    session_id_template: Optional[str] = None,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """Create ellipsoid mesh(es) from pick points.

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
        subdivisions: Number of subdivisions for ellipsoid resolution.
        create_multiple: If True and clustering is used, create separate meshes for each cluster.
        deduplicate_ellipsoids_flag: Whether to merge overlapping ellipsoids.
        min_ellipsoid_distance: Minimum distance between ellipsoid centers for deduplication.
        individual_meshes: If True, create separate mesh objects for each ellipsoid.
        session_id_template: Template for individual mesh session IDs.

    Returns:
        Tuple of (CopickMesh object, stats dict) or None if creation failed.
        Stats dict contains 'vertices_created' and 'faces_created' totals.
    """
    if len(points) < 6:
        logger.warning(f"Need at least 6 points to fit an ellipsoid, got {len(points)}")
        return None
    
    if clustering_params is None:
        clustering_params = {}
    
    # Cluster points if requested
    if use_clustering:
        point_clusters = cluster(points, clustering_method, **clustering_params)
        
        if not point_clusters:
            logger.warning("No valid clusters found")
            return None
        
        logger.info(f"Found {len(point_clusters)} clusters")
        
        if create_multiple and len(point_clusters) > 1:
            # Create ellipsoid parameters from all clusters
            ellipsoid_params = []
            for i, cluster_points in enumerate(point_clusters):
                try:
                    center, semi_axes, rotation_matrix = fit_ellipsoid_to_points(cluster_points)
                    ellipsoid_params.append((center, semi_axes, rotation_matrix))
                    logger.info(f"Cluster {i}: ellipsoid at {center} with semi-axes {semi_axes}")
                except Exception as e:
                    logger.critical(f"Failed to fit ellipsoid to cluster {i}: {e}")
                    continue

            if not ellipsoid_params:
                logger.warning("No valid ellipsoids created from clusters")
                return None

            # Deduplicate overlapping ellipsoids if requested
            if deduplicate_ellipsoids_flag:
                final_ellipsoids = deduplicate_ellipsoids(ellipsoid_params, min_ellipsoid_distance)
            else:
                final_ellipsoids = ellipsoid_params

            if individual_meshes:
                # Create separate mesh objects for each ellipsoid
                created_meshes = []
                total_vertices = 0
                total_faces = 0

                for i, (center, semi_axes, rotation_matrix) in enumerate(final_ellipsoids):
                    ellipsoid_mesh = create_ellipsoid_mesh(center, semi_axes, rotation_matrix, subdivisions)

                    # Generate session ID using template if provided
                    if session_id_template:
                        ellipsoid_session_id = session_id_template.format(
                            base_session_id=session_id,
                            ellipsoid_id=i,
                        )
                    else:
                        ellipsoid_session_id = f"{session_id}-{i:03d}"

                    try:
                        copick_mesh = run.new_mesh(object_name, ellipsoid_session_id, user_id, exist_ok=True)
                        copick_mesh.mesh = ellipsoid_mesh
                        copick_mesh.store()
                        created_meshes.append(copick_mesh)
                        total_vertices += len(ellipsoid_mesh.vertices)
                        total_faces += len(ellipsoid_mesh.faces)
                        logger.info(f"Created individual ellipsoid mesh {i} with {len(ellipsoid_mesh.vertices)} vertices")
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
                # Create meshes from final ellipsoids and combine them
                all_meshes = []
                for center, semi_axes, rotation_matrix in final_ellipsoids:
                    ellipsoid_mesh = create_ellipsoid_mesh(center, semi_axes, rotation_matrix, subdivisions)
                    all_meshes.append(ellipsoid_mesh)

                # Combine all meshes
                combined_mesh = tm.util.concatenate(all_meshes)
        else:
            # Use largest cluster
            cluster_sizes = [len(cluster) for cluster in point_clusters]
            largest_cluster_idx = np.argmax(cluster_sizes)
            points_to_use = point_clusters[largest_cluster_idx]
            logger.info(f"Using largest cluster with {len(points_to_use)} points")
            
            center, semi_axes, rotation_matrix = fit_ellipsoid_to_points(points_to_use)
            combined_mesh = create_ellipsoid_mesh(center, semi_axes, rotation_matrix, subdivisions)
    else:
        # Fit single ellipsoid to all points
        center, semi_axes, rotation_matrix = fit_ellipsoid_to_points(points)
        combined_mesh = create_ellipsoid_mesh(center, semi_axes, rotation_matrix, subdivisions)
        logger.info(f"Fitted ellipsoid at {center} with semi-axes {semi_axes}")
    
    # Create copick mesh
    try:
        copick_mesh = run.new_mesh(object_name, session_id, user_id, exist_ok=True)
        copick_mesh.mesh = combined_mesh
        copick_mesh.store()

        stats = {
            "vertices_created": len(combined_mesh.vertices),
            "faces_created": len(combined_mesh.faces),
        }
        logger.info(
            f"Created ellipsoid mesh with {len(combined_mesh.vertices)} vertices and {len(combined_mesh.faces)} faces",
        )
        return copick_mesh, stats
        
    except Exception as e:
        logger.critical(f"Error creating mesh: {e}")
        return None


def _ellipsoid_from_picks_worker(
    run: "CopickRun",
    pick_object_name: str,
    pick_user_id: str,
    pick_session_id: str,
    mesh_object_name: str,
    mesh_session_id: str,
    mesh_user_id: str,
    use_clustering: bool,
    clustering_method: str,
    clustering_params: Dict[str, Any],
    subdivisions: int,
    create_multiple: bool,
    deduplicate_ellipsoids_flag: bool,
    min_ellipsoid_distance: Optional[float],
    individual_meshes: bool,
    session_id_template: Optional[str],
) -> Dict[str, Any]:
    """Worker function for batch conversion of picks to ellipsoid meshes."""
    try:
        # Get picks
        picks_list = run.get_picks(object_name=pick_object_name, user_id=pick_user_id, session_id=pick_session_id)

        if not picks_list:
            return {"processed": 0, "errors": [f"No picks found for {run.name}"]}

        picks = picks_list[0]
        points, _ = picks.numpy()

        if points is None or len(points) == 0:
            return {"processed": 0, "errors": [f"Could not load pick data for {run.name}"]}

        # Use points directly - copick coordinates are already in angstroms
        positions = points[:, :3]

        result = ellipsoid_from_picks(
            points=positions,
            use_clustering=use_clustering,
            clustering_method=clustering_method,
            clustering_params=clustering_params,
            subdivisions=subdivisions,
            create_multiple=create_multiple,
            deduplicate_ellipsoids_flag=deduplicate_ellipsoids_flag,
            min_ellipsoid_distance=min_ellipsoid_distance,
            individual_meshes=individual_meshes,
            session_id_template=session_id_template,
            run=run,
            object_name=mesh_object_name,
            session_id=mesh_session_id,
            user_id=mesh_user_id,
        )

        if result:
            mesh_obj, stats = result
            return {
                "processed": 1,
                "errors": [],
                "result": mesh_obj,
                "vertices_created": stats["vertices_created"],
                "faces_created": stats["faces_created"],
            }
        else:
            return {"processed": 0, "errors": [f"No ellipsoid mesh generated for {run.name}"]}

    except Exception as e:
        return {"processed": 0, "errors": [f"Error processing {run.name}: {e}"]}


def ellipsoid_from_picks_batch(
    root: "CopickRoot",
    pick_object_name: str,
    pick_user_id: str,
    pick_session_id: str,
    mesh_object_name: str,
    mesh_session_id: str,
    mesh_user_id: str,
    use_clustering: bool = False,
    clustering_method: str = "dbscan",
    clustering_params: Optional[Dict[str, Any]] = None,
    subdivisions: int = 2,
    create_multiple: bool = False,
    deduplicate_ellipsoids: bool = True,
    min_ellipsoid_distance: Optional[float] = None,
    individual_meshes: bool = False,
    session_id_template: Optional[str] = None,
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """Batch convert picks to ellipsoid meshes across multiple runs.

    Args:
        root: The copick root containing runs to process.
        pick_object_name: Name of the pick object to convert.
        pick_user_id: User ID of the picks to convert.
        pick_session_id: Session ID of the picks to convert.
        mesh_object_name: Name of the mesh object to create.
        mesh_session_id: Session ID for created mesh.
        mesh_user_id: User ID for created mesh.
        use_clustering: Whether to cluster points first. Default is False.
        clustering_method: Clustering method ('dbscan', 'kmeans'). Default is 'dbscan'.
        clustering_params: Parameters for clustering method.
        subdivisions: Number of subdivisions for ellipsoid resolution. Default is 2.
        create_multiple: Create separate meshes for each cluster. Default is False.
        deduplicate_ellipsoids: Whether to merge overlapping ellipsoids. Default is True.
        min_ellipsoid_distance: Minimum distance between ellipsoid centers for deduplication.
        individual_meshes: If True, create separate mesh objects for each ellipsoid. Default is False.
        session_id_template: Template for individual mesh session IDs.
        run_names: List of run names to process. If None, processes all runs.
        workers: Number of worker processes. Default is 8.

    Returns:
        Dictionary with processing results and statistics.
    """
    from copick.ops.run import map_runs
    
    if clustering_params is None:
        clustering_params = {}
    
    runs_to_process = [run.name for run in root.runs] if run_names is None else run_names
    
    results = map_runs(
        callback=_ellipsoid_from_picks_worker,
        root=root,
        runs=runs_to_process,
        workers=workers,
        task_desc="Converting picks to ellipsoid meshes",
        pick_object_name=pick_object_name,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        mesh_object_name=mesh_object_name,
        mesh_session_id=mesh_session_id,
        mesh_user_id=mesh_user_id,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        subdivisions=subdivisions,
        create_multiple=create_multiple,
        deduplicate_ellipsoids_flag=deduplicate_ellipsoids,
        min_ellipsoid_distance=min_ellipsoid_distance,
        individual_meshes=individual_meshes,
        session_id_template=session_id_template,
    )
    
    return results