from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN, KMeans

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickRoot, CopickRun

logger = get_logger(__name__)


def fit_sphere_to_points(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit a sphere to a set of 3D points using least squares.

    Args:
        points: Nx3 array of points

    Returns:
        Tuple of (center, radius)
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
    spheres: List[Tuple[np.ndarray, float]], min_distance: float = None,
) -> List[Tuple[np.ndarray, float]]:
    """
    Merge spheres that are too close to each other.

    Args:
        spheres: List of (center, radius) tuples
        min_distance: Minimum distance between sphere centers. If None, uses average radius

    Returns:
        List of deduplicated (center, radius) tuples
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
    """
    Create a sphere mesh with given center and radius.

    Args:
        center: 3D center point
        radius: Sphere radius
        subdivisions: Number of subdivisions for sphere resolution

    Returns:
        Trimesh sphere object
    """
    # Create unit sphere and scale/translate
    sphere = tm.creation.icosphere(subdivisions=subdivisions, radius=radius)
    sphere.apply_translation(center)
    return sphere


def cluster(points: np.ndarray, method: str = "dbscan", **kwargs) -> List[np.ndarray]:
    """
    Cluster points using the specified method.

    Args:
        points: Nx3 array of points
        method: Clustering method ('dbscan', 'kmeans')
        **kwargs: Additional parameters for clustering

    Returns:
        List of point arrays, one per cluster
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
                if len(cluster_points) >= 4:  # Need at least 4 points for sphere fitting
                    clusters.append(cluster_points)

        return clusters

    elif method == "kmeans":
        n_clusters = kwargs.get("n_clusters", 1)

        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clustering.fit_predict(points)

        clusters = []
        for i in range(n_clusters):
            cluster_points = points[labels == i]
            if len(cluster_points) >= 4:  # Need at least 4 points for sphere fitting
                clusters.append(cluster_points)

        return clusters

    else:
        raise ValueError(f"Unknown clustering method: {method}")


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
    create_multiple: bool = False,
    deduplicate_spheres_flag: bool = True,
    min_sphere_distance: Optional[float] = None,
    individual_meshes: bool = False,
    session_id_template: Optional[str] = None,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """
    Create sphere mesh(es) from pick points.

    Args:
        points: Nx3 array of pick positions
        run: Copick run object
        object_name: Name of the mesh object
        session_id: Session ID for the mesh
        user_id: User ID for the mesh
        use_clustering: Whether to cluster points first
        clustering_method: Clustering method ('dbscan', 'kmeans')
        clustering_params: Parameters for clustering
        subdivisions: Number of subdivisions for sphere resolution
        create_multiple: If True and clustering is used, create separate meshes for each cluster
        deduplicate_spheres_flag: Whether to merge overlapping spheres
        min_sphere_distance: Minimum distance between sphere centers for deduplication
        individual_meshes: If True, create separate mesh objects for each sphere
        session_id_template: Template for individual mesh session IDs

    Returns:
        Tuple of (CopickMesh object, stats dict) or None if creation failed
        Stats dict contains 'vertices_created' and 'faces_created' totals
    """
    if len(points) < 4:
        logger.warning(f"Need at least 4 points to fit a sphere, got {len(points)}")
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
                            sphere_id=i,
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
            f"Created sphere mesh with {len(combined_mesh.vertices)} vertices and {len(combined_mesh.faces)} faces",
        )
        return copick_mesh, stats

    except Exception as e:
        logger.critical(f"Error creating mesh: {e}")
        return None


def _sphere_from_picks_worker(
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
    deduplicate_spheres_flag: bool,
    min_sphere_distance: Optional[float],
    individual_meshes: bool,
    session_id_template: Optional[str],
) -> Dict[str, Any]:
    """Worker function for batch conversion of picks to sphere meshes."""
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

        result = sphere_from_picks(
            points=positions,
            use_clustering=use_clustering,
            clustering_method=clustering_method,
            clustering_params=clustering_params,
            subdivisions=subdivisions,
            create_multiple=create_multiple,
            deduplicate_spheres_flag=deduplicate_spheres_flag,
            min_sphere_distance=min_sphere_distance,
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
            return {"processed": 0, "errors": [f"No sphere mesh generated for {run.name}"]}

    except Exception as e:
        return {"processed": 0, "errors": [f"Error processing {run.name}: {e}"]}


def sphere_from_picks_batch(
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
    deduplicate_spheres: bool = True,
    min_sphere_distance: Optional[float] = None,
    individual_meshes: bool = False,
    session_id_template: Optional[str] = None,
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Batch convert picks to sphere meshes across multiple runs.

    Parameters:
    -----------
    root : copick.Root
        The copick root containing runs to process.
    pick_object_name : str
        Name of the pick object to convert.
    pick_user_id : str
        User ID of the picks to convert.
    pick_session_id : str
        Session ID of the picks to convert.
    mesh_object_name : str
        Name of the mesh object to create.
    mesh_session_id : str
        Session ID for created mesh.
    mesh_user_id : str
        User ID for created mesh.
    use_clustering : bool, optional
        Whether to cluster points first. Default is False.
    clustering_method : str, optional
        Clustering method ('dbscan', 'kmeans'). Default is 'dbscan'.
    clustering_params : dict, optional
        Parameters for clustering method.
    subdivisions : int, optional
        Number of subdivisions for sphere resolution. Default is 2.
    create_multiple : bool, optional
        Create separate meshes for each cluster. Default is False.
    voxel_spacing : float, optional
        Voxel spacing for coordinate scaling. Default is 1.0.
    run_names : list, optional
        List of run names to process. If None, processes all runs.
    workers : int, optional
        Number of worker processes. Default is 8.

    Returns:
    --------
    dict
        Dictionary with processing results and statistics.
    """
    from copick.ops.run import map_runs

    if clustering_params is None:
        clustering_params = {}

    runs_to_process = [run.name for run in root.runs] if run_names is None else run_names

    results = map_runs(
        callback=_sphere_from_picks_worker,
        root=root,
        runs=runs_to_process,
        workers=workers,
        task_desc="Converting picks to sphere meshes",
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
        deduplicate_spheres_flag=deduplicate_spheres,
        min_sphere_distance=min_sphere_distance,
        individual_meshes=individual_meshes,
        session_id_template=session_id_template,
    )

    return results
