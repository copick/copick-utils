from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh as tm
from copick.util.log import get_logger
from scipy.interpolate import Rbf, griddata
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickRoot, CopickRun

logger = get_logger(__name__)


def fit_2d_surface_to_points(points: np.ndarray, method: str = "delaunay", 
                           grid_resolution: int = 50) -> tm.Trimesh:
    """Fit a 2D surface to 3D points using different interpolation methods.

    Args:
        points: Nx3 array of points.
        method: Surface fitting method ('delaunay', 'rbf', 'grid').
        grid_resolution: Resolution for grid-based methods.

    Returns:
        Trimesh surface object.
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a surface")
    
    if method == "delaunay":
        return delaunay_surface(points)
    elif method == "rbf":
        return rbf_surface(points, grid_resolution)
    elif method == "grid":
        return grid_surface(points, grid_resolution)
    else:
        raise ValueError(f"Unknown surface method: {method}")


def delaunay_surface(points: np.ndarray) -> tm.Trimesh:
    """Create a surface using Delaunay triangulation.

    Args:
        points: Nx3 array of points.

    Returns:
        Trimesh surface object.
    """
    # Find the best 2D projection plane using PCA
    center = np.mean(points, axis=0)
    centered_points = points - center
    
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    
    # Use first two principal components for 2D projection
    projected_2d = pca.transform(centered_points)[:, :2]
    
    # Create Delaunay triangulation in 2D
    tri = Delaunay(projected_2d)
    
    # Use original 3D points as vertices with 2D triangulation
    return tm.Trimesh(vertices=points, faces=tri.simplices)


def rbf_surface(points: np.ndarray, grid_resolution: int) -> tm.Trimesh:
    """Create a surface using RBF (Radial Basis Function) interpolation.

    Args:
        points: Nx3 array of points.
        grid_resolution: Resolution of the output grid.

    Returns:
        Trimesh surface object.
    """
    # Find the dominant plane using PCA
    center = np.mean(points, axis=0)
    centered_points = points - center
    
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    
    # Project points onto the first two principal components
    projected_2d = pca.transform(centered_points)[:, :2]
    heights = pca.transform(centered_points)[:, 2]  # Third component as height
    
    # Create grid for interpolation
    x_min, x_max = projected_2d[:, 0].min(), projected_2d[:, 0].max()
    y_min, y_max = projected_2d[:, 1].min(), projected_2d[:, 1].max()
    
    xi = np.linspace(x_min, y_min, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # RBF interpolation
    rbf = Rbf(projected_2d[:, 0], projected_2d[:, 1], heights, function='thin_plate')
    zi_grid = rbf(xi_grid, yi_grid)
    
    # Convert grid back to 3D coordinates
    grid_points_2d = np.column_stack([xi_grid.flatten(), yi_grid.flatten(), zi_grid.flatten()])
    grid_points_3d = pca.inverse_transform(grid_points_2d) + center
    
    # Create triangulation for the grid
    vertices = grid_points_3d.reshape((grid_resolution, grid_resolution, 3))
    faces = []
    
    for i in range(grid_resolution - 1):
        for j in range(grid_resolution - 1):
            # Two triangles per grid cell
            v1 = i * grid_resolution + j
            v2 = i * grid_resolution + (j + 1)
            v3 = (i + 1) * grid_resolution + j
            v4 = (i + 1) * grid_resolution + (j + 1)
            
            faces.extend([[v1, v2, v3], [v2, v4, v3]])
    
    return tm.Trimesh(vertices=grid_points_3d, faces=faces)


def grid_surface(points: np.ndarray, grid_resolution: int) -> tm.Trimesh:
    """Create a surface using grid-based interpolation.

    Args:
        points: Nx3 array of points.
        grid_resolution: Resolution of the output grid.

    Returns:
        Trimesh surface object.
    """
    # Find bounding box
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # Find the dimension with smallest range (likely the "height" dimension)
    ranges = max_coords - min_coords
    height_dim = np.argmin(ranges)
    
    # Use other two dimensions for grid
    other_dims = [i for i in range(3) if i != height_dim]
    
    # Create grid
    x_coords = np.linspace(min_coords[other_dims[0]], max_coords[other_dims[0]], grid_resolution)
    y_coords = np.linspace(min_coords[other_dims[1]], max_coords[other_dims[1]], grid_resolution)
    xi, yi = np.meshgrid(x_coords, y_coords)
    
    # Interpolate height values
    zi = griddata(
        points[:, other_dims], 
        points[:, height_dim], 
        (xi, yi), 
        method='linear',
        fill_value=np.mean(points[:, height_dim])
    )
    
    # Build 3D vertices
    vertices = np.zeros((grid_resolution * grid_resolution, 3))
    vertices[:, other_dims[0]] = xi.flatten()
    vertices[:, other_dims[1]] = yi.flatten() 
    vertices[:, height_dim] = zi.flatten()
    
    # Create triangulation
    faces = []
    for i in range(grid_resolution - 1):
        for j in range(grid_resolution - 1):
            # Two triangles per grid cell
            v1 = i * grid_resolution + j
            v2 = i * grid_resolution + (j + 1)
            v3 = (i + 1) * grid_resolution + j
            v4 = (i + 1) * grid_resolution + (j + 1)
            
            faces.extend([[v1, v2, v3], [v2, v4, v3]])
    
    return tm.Trimesh(vertices=vertices, faces=faces)


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
                if len(cluster_points) >= 3:  # Need at least 3 points for surface fitting
                    clusters.append(cluster_points)
        
        return clusters
        
    elif method == "kmeans":
        n_clusters = kwargs.get("n_clusters", 1)
        
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clustering.fit_predict(points)
        
        clusters = []
        for i in range(n_clusters):
            cluster_points = points[labels == i]
            if len(cluster_points) >= 3:  # Need at least 3 points for surface fitting
                clusters.append(cluster_points)
        
        return clusters
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def surface_from_picks(
    points: np.ndarray,
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    surface_method: str = "delaunay",
    grid_resolution: int = 50,
    use_clustering: bool = False,
    clustering_method: str = "dbscan",
    clustering_params: Optional[Dict[str, Any]] = None,
    create_multiple: bool = False,
    individual_meshes: bool = False,
    session_id_template: Optional[str] = None,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """Create surface mesh(es) from pick points.

    Args:
        points: Nx3 array of pick positions.
        run: Copick run object.
        object_name: Name of the mesh object.
        session_id: Session ID for the mesh.
        user_id: User ID for the mesh.
        surface_method: Surface fitting method ('delaunay', 'rbf', 'grid').
        grid_resolution: Resolution for grid-based methods.
        use_clustering: Whether to cluster points first.
        clustering_method: Clustering method ('dbscan', 'kmeans').
        clustering_params: Parameters for clustering.
        create_multiple: If True and clustering is used, create separate meshes for each cluster.
        individual_meshes: If True, create separate mesh objects for each surface.
        session_id_template: Template for individual mesh session IDs.

    Returns:
        Tuple of (CopickMesh object, stats dict) or None if creation failed.
        Stats dict contains 'vertices_created' and 'faces_created' totals.
    """
    if len(points) < 3:
        logger.warning(f"Need at least 3 points to fit a surface, got {len(points)}")
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
            # Create combined mesh from all clusters
            all_meshes = []
            for i, cluster_points in enumerate(point_clusters):
                try:
                    surface_mesh = fit_2d_surface_to_points(cluster_points, surface_method, grid_resolution)
                    all_meshes.append(surface_mesh)
                    logger.info(f"Cluster {i}: surface with {len(surface_mesh.vertices)} vertices")
                except Exception as e:
                    logger.critical(f"Failed to fit surface to cluster {i}: {e}")
                    continue
            
            if not all_meshes:
                logger.warning("No valid surfaces created from clusters")
                return None
            
            # Combine all meshes
            combined_mesh = tm.util.concatenate(all_meshes)
        else:
            # Use largest cluster
            cluster_sizes = [len(cluster) for cluster in point_clusters]
            largest_cluster_idx = np.argmax(cluster_sizes)
            points_to_use = point_clusters[largest_cluster_idx]
            logger.info(f"Using largest cluster with {len(points_to_use)} points")
            
            combined_mesh = fit_2d_surface_to_points(points_to_use, surface_method, grid_resolution)
    else:
        # Fit single surface to all points
        combined_mesh = fit_2d_surface_to_points(points, surface_method, grid_resolution)
        logger.info(f"Fitted {surface_method} surface with {len(combined_mesh.vertices)} vertices")
    
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
            f"Created surface mesh with {len(combined_mesh.vertices)} vertices and {len(combined_mesh.faces)} faces",
        )
        return copick_mesh, stats
        
    except Exception as e:
        logger.critical(f"Error creating mesh: {e}")
        return None


def _surface_from_picks_worker(
    run: "CopickRun",
    pick_object_name: str,
    pick_user_id: str,
    pick_session_id: str,
    mesh_object_name: str,
    mesh_session_id: str,
    mesh_user_id: str,
    surface_method: str,
    grid_resolution: int,
    use_clustering: bool,
    clustering_method: str,
    clustering_params: Dict[str, Any],
    create_multiple: bool,
    individual_meshes: bool,
    session_id_template: Optional[str],
) -> Dict[str, Any]:
    """Worker function for batch conversion of picks to surface meshes."""
    try:
        # Get picks
        picks_list = run.get_picks(
            object_name=pick_object_name,
            user_id=pick_user_id,
            session_id=pick_session_id
        )
        
        if not picks_list:
            return {"processed": 0, "errors": [f"No picks found for {run.name}"]}
        
        picks = picks_list[0]
        points = picks.numpy()
        
        if points is None or len(points) == 0:
            return {"processed": 0, "errors": [f"Could not load pick data for {run.name}"]}
        
        # Use points directly - copick coordinates are already in angstroms
        positions = points[:, :3]
        
        mesh_obj = surface_from_picks(
            points=positions,
            surface_method=surface_method,
            grid_resolution=grid_resolution,
            use_clustering=use_clustering,
            clustering_method=clustering_method,
            clustering_params=clustering_params,
            create_multiple=create_multiple,
            individual_meshes=individual_meshes,
            session_id_template=session_id_template,
            run=run,
            object_name=mesh_object_name,
            session_id=mesh_session_id,
            user_id=mesh_user_id,
        )
        
        if mesh_obj:
            copick_mesh, stats = mesh_obj
            return {
                "processed": 1, 
                "errors": [], 
                "result": copick_mesh,
                "vertices_created": stats["vertices_created"],
                "faces_created": stats["faces_created"]
            }
        else:
            return {"processed": 0, "errors": [f"No surface mesh generated for {run.name}"]}
        
    except Exception as e:
        return {"processed": 0, "errors": [f"Error processing {run.name}: {e}"]}


def surface_from_picks_batch(
    root: "CopickRoot",
    pick_object_name: str,
    pick_user_id: str,
    pick_session_id: str,
    mesh_object_name: str,
    mesh_session_id: str,
    mesh_user_id: str,
    surface_method: str = "delaunay",
    grid_resolution: int = 50,
    use_clustering: bool = False,
    clustering_method: str = "dbscan",
    clustering_params: Optional[Dict[str, Any]] = None,
    create_multiple: bool = False,
    individual_meshes: bool = False,
    session_id_template: Optional[str] = None,
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Batch convert picks to surface meshes across multiple runs.
    
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
    surface_method : str, optional
        Surface fitting method ('delaunay', 'rbf', 'grid'). Default is 'delaunay'.
    grid_resolution : int, optional
        Resolution for grid-based methods. Default is 50.
    use_clustering : bool, optional
        Whether to cluster points first. Default is False.
    clustering_method : str, optional
        Clustering method ('dbscan', 'kmeans'). Default is 'dbscan'.
    clustering_params : dict, optional
        Parameters for clustering method.
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
        callback=_surface_from_picks_worker,
        root=root,
        runs=runs_to_process,
        workers=workers,
        task_desc="Converting picks to surface meshes",
        pick_object_name=pick_object_name,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        mesh_object_name=mesh_object_name,
        mesh_session_id=mesh_session_id,
        mesh_user_id=mesh_user_id,
        surface_method=surface_method,
        grid_resolution=grid_resolution,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        create_multiple=create_multiple,
        individual_meshes=individual_meshes,
        session_id_template=session_id_template,
    )
    
    return results