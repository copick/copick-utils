from typing import TYPE_CHECKING, Optional, Dict, Any, List, Tuple
import numpy as np
import trimesh as tm
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickMesh, CopickRoot


def fit_plane_to_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a plane to a set of 3D points using PCA.
    
    Args:
        points: Nx3 array of points
        
    Returns:
        Tuple of (center, normal_vector)
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


def create_plane_mesh(center: np.ndarray, normal: np.ndarray, 
                     points: np.ndarray, padding: float = 1.0) -> tm.Trimesh:
    """
    Create a plane mesh that encompasses the given points.
    
    Args:
        center: Plane center point
        normal: Plane normal vector
        points: Original points to determine plane size
        padding: Extra padding factor for plane size
        
    Returns:
        Trimesh plane object
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
        center + (u_center - u_range/2) * u + (v_center - v_range/2) * v,
        center + (u_center + u_range/2) * u + (v_center - v_range/2) * v,
        center + (u_center + u_range/2) * u + (v_center + v_range/2) * v,
        center + (u_center - u_range/2) * u + (v_center + v_range/2) * v,
    ]
    
    # Create two triangular faces for the plane
    faces = [
        [0, 1, 2],
        [0, 2, 3]
    ]
    
    return tm.Trimesh(vertices=vertices, faces=faces)


def cluster_points(points: np.ndarray, method: str = "dbscan", **kwargs) -> List[np.ndarray]:
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
                if len(cluster_points) >= 3:  # Need at least 3 points for plane fitting
                    clusters.append(cluster_points)
        
        return clusters
        
    elif method == "kmeans":
        n_clusters = kwargs.get("n_clusters", 1)
        
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clustering.fit_predict(points)
        
        clusters = []
        for i in range(n_clusters):
            cluster_points = points[labels == i]
            if len(cluster_points) >= 3:  # Need at least 3 points for plane fitting
                clusters.append(cluster_points)
        
        return clusters
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")


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
    create_multiple: bool = False,
) -> Optional["CopickMesh"]:
    """
    Create plane mesh(es) from pick points.
    
    Args:
        points: Nx3 array of pick positions
        run: Copick run object
        object_name: Name of the mesh object
        session_id: Session ID for the mesh
        user_id: User ID for the mesh
        use_clustering: Whether to cluster points first
        clustering_method: Clustering method ('dbscan', 'kmeans')
        clustering_params: Parameters for clustering
        padding: Padding factor for plane size (1.0 = exact fit, >1.0 = larger plane)
        create_multiple: If True and clustering is used, create separate meshes for each cluster

    Returns:
        CopickMesh object or None if creation failed
    """
    if len(points) < 3:
        print(f"Need at least 3 points to fit a plane, got {len(points)}")
        return None
    
    if clustering_params is None:
        clustering_params = {}
    
    # Cluster points if requested
    if use_clustering:
        point_clusters = cluster_points(points, clustering_method, **clustering_params)
        
        if not point_clusters:
            print("No valid clusters found")
            return None
        
        print(f"Found {len(point_clusters)} clusters")
        
        if create_multiple and len(point_clusters) > 1:
            # Create combined mesh from all clusters
            all_meshes = []
            for i, cluster_points in enumerate(point_clusters):
                try:
                    center, normal = fit_plane_to_points(cluster_points)
                    plane_mesh = create_plane_mesh(center, normal, cluster_points, padding)
                    all_meshes.append(plane_mesh)
                    print(f"Cluster {i}: plane at {center} with normal {normal}")
                except Exception as e:
                    print(f"Failed to fit plane to cluster {i}: {e}")
                    continue
            
            if not all_meshes:
                print("No valid planes created from clusters")
                return None
            
            # Combine all meshes
            combined_mesh = tm.util.concatenate(all_meshes)
        else:
            # Use largest cluster
            cluster_sizes = [len(cluster) for cluster in point_clusters]
            largest_cluster_idx = np.argmax(cluster_sizes)
            points_to_use = point_clusters[largest_cluster_idx]
            print(f"Using largest cluster with {len(points_to_use)} points")
            
            center, normal = fit_plane_to_points(points_to_use)
            combined_mesh = create_plane_mesh(center, normal, points_to_use, padding)
    else:
        # Fit single plane to all points
        center, normal = fit_plane_to_points(points)
        combined_mesh = create_plane_mesh(center, normal, points, padding)
        print(f"Fitted plane at {center} with normal {normal}")
    
    # Create copick mesh
    try:
        copick_mesh = run.new_mesh(object_name, session_id, user_id, exist_ok=True)
        copick_mesh.mesh = combined_mesh
        copick_mesh.store()
        
        print(f"Created plane mesh with {len(combined_mesh.vertices)} vertices and {len(combined_mesh.faces)} faces")
        return copick_mesh
        
    except Exception as e:
        print(f"Error creating mesh: {e}")
        return None


def _plane_from_picks_worker(
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
    padding: float,
    create_multiple: bool,
    voxel_spacing: float,
    root: "CopickRoot",
) -> Dict[str, Any]:
    """Worker function for batch conversion of picks to plane meshes."""
    try:
        # Check if mesh object exists in config
        mesh_object = root.get_object(mesh_object_name)
        if not mesh_object:
            return {"processed": 0, "errors": [f"Mesh object '{mesh_object_name}' not found in config"]}
        
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
        
        # Scale points by voxel spacing
        positions = points[:, :3] / voxel_spacing
        
        mesh_obj = plane_from_picks(
            points=positions,
            use_clustering=use_clustering,
            clustering_method=clustering_method,
            clustering_params=clustering_params,
            padding=padding,
            create_multiple=create_multiple,
            run=run,
            object_name=mesh_object_name,
            session_id=mesh_session_id,
            user_id=mesh_user_id,
        )
        
        if mesh_obj and mesh_obj.mesh:
            return {
                "processed": 1, 
                "errors": [], 
                "result": mesh_obj,
                "vertices_created": len(mesh_obj.mesh.vertices),
                "faces_created": len(mesh_obj.mesh.faces)
            }
        else:
            return {"processed": 0, "errors": [f"No plane mesh generated for {run.name}"]}
        
    except Exception as e:
        return {"processed": 0, "errors": [f"Error processing {run.name}: {e}"]}


def plane_from_picks_batch(
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
    padding: float = 1.2,
    create_multiple: bool = False,
    voxel_spacing: float = 1.0,
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Batch convert picks to plane meshes across multiple runs.
    
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
    padding : float, optional
        Padding factor for plane size. Default is 1.2.
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
        callback=_plane_from_picks_worker,
        root=root,
        runs=runs_to_process,
        workers=workers,
        task_desc="Converting picks to plane meshes",
        pick_object_name=pick_object_name,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        mesh_object_name=mesh_object_name,
        mesh_session_id=mesh_session_id,
        mesh_user_id=mesh_user_id,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        padding=padding,
        create_multiple=create_multiple,
        voxel_spacing=voxel_spacing,
    )
    
    return results