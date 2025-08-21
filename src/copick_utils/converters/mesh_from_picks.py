from typing import TYPE_CHECKING, Optional, Dict, Any, List, Union
import numpy as np
import trimesh as tm
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN, KMeans

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickMesh, CopickRoot


def convex_hull_mesh(points: np.ndarray) -> tm.Trimesh:
    """
    Create a convex hull mesh from points.
    
    Args:
        points: Nx3 array of points
        
    Returns:
        Trimesh object representing the convex hull
    """
    if len(points) < 4:
        raise ValueError("Need at least 4 points to create a convex hull")
    
    hull = ConvexHull(points)
    return tm.Trimesh(vertices=points[hull.vertices], faces=hull.simplices)


def alpha_shape_mesh(points: np.ndarray, alpha: float) -> tm.Trimesh:
    """
    Create an alpha shape mesh from points.
    
    Args:
        points: Nx3 array of points
        alpha: Alpha parameter controlling the shape detail
        
    Returns:
        Trimesh object representing the alpha shape
    """
    try:
        # Use scipy's Delaunay triangulation for alpha shapes
        from scipy.spatial import Delaunay
        
        if len(points) < 4:
            raise ValueError("Need at least 4 points to create an alpha shape")
        
        # Create Delaunay triangulation
        tri = Delaunay(points)
        
        # Filter tetrahedra based on circumradius (alpha criterion)
        valid_faces = []
        for simplex in tri.simplices:
            # Get the four points of the tetrahedron
            tetra_points = points[simplex]
            
            # Calculate circumradius
            circumradius = calculate_circumradius_3d(tetra_points)
            
            if circumradius < 1.0 / alpha:
                # Add all four triangular faces of the tetrahedron
                faces = [
                    [simplex[0], simplex[1], simplex[2]],
                    [simplex[0], simplex[1], simplex[3]], 
                    [simplex[0], simplex[2], simplex[3]],
                    [simplex[1], simplex[2], simplex[3]]
                ]
                valid_faces.extend(faces)
        
        if not valid_faces:
            print("Warning: No valid faces found with given alpha, falling back to convex hull")
            return convex_hull_mesh(points)
        
        # Remove duplicate faces (faces that appear on boundary of alpha shape)
        face_counts = {}
        for face in valid_faces:
            face_tuple = tuple(sorted(face))
            face_counts[face_tuple] = face_counts.get(face_tuple, 0) + 1
        
        # Keep only faces that appear exactly once (boundary faces)
        boundary_faces = []
        for face_tuple, count in face_counts.items():
            if count == 1:
                boundary_faces.append(list(face_tuple))
        
        if not boundary_faces:
            print("Warning: No boundary faces found, falling back to convex hull")
            return convex_hull_mesh(points)
        
        return tm.Trimesh(vertices=points, faces=boundary_faces)
        
    except Exception as e:
        print(f"Warning: Alpha shape creation failed ({e}), falling back to convex hull")
        return convex_hull_mesh(points)


def calculate_circumradius_3d(tetra_points: np.ndarray) -> float:
    """
    Calculate circumradius of a tetrahedron.
    
    Args:
        tetra_points: 4x3 array of tetrahedron vertices
        
    Returns:
        Circumradius of the tetrahedron
    """
    # Use the formula for circumradius of a tetrahedron
    a, b, c, d = tetra_points
    
    # Create matrix for calculation
    matrix = np.array([
        [a[0] - d[0], a[1] - d[1], a[2] - d[2]],
        [b[0] - d[0], b[1] - d[1], b[2] - d[2]],
        [c[0] - d[0], c[1] - d[1], c[2] - d[2]]
    ])
    
    try:
        det = np.linalg.det(matrix)
        if abs(det) < 1e-10:
            return float('inf')  # Degenerate tetrahedron
        
        # Calculate edge lengths
        edge_lengths_sq = [
            np.sum((a - b)**2), np.sum((a - c)**2), np.sum((a - d)**2),
            np.sum((b - c)**2), np.sum((b - d)**2), np.sum((c - d)**2)
        ]
        
        # Volume of tetrahedron
        volume = abs(det) / 6.0
        
        # Circumradius formula
        numerator = np.sqrt(np.prod(edge_lengths_sq))
        denominator = 24.0 * volume
        
        if denominator == 0:
            return float('inf')
        
        return numerator / denominator
        
    except:
        return float('inf')


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
                if len(cluster_points) >= 4:  # Need at least 4 points for 3D mesh
                    clusters.append(cluster_points)
        
        return clusters
        
    elif method == "kmeans":
        n_clusters = kwargs.get("n_clusters", 1)
        
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clustering.fit_predict(points)
        
        clusters = []
        for i in range(n_clusters):
            cluster_points = points[labels == i]
            if len(cluster_points) >= 4:  # Need at least 4 points for 3D mesh
                clusters.append(cluster_points)
        
        return clusters
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def mesh_from_picks(
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    points: np.ndarray,
    mesh_type: str = "convex_hull",
    alpha: Optional[float] = None,
    use_clustering: bool = False,
    clustering_method: str = "dbscan",
    clustering_params: Optional[Dict[str, Any]] = None,
) -> Optional["CopickMesh"]:
    """
    Create mesh(es) from pick points.
    
    Args:
        run: Copick run object
        object_name: Name of the mesh object
        session_id: Session ID for the mesh
        user_id: User ID for the mesh
        points: Nx3 array of pick positions
        mesh_type: Type of mesh to create ('convex_hull', 'alpha_shape')
        alpha: Alpha parameter for alpha shapes (required if mesh_type='alpha_shape')
        use_clustering: Whether to cluster points first
        clustering_method: Clustering method ('dbscan', 'kmeans')
        clustering_params: Parameters for clustering
        
    Returns:
        CopickMesh object or None if creation failed
    """
    if len(points) < 4:
        print(f"Need at least 4 points to create a mesh, got {len(points)}")
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
        
        # For now, use the largest cluster
        # TODO: Could create separate meshes for each cluster
        cluster_sizes = [len(cluster) for cluster in point_clusters]
        largest_cluster_idx = np.argmax(cluster_sizes)
        points_to_use = point_clusters[largest_cluster_idx]
        
        print(f"Using largest cluster with {len(points_to_use)} points")
    else:
        points_to_use = points
    
    # Create mesh based on type
    try:
        if mesh_type == "convex_hull":
            mesh = convex_hull_mesh(points_to_use)
        elif mesh_type == "alpha_shape":
            if alpha is None:
                raise ValueError("Alpha parameter is required for alpha shapes")
            mesh = alpha_shape_mesh(points_to_use, alpha)
        else:
            raise ValueError(f"Unknown mesh type: {mesh_type}")
        
        # Create copick mesh
        copick_mesh = run.new_mesh(object_name, session_id, user_id, exist_ok=True)
        copick_mesh.mesh = mesh
        copick_mesh.store()
        
        print(f"Created {mesh_type} mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        return copick_mesh
        
    except Exception as e:
        print(f"Error creating mesh: {e}")
        return None


def _mesh_from_picks_worker(
    run: "CopickRun",
    pick_object_name: str,
    pick_user_id: str,
    pick_session_id: str,
    mesh_object_name: str,
    mesh_session_id: str,
    mesh_user_id: str,
    mesh_type: str,
    alpha: Optional[float],
    use_clustering: bool,
    clustering_method: str,
    clustering_params: Dict[str, Any],
    voxel_spacing: float,
    root: "CopickRoot",
) -> Dict[str, Any]:
    """Worker function for batch conversion of picks to meshes."""
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
        
        mesh_obj = mesh_from_picks(
            points=positions,
            mesh_type=mesh_type,
            alpha=alpha,
            use_clustering=use_clustering,
            clustering_method=clustering_method,
            clustering_params=clustering_params,
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
            return {"processed": 0, "errors": [f"No mesh generated for {run.name}"]}
        
    except Exception as e:
        return {"processed": 0, "errors": [f"Error processing {run.name}: {e}"]}


def mesh_from_picks_batch(
    root: "CopickRoot",
    pick_object_name: str,
    pick_user_id: str,
    pick_session_id: str,
    mesh_object_name: str,
    mesh_session_id: str,
    mesh_user_id: str,
    mesh_type: str = "convex_hull",
    alpha: Optional[float] = None,
    use_clustering: bool = False,
    clustering_method: str = "dbscan",
    clustering_params: Optional[Dict[str, Any]] = None,
    voxel_spacing: float = 1.0,
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Batch convert picks to meshes across multiple runs.
    
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
    mesh_type : str, optional
        Type of mesh ('convex_hull', 'alpha_shape'). Default is 'convex_hull'.
    alpha : float, optional
        Alpha parameter for alpha shapes.
    use_clustering : bool, optional
        Whether to cluster points first. Default is False.
    clustering_method : str, optional
        Clustering method ('dbscan', 'kmeans'). Default is 'dbscan'.
    clustering_params : dict, optional
        Parameters for clustering method.
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
        callback=_mesh_from_picks_worker,
        root=root,
        runs=runs_to_process,
        workers=workers,
        task_desc="Converting picks to meshes",
        pick_object_name=pick_object_name,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        mesh_object_name=mesh_object_name,
        mesh_session_id=mesh_session_id,
        mesh_user_id=mesh_user_id,
        mesh_type=mesh_type,
        alpha=alpha,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        voxel_spacing=voxel_spacing,
    )
    
    return results