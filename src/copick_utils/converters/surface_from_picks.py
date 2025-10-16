from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger
from scipy.interpolate import Rbf, griddata
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA

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


def fit_2d_surface_to_points(
    points: np.ndarray,
    method: str = "delaunay",
    grid_resolution: int = 50,
) -> tm.Trimesh:
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

    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # RBF interpolation
    rbf = Rbf(projected_2d[:, 0], projected_2d[:, 1], heights, function="thin_plate")
    zi_grid = rbf(xi_grid, yi_grid)

    # Convert grid back to 3D coordinates
    grid_points_2d = np.column_stack([xi_grid.flatten(), yi_grid.flatten(), zi_grid.flatten()])
    grid_points_3d = pca.inverse_transform(grid_points_2d) + center

    # Create triangulation for the grid
    grid_points_3d.reshape((grid_resolution, grid_resolution, 3))
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
        method="linear",
        fill_value=np.mean(points[:, height_dim]),
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
    all_clusters: bool = True,
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
            e.g.
                - {'eps': 5.0, 'min_samples': 3} for DBSCAN
                - {'n_clusters': 3} for KMeans
        all_clusters: If True, use all clusters; if False, use only the largest cluster.
        individual_meshes: If True, create separate mesh objects for each surface.
        session_id_template: Template for individual mesh session IDs.

    Returns:
        Tuple of (CopickMesh object, stats dict) or None if creation failed.
        Stats dict contains 'vertices_created' and 'faces_created' totals.
    """
    if not validate_points(points, 3, "surface"):
        return None

    if clustering_params is None:
        clustering_params = {}

    # Define surface creation function
    def create_surface_from_points(cluster_points):
        return fit_2d_surface_to_points(cluster_points, surface_method, grid_resolution)

    # Handle clustering workflow with special surface logic
    if use_clustering:
        point_clusters = cluster(
            points,
            clustering_method,
            min_points_per_cluster=3,  # Surfaces need at least 3 points
            **clustering_params,
        )

        if not point_clusters:
            logger.warning("No valid clusters found")
            return None

        logger.info(f"Found {len(point_clusters)} clusters")

        if all_clusters and len(point_clusters) > 1:
            if individual_meshes:
                # Create separate mesh objects for each surface
                created_meshes = []
                total_vertices = 0
                total_faces = 0

                for i, cluster_points in enumerate(point_clusters):
                    try:
                        surface_mesh = create_surface_from_points(cluster_points)

                        # Generate session ID using template if provided
                        if session_id_template:
                            surface_session_id = session_id_template.format(
                                base_session_id=session_id,
                                instance_id=i,
                            )
                        else:
                            surface_session_id = f"{session_id}-{i:03d}"

                        copick_mesh = run.new_mesh(object_name, surface_session_id, user_id, exist_ok=True)
                        copick_mesh.mesh = surface_mesh
                        copick_mesh.store()
                        created_meshes.append(copick_mesh)
                        total_vertices += len(surface_mesh.vertices)
                        total_faces += len(surface_mesh.faces)
                        logger.info(
                            f"Created individual surface mesh {i} with {len(surface_mesh.vertices)} vertices",
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
                    surface_mesh = create_surface_from_points(cluster_points)
                    all_meshes.append(surface_mesh)

                # Combine all meshes
                combined_mesh = tm.util.concatenate(all_meshes)
        else:
            # Use largest cluster
            cluster_sizes = [len(cluster) for cluster in point_clusters]
            largest_cluster_idx = np.argmax(cluster_sizes)
            points_to_use = point_clusters[largest_cluster_idx]
            logger.info(f"Using largest cluster with {len(points_to_use)} points")

            combined_mesh = create_surface_from_points(points_to_use)
    else:
        # Use all points without clustering
        combined_mesh = create_surface_from_points(points)

    # Store mesh and return stats
    try:
        return store_mesh_with_stats(run, combined_mesh, object_name, session_id, user_id, "surface")
    except Exception as e:
        logger.critical(f"Error creating mesh: {e}")
        return None


# Create worker function using common infrastructure
_surface_from_picks_worker = create_batch_worker(surface_from_picks, "surface", min_points=3)


# Create batch converter using common infrastructure
surface_from_picks_batch = create_batch_converter(
    surface_from_picks,
    "Converting picks to surface meshes",
    "surface",
    min_points=3,
)

# Lazy batch converter for new architecture
surface_from_picks_lazy_batch = create_lazy_batch_converter(
    converter_func=surface_from_picks,
    task_description="Converting picks to surface meshes",
)
