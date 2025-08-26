from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger
from scipy.spatial import ConvexHull

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


def convex_hull_mesh(points: np.ndarray) -> tm.Trimesh:
    """Create a convex hull mesh from points.

    Args:
        points: Nx3 array of points.

    Returns:
        Trimesh object representing the convex hull.
    """
    if len(points) < 4:
        raise ValueError("Need at least 4 points to create a convex hull")

    hull = ConvexHull(points)
    return tm.Trimesh(vertices=points[hull.vertices], faces=hull.simplices)


def alpha_shape_mesh(points: np.ndarray, alpha: float) -> tm.Trimesh:
    """Create an alpha shape mesh from points.

    Args:
        points: Nx3 array of points.
        alpha: Alpha parameter controlling the shape detail.

    Returns:
        Trimesh object representing the alpha shape.
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
                    [simplex[1], simplex[2], simplex[3]],
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
    """Calculate circumradius of a tetrahedron.

    Args:
        tetra_points: 4x3 array of tetrahedron vertices.

    Returns:
        Circumradius of the tetrahedron.
    """
    # Use the formula for circumradius of a tetrahedron
    a, b, c, d = tetra_points

    # Create matrix for calculation
    matrix = np.array(
        [
            [a[0] - d[0], a[1] - d[1], a[2] - d[2]],
            [b[0] - d[0], b[1] - d[1], b[2] - d[2]],
            [c[0] - d[0], c[1] - d[1], c[2] - d[2]],
        ],
    )

    try:
        det = np.linalg.det(matrix)
        if abs(det) < 1e-10:
            return float("inf")  # Degenerate tetrahedron

        # Calculate edge lengths
        edge_lengths_sq = [
            np.sum((a - b) ** 2),
            np.sum((a - c) ** 2),
            np.sum((a - d) ** 2),
            np.sum((b - c) ** 2),
            np.sum((b - d) ** 2),
            np.sum((c - d) ** 2),
        ]

        # Volume of tetrahedron
        volume = abs(det) / 6.0

        # Circumradius formula
        numerator = np.sqrt(np.prod(edge_lengths_sq))
        denominator = 24.0 * volume

        if denominator == 0:
            return float("inf")

        return numerator / denominator

    except Exception:
        return float("inf")


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

    # Define mesh creation function for clustering workflow
    def create_mesh_from_points(cluster_points):
        if mesh_type == "convex_hull":
            return convex_hull_mesh(cluster_points)
        elif mesh_type == "alpha_shape":
            if alpha is None:
                raise ValueError("Alpha parameter is required for alpha shapes")
            return alpha_shape_mesh(cluster_points, alpha)
        else:
            raise ValueError(f"Unknown mesh type: {mesh_type}")

    # Handle clustering workflow with special mesh logic for individual meshes
    if use_clustering and individual_meshes and all_clusters:
        from copick_utils.converters.converter_common import cluster

        point_clusters = cluster(
            points,
            clustering_method,
            min_points_per_cluster=4,
            **clustering_params,
        )

        if not point_clusters:
            logger.warning("No valid clusters found")
            return None

        logger.info(f"Found {len(point_clusters)} clusters")

        # Create separate mesh objects for each cluster
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
                        mesh_id=i,
                    )
                else:
                    mesh_session_id = f"{session_id}-{i:03d}"

                try:
                    copick_mesh = run.new_mesh(object_name, mesh_session_id, user_id, exist_ok=True)
                    copick_mesh.mesh = cluster_mesh
                    copick_mesh.store()
                    created_meshes.append(copick_mesh)
                    total_vertices += len(cluster_mesh.vertices)
                    total_faces += len(cluster_mesh.faces)
                    logger.info(f"Created individual mesh {i} with {len(cluster_mesh.vertices)} vertices")
                except Exception as e:
                    logger.error(f"Failed to create mesh {i}: {e}")
                    continue
            except Exception as e:
                logger.critical(f"Failed to create mesh from cluster {i}: {e}")
                continue

        # Return the first mesh and total stats
        if created_meshes:
            stats = {"vertices_created": total_vertices, "faces_created": total_faces}
            return created_meshes[0], stats
        else:
            return None
    else:
        # Use standard clustering workflow
        combined_mesh, points_used = handle_clustering_workflow(
            points=points,
            use_clustering=use_clustering,
            clustering_method=clustering_method,
            clustering_params=clustering_params,
            all_clusters=all_clusters,
            min_points_per_cluster=4,
            shape_creation_func=create_mesh_from_points,
            shape_name="mesh",
        )

        if combined_mesh is None:
            return None

        # Store mesh and return stats
        try:
            return store_mesh_with_stats(run, combined_mesh, object_name, session_id, user_id, "mesh")
        except Exception as e:
            logger.critical(f"Error creating mesh: {e}")
            return None


# Create worker function using common infrastructure
_mesh_from_picks_worker = create_batch_worker(mesh_from_picks, "mesh", min_points=4)


# Create batch converter using common infrastructure
mesh_from_picks_batch = create_batch_converter(mesh_from_picks, "Converting picks to meshes", "mesh", min_points=4)
