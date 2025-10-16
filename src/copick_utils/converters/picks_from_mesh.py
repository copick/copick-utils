from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger
from scipy.stats.qmc import PoissonDisk

from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickPicks, CopickRoot, CopickRun

logger = get_logger(__name__)


def ensure_mesh(trimesh_object):
    """
    Ensure that the input is a valid Trimesh object.

    Args:
        trimesh_object: Trimesh or Scene object

    Returns:
        Trimesh object or None if no geometry found
    """
    if isinstance(trimesh_object, tm.Scene):
        if len(trimesh_object.geometry) == 0:
            return None
        else:
            return tm.util.concatenate(list(trimesh_object.geometry.values()))
    elif isinstance(trimesh_object, tm.Trimesh):
        return trimesh_object
    else:
        raise ValueError("Input must be a Trimesh or Scene object")


def poisson_disk_in_out(
    n_in: int,
    n_out: int,
    mesh: tm.Trimesh,
    max_dim: Sequence[float],
    min_dist: float,
    edge_dist: float,
    input_points: np.ndarray,
    seed: int = 1234,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Poisson disk sampled points inside and outside the mesh.

    Args:
        n_in: Number of points to sample inside the mesh
        n_out: Number of points to sample outside the mesh
        mesh: Trimesh object
        max_dim: Maximum dimensions of the volume
        min_dist: Minimum distance between points
        edge_dist: Distance from volume edges
        input_points: Existing points to avoid
        seed: Random seed

    Returns:
        Tuple of (points_in, points_out) arrays
    """
    max_max = np.max(max_dim)
    min_dist = min_dist / max_max

    engine = PoissonDisk(d=3, radius=min_dist, seed=seed)

    # Fill space
    points = engine.fill_space() * max_max

    # Reject points outside the volume
    lb = np.array([edge_dist, edge_dist, edge_dist])
    ub = max_dim - np.array([edge_dist, edge_dist, edge_dist])
    points = points[np.all(np.logical_and(points > lb, points < ub), axis=1), :]

    # Reject points that are too close to the input points
    for pt in input_points:
        dist = np.linalg.norm(points - pt, axis=1)
        points = points[dist > min_dist]

    # Check if points are inside/outside the mesh
    mask = mesh.contains(points)
    inv_mask = np.logical_not(mask)

    points_in = points[mask, :]
    points_out = points[inv_mask, :]

    # Shuffle output
    np.random.default_rng(seed).shuffle(points_in)
    np.random.default_rng(seed).shuffle(points_out)

    # Limit number of points to n_in and n_out
    if n_in > points_in.shape[0]:
        print(f"Warning: Not enough points inside the mesh. Requested {n_in}, found {points_in.shape[0]}")
    n_in = min(n_in, points_in.shape[0])
    final_points_in = points_in[:n_in, :]

    if n_out > points_out.shape[0]:
        print(f"Warning: Not enough points outside the mesh. Requested {n_out}, found {points_out.shape[0]}")
    n_out = min(n_out, points_out.shape[0])
    final_points_out = points_out[:n_out, :]

    return final_points_in, final_points_out


def generate_random_orientations(n_points: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random orientations as 4x4 transformation matrices.

    Args:
        n_points: Number of transformation matrices to generate
        seed: Random seed

    Returns:
        Array of shape (n_points, 4, 4) containing transformation matrices
    """
    if seed is not None:
        np.random.seed(seed)

    transforms = np.zeros((n_points, 4, 4))

    for i in range(n_points):
        # Generate random rotation matrix using quaternions
        # Generate random quaternion (uniform distribution on unit sphere)
        u1, u2, u3 = np.random.random(3)
        q = np.array(
            [
                np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                np.sqrt(u1) * np.sin(2 * np.pi * u3),
                np.sqrt(u1) * np.cos(2 * np.pi * u3),
            ],
        )

        # Convert quaternion to rotation matrix
        qx, qy, qz, qw = q
        rotation_matrix = np.array(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
            ],
        )

        transforms[i, :3, :3] = rotation_matrix
        transforms[i, 3, 3] = 1.0

    return transforms


def picks_from_mesh(
    mesh: tm.Trimesh,
    sampling_type: str,
    n_points: int,
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    voxel_spacing: float,
    tomo_type: str = "wbp",
    min_dist: Optional[float] = None,
    edge_dist: float = 32.0,
    include_normals: bool = False,
    random_orientations: bool = False,
    seed: Optional[int] = None,
) -> Optional["CopickPicks"]:
    """
    Sample points from a mesh using different strategies.

    Args:
        mesh: Trimesh object to sample from
        sampling_type: Type of sampling ('inside', 'surface', 'outside', 'vertices')
        n_points: Number of points to sample (ignored for 'vertices')
        run: Copick run object
        object_name: Name of the object for the picks
        session_id: Session ID for the picks
        user_id: User ID for the picks
        voxel_spacing: Voxel spacing for coordinate scaling
        tomo_type: Tomogram type for getting volume dimensions
        min_dist: Minimum distance between points (if None, uses voxel_spacing)
        edge_dist: Distance from volume edges in voxels
        include_normals: Include surface normals as orientations (surface sampling only)
        random_orientations: Generate random orientations for points
        seed: Random seed for reproducible results

    Returns:
        CopickPicks object or None if sampling failed
    """
    if not mesh.is_watertight and sampling_type in ["inside", "outside"]:
        print(f"Warning: Mesh is not watertight, {sampling_type} sampling may be unreliable")

    # Get tomogram dimensions
    vs = run.get_voxel_spacing(voxel_spacing)
    tomo = vs.get_tomogram(tomo_type)

    if tomo is None:
        print(f"Warning: Could not find tomogram of type '{tomo_type}' for run {run.name}")
        return None

    import zarr

    pixel_max_dim = zarr.open(tomo.zarr())["0"].shape[::-1]
    max_dim = np.array([d * voxel_spacing for d in pixel_max_dim])

    # Set default min_dist if not provided
    if min_dist is None:
        min_dist = voxel_spacing * 2

    edge_dist_physical = edge_dist * voxel_spacing

    if seed is not None:
        np.random.seed(seed)

    points = None
    orientations = None

    if sampling_type == "vertices":
        # Return mesh vertices directly
        points = mesh.vertices.copy()

    elif sampling_type == "surface":
        # Sample points on mesh surface
        points, face_indices = tm.sample.sample_surface_even(mesh, n_points, radius=min_dist, seed=seed)

        if include_normals:
            # Get face normals for the sampled points
            face_normals = mesh.face_normals[face_indices]
            # Convert normals to transformation matrices
            orientations = np.zeros((len(points), 4, 4))
            for i, normal in enumerate(face_normals):
                # Create rotation matrix that aligns z-axis with normal
                z_axis = np.array([0, 0, 1])
                if np.allclose(normal, z_axis):
                    rot_matrix = np.eye(3)
                elif np.allclose(normal, -z_axis):
                    rot_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
                else:
                    v = np.cross(z_axis, normal)
                    s = np.linalg.norm(v)
                    c = np.dot(z_axis, normal)
                    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    rot_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))

                orientations[i, :3, :3] = rot_matrix
                orientations[i, 3, 3] = 1.0

    elif sampling_type in ["inside", "outside"]:
        # Use Poisson disk sampling
        if sampling_type == "inside":
            points_in, _ = poisson_disk_in_out(
                n_points,
                0,
                mesh,
                max_dim,
                min_dist,
                edge_dist_physical,
                np.array([]),
                seed,
            )
            points = points_in
        else:  # outside
            _, points_out = poisson_disk_in_out(
                0,
                n_points,
                mesh,
                max_dim,
                min_dist,
                edge_dist_physical,
                np.array([]),
                seed,
            )
            points = points_out

    else:
        raise ValueError(
            f"Invalid sampling_type: {sampling_type}. Must be 'inside', 'surface', 'outside', or 'vertices'",
        )

    if points is None or len(points) == 0:
        print(f"No points generated for {sampling_type} sampling")
        return None

    # Filter points that are too close to edges
    valid_mask = np.all(
        [
            points[:, 0] >= edge_dist_physical,
            points[:, 0] <= max_dim[0] - edge_dist_physical,
            points[:, 1] >= edge_dist_physical,
            points[:, 1] <= max_dim[1] - edge_dist_physical,
            points[:, 2] >= edge_dist_physical,
            points[:, 2] <= max_dim[2] - edge_dist_physical,
        ],
        axis=0,
    )

    points = points[valid_mask]
    if orientations is not None:
        orientations = orientations[valid_mask]

    if len(points) == 0:
        print("No valid points after edge filtering")
        return None

    # Generate random orientations if requested and not already set
    if random_orientations and orientations is None:
        orientations = generate_random_orientations(len(points), seed)

    # Create picks
    pick_set = run.new_picks(object_name, session_id, user_id, exist_ok=True)
    pick_set.from_numpy(positions=points, transforms=orientations)
    pick_set.store()

    print(f"Created {len(points)} picks using {sampling_type} sampling")
    return pick_set


def _picks_from_mesh_worker(
    run: "CopickRun",
    mesh_object_name: str,
    mesh_user_id: str,
    mesh_session_id: str,
    sampling_type: str,
    n_points: int,
    pick_object_name: str,
    pick_session_id: str,
    pick_user_id: str,
    voxel_spacing: float,
    tomo_type: str,
    min_dist: Optional[float],
    edge_dist: float,
    include_normals: bool,
    random_orientations: bool,
    seed: Optional[int],
) -> Dict[str, Any]:
    """Worker function for batch conversion of meshes to picks."""
    try:
        # Get mesh
        meshes = run.get_meshes(object_name=mesh_object_name, user_id=mesh_user_id, session_id=mesh_session_id)

        if not meshes:
            return {"processed": 0, "errors": [f"No meshes found for {run.name}"]}

        mesh_obj = meshes[0]
        mesh = mesh_obj.mesh
        mesh = ensure_mesh(mesh)

        if mesh is None:
            return {"processed": 0, "errors": [f"Could not load mesh data for {run.name}"]}

        pick_set = picks_from_mesh(
            mesh=mesh,
            sampling_type=sampling_type,
            n_points=n_points,
            run=run,
            object_name=pick_object_name,
            session_id=pick_session_id,
            user_id=pick_user_id,
            voxel_spacing=voxel_spacing,
            tomo_type=tomo_type,
            min_dist=min_dist,
            edge_dist=edge_dist,
            include_normals=include_normals,
            random_orientations=random_orientations,
            seed=seed,
        )

        if pick_set and pick_set.points:
            return {"processed": 1, "errors": [], "result": pick_set, "points_created": len(pick_set.points)}
        else:
            return {"processed": 0, "errors": [f"No picks generated for {run.name}"]}

    except Exception as e:
        return {"processed": 0, "errors": [f"Error processing {run.name}: {e}"]}


def picks_from_mesh_standard(
    mesh: "CopickMesh",
    run: "CopickRun",
    output_object_name: str,
    output_session_id: str,
    output_user_id: str,
    sampling_type: str,
    n_points: int,
    voxel_spacing: float,
    tomo_type: str = "wbp",
    min_dist: Optional[float] = None,
    edge_dist: float = 32.0,
    include_normals: bool = False,
    random_orientations: bool = False,
    seed: Optional[int] = None,
    **kwargs,
) -> Optional[Tuple["CopickPicks", Dict[str, int]]]:
    """
    Standard signature wrapper for picks_from_mesh to match converter pattern.

    Args:
        mesh: CopickMesh object to sample from
        run: Copick run object
        output_object_name: Name for the output picks
        output_session_id: Session ID for the output picks
        output_user_id: User ID for the output picks
        sampling_type: Type of sampling ('inside', 'surface', 'outside', 'vertices')
        n_points: Number of points to sample (ignored for 'vertices')
        voxel_spacing: Voxel spacing for coordinate scaling
        tomo_type: Tomogram type for getting volume dimensions
        min_dist: Minimum distance between points
        edge_dist: Distance from volume edges in voxels
        include_normals: Include surface normals as orientations
        random_orientations: Generate random orientations for points
        seed: Random seed for reproducible results
        **kwargs: Additional arguments (ignored)

    Returns:
        Tuple of (CopickPicks object, stats dict) or None if conversion failed
    """
    try:
        # Get the trimesh object
        trimesh_obj = mesh.mesh
        if trimesh_obj is None:
            logger.error("Could not load mesh data")
            return None

        # Handle Scene objects
        if isinstance(trimesh_obj, tm.Scene):
            if len(trimesh_obj.geometry) == 0:
                logger.error("Mesh is empty")
                return None
            trimesh_obj = tm.util.concatenate(list(trimesh_obj.geometry.values()))

        # Call the original picks_from_mesh function
        result_picks = picks_from_mesh(
            mesh=trimesh_obj,
            sampling_type=sampling_type,
            n_points=n_points,
            run=run,
            object_name=output_object_name,
            session_id=output_session_id,
            user_id=output_user_id,
            voxel_spacing=voxel_spacing,
            tomo_type=tomo_type,
            min_dist=min_dist,
            edge_dist=edge_dist,
            include_normals=include_normals,
            random_orientations=random_orientations,
            seed=seed,
        )

        if result_picks is None:
            return None

        # Get point count for stats
        points, _ = result_picks.numpy()
        stats = {"points_created": len(points) if points is not None else 0}

        return result_picks, stats

    except Exception as e:
        logger.error(f"Error converting mesh to picks: {e}")
        return None


def picks_from_mesh_batch(
    root: "CopickRoot",
    mesh_object_name: str,
    mesh_user_id: str,
    mesh_session_id: str,
    sampling_type: str,
    n_points: int,
    pick_object_name: str,
    pick_session_id: str,
    pick_user_id: str,
    voxel_spacing: float,
    tomo_type: str = "wbp",
    min_dist: Optional[float] = None,
    edge_dist: float = 32.0,
    include_normals: bool = False,
    random_orientations: bool = False,
    seed: Optional[int] = None,
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Batch convert meshes to picks across multiple runs.

    Args:
        root: The copick root containing runs to process.
        mesh_object_name: Name of the mesh object to sample from.
        mesh_user_id: User ID of the mesh to convert.
        mesh_session_id: Session ID of the mesh to convert.
        sampling_type: Type of sampling ('inside', 'surface', 'outside', 'vertices').
        n_points: Number of points to sample (ignored for 'vertices' type).
        pick_object_name: Name of the object for created picks.
        pick_session_id: Session ID for created picks.
        pick_user_id: User ID for created picks.
        voxel_spacing: Voxel spacing for coordinate scaling.
        tomo_type: Tomogram type for getting volume dimensions. Default is 'wbp'.
        min_dist: Minimum distance between points. If None, uses 2 * voxel_spacing.
        edge_dist: Distance from volume edges in voxels. Default is 32.0.
        include_normals: Include surface normals as orientations (surface sampling only). Default is False.
        random_orientations: Generate random orientations for points. Default is False.
        seed: Random seed for reproducible results.
        run_names: List of run names to process. If None, processes all runs.
        workers: Number of worker processes. Default is 8.

    Returns:
        Dictionary with processing results and statistics.
    """
    from copick.ops.run import map_runs

    runs_to_process = [run.name for run in root.runs] if run_names is None else run_names

    results = map_runs(
        callback=_picks_from_mesh_worker,
        root=root,
        runs=runs_to_process,
        workers=workers,
        task_desc="Converting meshes to picks",
        mesh_object_name=mesh_object_name,
        mesh_user_id=mesh_user_id,
        mesh_session_id=mesh_session_id,
        sampling_type=sampling_type,
        n_points=n_points,
        pick_object_name=pick_object_name,
        pick_session_id=pick_session_id,
        pick_user_id=pick_user_id,
        voxel_spacing=voxel_spacing,
        tomo_type=tomo_type,
        min_dist=min_dist,
        edge_dist=edge_dist,
        include_normals=include_normals,
        random_orientations=random_orientations,
        seed=seed,
    )

    return results


# Lazy batch converter for new architecture
picks_from_mesh_lazy_batch = create_lazy_batch_converter(
    converter_func=picks_from_mesh_standard,
    task_description="Converting meshes to picks",
)
