"""Generate valid area box meshes for tomographic reconstructions."""
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import trimesh as tm
import zarr

if TYPE_CHECKING:
    from copick.models import CopickRoot, CopickRun


def shift_3d(shift: np.ndarray) -> np.ndarray:
    """
    Create a 3D translation transformation matrix.

    Args:
        shift: Translation vector [x, y, z]

    Returns:
        4x4 homogeneous transformation matrix
    """
    return np.array(
        [
            [1, 0, 0, shift[0]],
            [0, 1, 0, shift[1]],
            [0, 0, 1, shift[2]],
            [0, 0, 0, 1],
        ],
    )


def rotation_3d_x(angle: float) -> np.ndarray:
    """
    Create a 3D rotation transformation matrix around X-axis.

    Args:
        angle: Rotation angle in degrees

    Returns:
        4x4 homogeneous transformation matrix
    """
    phi = np.radians(angle)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
    )


def rotation_3d_y(angle: float) -> np.ndarray:
    """
    Create a 3D rotation transformation matrix around Y-axis.

    Args:
        angle: Rotation angle in degrees

    Returns:
        4x4 homogeneous transformation matrix
    """
    phi = np.radians(angle)
    return np.array(
        [
            [np.cos(phi), 0, np.sin(phi), 0],
            [0, 1, 0, 0],
            [-np.sin(phi), 0, np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
    )


def rotation_3d_z(angle: float) -> np.ndarray:
    """
    Create a 3D rotation transformation matrix around Z-axis.

    Args:
        angle: Rotation angle in degrees

    Returns:
        4x4 homogeneous transformation matrix
    """
    phi = np.radians(angle)
    return np.array(
        [
            [np.cos(phi), -np.sin(phi), 0, 0],
            [np.sin(phi), np.cos(phi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )


def rotation_center(rot_matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Create a rotation transformation around a center point.

    Args:
        rot_matrix: 4x4 rotation matrix
        center: Center point [x, y, z]

    Returns:
        4x4 transformation matrix for rotation around center
    """
    s1 = shift_3d(-center)
    s2 = shift_3d(center)
    return s2 @ rot_matrix @ s1


def create_validbox_mesh(
    run: "CopickRun",
    voxel_spacing: float,
    tomo_type: str = "wbp",
    angle: float = 0.0,
) -> Optional[tm.Trimesh]:
    """
    Create a box mesh representing the valid area of a reconstruction.

    Args:
        run: Copick run object
        voxel_spacing: Voxel spacing for the tomogram
        tomo_type: Type of tomogram to use as reference
        angle: Rotation angle around Z-axis in degrees

    Returns:
        Trimesh box object or None if tomogram not found
    """
    # Negate angle to match coordinate system conventions
    angle = -angle

    # Get tomogram dimensions
    vs = run.get_voxel_spacing(voxel_spacing)
    tomo = vs.get_tomograms(tomo_type)[0]

    if tomo is None:
        print(f"Warning: Could not find tomogram of type '{tomo_type}' for run {run.name}")
        return None

    # Get pixel dimensions and calculate physical dimensions
    pixel_max_dim = zarr.open(tomo.zarr())["0"].shape[::-1]
    pixel_center = np.floor(np.array(pixel_max_dim) / 2) + 1
    max_dim = np.array([d * voxel_spacing for d in pixel_max_dim])
    center = np.array([c * voxel_spacing for c in pixel_center])

    # Create rotation transformation
    r = rotation_3d_z(angle)
    transform = rotation_center(r, center)

    # Create rotated box mesh with original tomogram dimensions
    box = tm.creation.box(
        extents=max_dim,
        transform=transform @ shift_3d(max_dim / 2),
    )

    # Define the tomogram bounding planes
    # The tomogram extends from (0, 0, 0) to max_dim
    bounding_planes = [
        # X planes: normal points inward (positive X direction for min plane, negative for max)
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),  # x >= 0
        (np.array([-1.0, 0.0, 0.0]), np.array([max_dim[0], 0.0, 0.0])),  # x <= max_dim[0]
        # Y planes
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])),  # y >= 0
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, max_dim[1], 0.0])),  # y <= max_dim[1]
        # Z planes
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])),  # z >= 0
        (np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, max_dim[2]])),  # z <= max_dim[2]
    ]

    # Start with the rotated box mesh
    current_mesh = box

    # Slice the rotated box with each bounding plane to clip it to tomogram bounds
    for plane_normal, plane_origin in bounding_planes:
        # Use trimesh's slice_plane method which properly caps the mesh
        current_mesh = current_mesh.slice_plane(
            plane_origin=plane_origin,
            plane_normal=plane_normal,
            cap=True,  # This caps the mesh where it was sliced
        )

    # Clean up the mesh to remove degenerate faces and unused vertices
    current_mesh.remove_unreferenced_vertices()
    current_mesh.fix_normals()

    return current_mesh


def generate_validbox(
    run: "CopickRun",
    voxel_spacing: float,
    mesh_object_name: str,
    mesh_user_id: str,
    mesh_session_id: str,
    tomo_type: str = "wbp",
    angle: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    Generate a valid area box mesh for a single run.

    Args:
        run: Copick run object
        voxel_spacing: Voxel spacing for the tomogram
        mesh_object_name: Name of the mesh object to create
        mesh_user_id: User ID for the mesh
        mesh_session_id: Session ID for the mesh
        tomo_type: Type of tomogram to use as reference
        angle: Rotation angle around Z-axis in degrees

    Returns:
        Dictionary with result information or None if failed
    """
    try:
        # Create the box mesh
        box = create_validbox_mesh(
            run=run,
            voxel_spacing=voxel_spacing,
            tomo_type=tomo_type,
            angle=angle,
        )

        if box is None:
            return {
                "processed": 0,
                "errors": [f"Could not create validbox for run {run.name}"],
                "vertices_created": 0,
                "faces_created": 0,
            }

        # Get or create mesh object
        existing_meshes = run.get_meshes(
            object_name=mesh_object_name,
            user_id=mesh_user_id,
            session_id=mesh_session_id,
        )

        if len(existing_meshes) == 0:
            mesh_obj = run.new_mesh(
                object_name=mesh_object_name,
                user_id=mesh_user_id,
                session_id=mesh_session_id,
            )
        else:
            mesh_obj = existing_meshes[0]

        # Store the mesh
        mesh_obj.mesh = box
        mesh_obj.store()

        return {
            "processed": 1,
            "errors": [],
            "vertices_created": len(box.vertices),
            "faces_created": len(box.faces),
        }

    except Exception as e:
        return {
            "processed": 0,
            "errors": [f"Error processing {run.name}: {e}"],
            "vertices_created": 0,
            "faces_created": 0,
        }


def _validbox_worker(
    run: "CopickRun",
    voxel_spacing: float,
    mesh_object_name: str,
    mesh_user_id: str,
    mesh_session_id: str,
    tomo_type: str,
    angle: float,
) -> Dict[str, Any]:
    """Worker function for batch validbox generation."""
    return generate_validbox(
        run=run,
        voxel_spacing=voxel_spacing,
        mesh_object_name=mesh_object_name,
        mesh_user_id=mesh_user_id,
        mesh_session_id=mesh_session_id,
        tomo_type=tomo_type,
        angle=angle,
    )


def validbox_batch(
    root: "CopickRoot",
    voxel_spacing: float,
    mesh_object_name: str,
    mesh_user_id: str,
    mesh_session_id: str,
    tomo_type: str = "wbp",
    angle: float = 0.0,
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Generate valid area box meshes across multiple runs.

    Args:
        root: The copick root containing runs to process
        voxel_spacing: Voxel spacing for the tomograms
        mesh_object_name: Name of the mesh object to create
        mesh_user_id: User ID for the meshes
        mesh_session_id: Session ID for the meshes
        tomo_type: Type of tomogram to use as reference. Default is 'wbp'.
        angle: Rotation angle around Z-axis in degrees. Default is 0.0.
        run_names: List of run names to process. If None, processes all runs.
        workers: Number of worker processes. Default is 8.

    Returns:
        Dictionary with processing results and statistics
    """
    from copick.ops.run import map_runs

    runs_to_process = [run.name for run in root.runs] if run_names is None else run_names

    results = map_runs(
        callback=_validbox_worker,
        root=root,
        runs=runs_to_process,
        workers=workers,
        task_desc="Generating validbox meshes",
        voxel_spacing=voxel_spacing,
        mesh_object_name=mesh_object_name,
        mesh_user_id=mesh_user_id,
        mesh_session_id=mesh_session_id,
        tomo_type=tomo_type,
        angle=angle,
    )

    return results
