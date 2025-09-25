"""Convert meshes to segmentation volumes."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger
from trimesh.ray.ray_triangle import RayMeshIntersector

from copick_utils.converters.converter_common import (
    create_batch_converter,
    create_batch_worker,
)
from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickRun, CopickSegmentation

logger = get_logger(__name__)


def ensure_mesh(trimesh_object) -> Optional[tm.Trimesh]:
    """
    Convert a trimesh object to a single mesh.

    Args:
        trimesh_object: A Trimesh or Scene object

    Returns:
        Single Trimesh object or None if empty

    Raises:
        ValueError: If input is not a Trimesh or Scene object
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


def _onesmask_z(mesh: tm.Trimesh, voxel_dims: Tuple[int, int, int], voxel_spacing: float) -> np.ndarray:
    """Create mask by ray casting in Z direction."""
    intersector = RayMeshIntersector(mesh)

    # Create a grid of rays in XY plane, shooting in Z direction
    grid_x, grid_y = np.mgrid[0 : voxel_dims[0], 0 : voxel_dims[1]]
    ray_grid = np.vstack([grid_x.ravel(), grid_y.ravel(), -np.ones((grid_x.size,))]).T * voxel_spacing
    ray_dir = np.zeros((ray_grid.shape[0], 3))
    ray_dir[:, 2] = 1

    loc, _, _ = intersector.intersects_location(ray_grid, ray_dir)

    # Convert to voxel coordinates and sort by z
    int_loc = np.round(loc / voxel_spacing).astype("int")
    sort_idx = int_loc[:, 2].argsort()
    int_loc = int_loc[sort_idx, :]

    # Build volume by tracking crossings
    img = np.zeros((voxel_dims[1], voxel_dims[0]), dtype="bool")
    vol = np.zeros((voxel_dims[2], voxel_dims[1], voxel_dims[0]), dtype="bool")

    for z in range(voxel_dims[2]):
        idx = int_loc[:, 2] == z
        img[int_loc[idx, 1], int_loc[idx, 0]] = np.logical_not(img[int_loc[idx, 1], int_loc[idx, 0]])
        vol[z, :, :] = img

    return vol


def _onesmask_x(mesh: tm.Trimesh, voxel_dims: Tuple[int, int, int], voxel_spacing: float) -> np.ndarray:
    """Create mask by ray casting in X direction."""
    intersector = RayMeshIntersector(mesh)

    # Create a grid of rays in YZ plane, shooting in X direction
    grid_y, grid_z = np.mgrid[0 : voxel_dims[1], 0 : voxel_dims[2]]
    ray_grid = np.vstack([-np.ones((grid_y.size,)), grid_y.ravel(), grid_z.ravel()]).T * voxel_spacing
    ray_dir = np.zeros((ray_grid.shape[0], 3))
    ray_dir[:, 0] = 1

    loc, _, _ = intersector.intersects_location(ray_grid, ray_dir)

    # Convert to voxel coordinates and sort by x
    int_loc = np.round(loc / voxel_spacing).astype("int")
    sort_idx = int_loc[:, 0].argsort()
    int_loc = int_loc[sort_idx, :]

    # Build volume by tracking crossings
    img = np.zeros((voxel_dims[2], voxel_dims[1]), dtype="bool")
    vol = np.zeros((voxel_dims[2], voxel_dims[1], voxel_dims[0]), dtype="bool")

    for x in range(voxel_dims[0]):
        idx = int_loc[:, 0] == x
        img[int_loc[idx, 2], int_loc[idx, 1]] = np.logical_not(img[int_loc[idx, 2], int_loc[idx, 1]])
        vol[:, :, x] = img

    return vol


def mesh_to_volume(mesh: tm.Trimesh, voxel_dims: Tuple[int, int, int], voxel_spacing: float) -> np.ndarray:
    """
    Convert a watertight mesh to a binary volume using ray casting.

    Args:
        mesh: Trimesh object representing the mesh
        voxel_dims: Dimensions of the output volume (x, y, z)
        voxel_spacing: Spacing between voxels in physical units

    Returns:
        Binary volume as numpy array with shape (z, y, x)
    """
    vols = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futs = [
            executor.submit(_onesmask_x, mesh.copy(), voxel_dims, voxel_spacing),
            executor.submit(_onesmask_z, mesh.copy(), voxel_dims, voxel_spacing),
        ]

        for f in as_completed(futs):
            vols.append(f.result())

    return np.logical_and(vols[0], vols[1])


def mesh_to_boundary_volume(
    mesh: tm.Trimesh,
    voxel_dims: Tuple[int, int, int],
    voxel_spacing: float,
    sampling_density: float = 1.0,
) -> np.ndarray:
    """
    Convert a mesh to a binary volume by voxelizing only the surface/boundary.

    Args:
        mesh: Trimesh object representing the mesh
        voxel_dims: Dimensions of the output volume (x, y, z)
        voxel_spacing: Spacing between voxels in physical units
        sampling_density: Density of surface sampling (samples per voxel edge length)

    Returns:
        Binary volume as numpy array with shape (z, y, x)
    """
    # Sample points on the mesh surface
    # Calculate number of points based on surface area and sampling density
    surface_area = mesh.area
    # Estimate points per unit area based on voxel spacing and density
    points_per_area = (sampling_density / voxel_spacing) ** 2
    n_points = max(int(surface_area * points_per_area), 1000)  # Minimum 1000 points

    # Sample points uniformly on the surface
    surface_points, _ = tm.sample.sample_surface(mesh, n_points)

    # Convert surface points to voxel coordinates
    voxel_coords = np.round(surface_points / voxel_spacing).astype(int)

    # Create binary volume
    vol = np.zeros((voxel_dims[2], voxel_dims[1], voxel_dims[0]), dtype=bool)

    # Filter coordinates to be within bounds
    valid_mask = (
        (voxel_coords[:, 0] >= 0)
        & (voxel_coords[:, 0] < voxel_dims[0])
        & (voxel_coords[:, 1] >= 0)
        & (voxel_coords[:, 1] < voxel_dims[1])
        & (voxel_coords[:, 2] >= 0)
        & (voxel_coords[:, 2] < voxel_dims[2])
    )
    voxel_coords = voxel_coords[valid_mask]

    # Set voxels at surface points
    if len(voxel_coords) > 0:
        # Convert from (x,y,z) to (z,y,x) indexing for the volume
        vol[voxel_coords[:, 2], voxel_coords[:, 1], voxel_coords[:, 0]] = True

    return vol


def segmentation_from_mesh(
    mesh: "CopickMesh",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    voxel_spacing: float,
    tomo_type: str = "wbp",
    is_multilabel: bool = False,
    mode: str = "watertight",
    boundary_sampling_density: float = 1.0,
    invert: bool = False,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Convert a CopickMesh to a segmentation volume.

    Args:
        mesh: CopickMesh object to convert
        run: CopickRun object
        object_name: Name for the output segmentation
        session_id: Session ID for the output segmentation
        user_id: User ID for the output segmentation
        voxel_spacing: Voxel spacing for the segmentation
        tomo_type: Type of tomogram to use for reference dimensions
        is_multilabel: Whether the segmentation is multilabel
        mode: Voxelization mode ('watertight' or 'boundary')
        boundary_sampling_density: Surface sampling density for boundary mode (samples per voxel edge length)
        invert: Whether to invert the volume (fill outside instead of inside)

    Returns:
        Tuple of (CopickSegmentation object, stats dict) or None if creation failed.
        Stats dict contains 'voxels_created'.
    """
    try:
        # Get the trimesh object from CopickMesh
        mesh_obj = ensure_mesh(mesh.mesh)
        if mesh_obj is None:
            logger.error("Empty mesh")
            return None

        # Get reference dimensions from tomogram
        vs = run.get_voxel_spacing(voxel_spacing)
        if not vs:
            logger.error(f"Voxel spacing {voxel_spacing} not found")
            return None

        tomos = vs.get_tomograms(tomo_type=tomo_type)
        if not tomos:
            logger.error(f"Tomogram type {tomo_type} not found")
            return None

        # Get dimensions from zarr
        import zarr

        tomo_array = zarr.open(tomos[0].zarr())["0"]
        vox_dim = tomo_array.shape[::-1]  # zarr is (z,y,x), we want (x,y,z)

        # Convert mesh to volume based on mode
        if mode == "watertight":
            vol = mesh_to_volume(mesh_obj, vox_dim, voxel_spacing)
        elif mode == "boundary":
            vol = mesh_to_boundary_volume(mesh_obj, vox_dim, voxel_spacing, boundary_sampling_density)
        else:
            raise ValueError(f"Unknown voxelization mode: {mode}. Must be 'watertight' or 'boundary'.")

        # Apply inversion if requested
        if invert:
            vol = ~vol

        # Create or get segmentation
        seg = run.new_segmentation(
            name=object_name,
            user_id=user_id,
            session_id=session_id,
            is_multilabel=is_multilabel,
            voxel_size=voxel_spacing,
            exist_ok=True,
        )

        # Store the volume using modern copick API
        seg.from_numpy(vol.astype(np.uint8))

        stats = {"voxels_created": int(np.sum(vol))}
        logger.info(f"Created segmentation with {stats['voxels_created']} voxels")
        return seg, stats

    except Exception as e:
        logger.error(f"Error creating segmentation: {e}")
        return None


# Create worker function using common infrastructure
_segmentation_from_mesh_worker = create_batch_worker(segmentation_from_mesh, "segmentation", "mesh", min_points=0)


# Create batch converter using common infrastructure
segmentation_from_mesh_batch = create_batch_converter(
    segmentation_from_mesh,
    "Converting meshes to segmentations",
    "segmentation",
    "mesh",
    min_points=0,
)

# Lazy batch converter for new architecture
segmentation_from_mesh_lazy_batch = create_lazy_batch_converter(
    converter_func=segmentation_from_mesh,
    task_description="Converting meshes to segmentations",
)
