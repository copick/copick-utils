"""Convert meshes to segmentation volumes."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger
from trimesh.ray.ray_triangle import RayMeshIntersector

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


#: Ray origins sit this far (in voxels) off the voxel centres, across the ray. Meshes made by marching cubes have
#: their vertices and edges on the voxel grid, where rays through the voxel centres would hit edges and vertices
#: exactly and count the crossing twice or not at all. Rounding back to voxel indices removes the offset; a voxel
#: centre within about this distance of a sloped surface may fall on either side of it.
_RAY_OFFSET = (0.01 * np.sqrt(2), 0.01 * np.sqrt(3))

#: A crossing this close (in voxels) to a voxel centre counts as before it, so a face through the centres (e.g. a
#: box at the volume's edge) includes them where the ray enters and excludes them where it leaves.
_CENTRE_TOLERANCE = 1e-6


def _crossing_voxels(loc: np.ndarray, voxel_spacing: float, axis: int) -> np.ndarray:
    """Voxel indices of ray crossings: the ray's own voxel across it, and along it the first voxel centre after the
    crossing, so a voxel is inside when its centre is. Crossings before the volume move to its first voxel."""
    pos = loc / voxel_spacing
    idx = np.round(pos).astype("int")
    idx[:, axis] = np.maximum(np.ceil(pos[:, axis] - _CENTRE_TOLERANCE).astype("int"), 0)
    return idx


def _fill_from_crossings(
    int_loc: np.ndarray,
    voxel_dims: Tuple[int, int, int],
    axis: int,
) -> np.ndarray:
    """Inside/outside of every voxel along rays on ``axis``, from the voxels of their crossings: a voxel is inside
    after an odd number of crossings. Crossings in the same voxel cancel in pairs. Returns a (z, y, x) volume."""
    toggles = np.zeros((voxel_dims[2], voxel_dims[1], voxel_dims[0]), dtype=np.uint8)
    zyx = int_loc[int_loc[:, axis] < voxel_dims[axis]][:, ::-1]
    np.add.at(toggles, (zyx[:, 0], zyx[:, 1], zyx[:, 2]), 1)
    toggles &= 1
    return np.bitwise_xor.accumulate(toggles, axis=2 - axis).astype(bool)


def _ray_start(mesh: tm.Trimesh, voxel_spacing: float, axis: int) -> float:
    """Where rays along an axis start (in voxels): before the volume and before the mesh."""
    return min(-1.0, np.floor(mesh.bounds[0, axis] / voxel_spacing) - 1)


def _onesmask_z(mesh: tm.Trimesh, voxel_dims: Tuple[int, int, int], voxel_spacing: float) -> np.ndarray:
    """Create mask by ray casting in Z direction."""
    intersector = RayMeshIntersector(mesh)

    # Create a grid of rays in XY plane (off the voxel centres, see _RAY_OFFSET), shooting in Z direction
    grid_x, grid_y = np.mgrid[0 : voxel_dims[0], 0 : voxel_dims[1]]
    ray_grid = (
        np.vstack(
            [
                grid_x.ravel() + _RAY_OFFSET[0],
                grid_y.ravel() + _RAY_OFFSET[1],
                np.full(grid_x.size, _ray_start(mesh, voxel_spacing, 2)),
            ],
        ).T
        * voxel_spacing
    )
    ray_dir = np.zeros((ray_grid.shape[0], 3))
    ray_dir[:, 2] = 1

    loc, _, _ = intersector.intersects_location(ray_grid, ray_dir)

    # Build volume by tracking crossings
    return _fill_from_crossings(_crossing_voxels(loc, voxel_spacing, 2), voxel_dims, 2)


def _onesmask_x(mesh: tm.Trimesh, voxel_dims: Tuple[int, int, int], voxel_spacing: float) -> np.ndarray:
    """Create mask by ray casting in X direction."""
    intersector = RayMeshIntersector(mesh)

    # Create a grid of rays in YZ plane (off the voxel centres, see _RAY_OFFSET), shooting in X direction
    grid_y, grid_z = np.mgrid[0 : voxel_dims[1], 0 : voxel_dims[2]]
    ray_grid = (
        np.vstack(
            [
                np.full(grid_y.size, _ray_start(mesh, voxel_spacing, 0)),
                grid_y.ravel() + _RAY_OFFSET[0],
                grid_z.ravel() + _RAY_OFFSET[1],
            ],
        ).T
        * voxel_spacing
    )
    ray_dir = np.zeros((ray_grid.shape[0], 3))
    ray_dir[:, 0] = 1

    loc, _, _ = intersector.intersects_location(ray_grid, ray_dir)

    # Build volume by tracking crossings
    return _fill_from_crossings(_crossing_voxels(loc, voxel_spacing, 0), voxel_dims, 0)


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


# Lazy batch converter for new architecture
segmentation_from_mesh_lazy_batch = create_lazy_batch_converter(
    converter_func=segmentation_from_mesh,
    task_description="Converting meshes to segmentations",
)
