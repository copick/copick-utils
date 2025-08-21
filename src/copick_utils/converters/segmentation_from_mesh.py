"""Convert meshes to segmentation volumes."""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import trimesh as tm
from trimesh.ray.ray_triangle import RayMeshIntersector
from concurrent.futures import ThreadPoolExecutor, as_completed

import copick
from copick.ops.run import map_runs

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickRoot, CopickMesh

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
            return tm.util.concatenate([g for g in trimesh_object.geometry.values()])
    elif isinstance(trimesh_object, tm.Trimesh):
        return trimesh_object
    else:
        raise ValueError("Input must be a Trimesh or Scene object")


def _onesmask_z(mesh: tm.Trimesh, voxel_dims: Tuple[int, int, int], voxel_spacing: float) -> np.ndarray:
    """Create mask by ray casting in Z direction."""
    intersector = RayMeshIntersector(mesh)

    # Create a grid of rays in XY plane, shooting in Z direction
    grid_x, grid_y = np.mgrid[0:voxel_dims[0], 0:voxel_dims[1]]
    ray_grid = (
        np.vstack([grid_x.ravel(), grid_y.ravel(), -np.ones((grid_x.size,))]).T * voxel_spacing
    )
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
    grid_y, grid_z = np.mgrid[0:voxel_dims[1], 0:voxel_dims[2]]
    ray_grid = (
        np.vstack([-np.ones((grid_y.size,)), grid_y.ravel(), grid_z.ravel()]).T * voxel_spacing
    )
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


def segmentation_from_mesh(
    run: "CopickRun",
    mesh_object_name: str,
    mesh_user_id: str, 
    mesh_session_id: str,
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str,
    voxel_spacing: float,
    tomo_type: str = "wbp",
    is_multilabel: bool = False,
) -> Dict[str, any]:
    """
    Convert a mesh to a segmentation volume.
    
    Args:
        run: CopickRun object
        mesh_object_name: Name of the mesh object
        mesh_user_id: User ID of the mesh
        mesh_session_id: Session ID of the mesh
        segmentation_name: Name for the output segmentation
        segmentation_user_id: User ID for the output segmentation  
        segmentation_session_id: Session ID for the output segmentation
        voxel_spacing: Voxel spacing for the segmentation
        tomo_type: Type of tomogram to use for reference dimensions
        is_multilabel: Whether the segmentation is multilabel
        
    Returns:
        Dictionary with conversion results
    """
    try:
        # Get the mesh
        meshes = run.get_meshes(
            object_name=mesh_object_name,
            user_id=mesh_user_id, 
            session_id=mesh_session_id
        )
        
        if not meshes:
            return {"processed": 0, "error": "No mesh found"}
            
        mesh_obj = ensure_mesh(meshes[0].mesh)
        if mesh_obj is None:
            return {"processed": 0, "error": "Empty mesh"}
            
        # Get reference dimensions from tomogram
        vs = run.get_voxel_spacing(voxel_spacing)
        if not vs:
            return {"processed": 0, "error": f"Voxel spacing {voxel_spacing} not found"}
            
        tomos = vs.get_tomograms(tomo_type=tomo_type)
        if not tomos:
            return {"processed": 0, "error": f"Tomogram type {tomo_type} not found"}
            
        # Get dimensions from zarr
        import zarr
        tomo_array = zarr.open(tomos[0].zarr())["0"]
        vox_dim = tomo_array.shape[::-1]  # zarr is (z,y,x), we want (x,y,z)
        
        # Convert mesh to volume
        vol = mesh_to_volume(mesh_obj, vox_dim, voxel_spacing)
        
        # Create or get segmentation
        existing_segs = run.get_segmentations(
            name=segmentation_name,
            user_id=segmentation_user_id,
            session_id=segmentation_session_id,
            is_multilabel=is_multilabel,
            voxel_size=voxel_spacing,
        )
        
        if existing_segs:
            seg = existing_segs[0]
        else:
            seg = run.new_segmentation(
                name=segmentation_name,
                user_id=segmentation_user_id,
                session_id=segmentation_session_id,
                is_multilabel=is_multilabel,
                voxel_size=voxel_spacing,
            )
            
        # Store the volume using modern copick API
        seg.from_numpy(vol.astype(np.uint8))
        
        return {"processed": 1, "voxels_created": np.sum(vol)}
        
    except Exception as e:
        return {"processed": 0, "error": str(e)}


def _segmentation_from_mesh_worker(
    run: "CopickRun",
    mesh_object_name: str,
    mesh_user_id: str,
    mesh_session_id: str,
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str,
    voxel_spacing: float,
    tomo_type: str,
    is_multilabel: bool,
) -> Dict:
    """Worker function for mesh to segmentation conversion."""
    try:            
        result = segmentation_from_mesh(
            run=run,
            mesh_object_name=mesh_object_name,
            mesh_user_id=mesh_user_id,
            mesh_session_id=mesh_session_id,
            segmentation_name=segmentation_name,
            segmentation_user_id=segmentation_user_id,
            segmentation_session_id=segmentation_session_id,
            voxel_spacing=voxel_spacing,
            tomo_type=tomo_type,
            is_multilabel=is_multilabel,
        )
        
        return result
        
    except Exception as e:
        return {"processed": 0, "errors": [str(e)]}


def segmentation_from_mesh_batch(
    root: "CopickRoot",
    mesh_object_name: str,
    mesh_user_id: str,
    mesh_session_id: str,
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str,
    voxel_spacing: float,
    tomo_type: str = "wbp",
    is_multilabel: bool = False,
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Dict]:
    """
    Batch convert meshes to segmentations across multiple runs.
    
    Args:
        root: CopickRoot object
        mesh_object_name: Name of the mesh object
        mesh_user_id: User ID of the mesh
        mesh_session_id: Session ID of the mesh
        segmentation_name: Name for the output segmentation
        segmentation_user_id: User ID for the output segmentation
        segmentation_session_id: Session ID for the output segmentation
        voxel_spacing: Voxel spacing for the segmentation
        tomo_type: Type of tomogram to use for reference dimensions
        is_multilabel: Whether the segmentation is multilabel
        run_names: List of run names to process (None for all runs)
        workers: Number of worker processes
        
    Returns:
        Dictionary mapping run names to conversion results
    """
    if run_names is None:
        run_names = [run.name for run in root.runs]
        
    return map_runs(
        callback=_segmentation_from_mesh_worker,
        root=root,
        runs=run_names,
        workers=workers,
        parallelism="process",
        show_progress=True,
        task_desc="Converting meshes to segmentations",
        mesh_object_name=mesh_object_name,
        mesh_user_id=mesh_user_id,
        mesh_session_id=mesh_session_id,
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        segmentation_session_id=segmentation_session_id,
        voxel_spacing=voxel_spacing,
        tomo_type=tomo_type,
        is_multilabel=is_multilabel,
    )