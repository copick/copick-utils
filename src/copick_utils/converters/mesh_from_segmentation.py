"""Convert segmentation volumes to meshes."""

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import trimesh as tm

import copick
from copick.ops.run import map_runs

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickRoot

def volume_to_mesh(
    volume: np.ndarray, 
    voxel_spacing: float,
    level: float = 0.5,
    step_size: int = 1
) -> tm.Trimesh:
    """
    Convert a binary volume to a mesh using marching cubes.
    
    Args:
        volume: Binary volume array with shape (z, y, x) 
        voxel_spacing: Spacing between voxels in physical units
        level: Isosurface level for marching cubes
        step_size: Step size for marching cubes (higher = coarser mesh)
        
    Returns:
        Trimesh object representing the mesh
    """
    from skimage import measure
    
    # Generate mesh using marching cubes
    verts, faces, normals, values = measure.marching_cubes(
        volume.astype(float), 
        level=level, 
        step_size=step_size,
        spacing=(voxel_spacing, voxel_spacing, voxel_spacing)
    )
    
    # Create trimesh object
    mesh = tm.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    return mesh


def mesh_from_segmentation(
    run: "CopickRun",
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str, 
    mesh_object_name: str,
    mesh_user_id: str,
    mesh_session_id: str,
    voxel_spacing: float,
    is_multilabel: bool = False,
    level: float = 0.5,
    step_size: int = 1,
) -> Dict[str, any]:
    """
    Convert a segmentation volume to a mesh.
    
    Args:
        run: CopickRun object
        segmentation_name: Name of the segmentation
        segmentation_user_id: User ID of the segmentation
        segmentation_session_id: Session ID of the segmentation
        mesh_object_name: Name for the output mesh object
        mesh_user_id: User ID for the output mesh
        mesh_session_id: Session ID for the output mesh  
        voxel_spacing: Voxel spacing of the segmentation
        is_multilabel: Whether the segmentation is multilabel
        level: Isosurface level for marching cubes
        step_size: Step size for marching cubes
        
    Returns:
        Dictionary with conversion results
    """
    try:
        # Get the segmentation
        segs = run.get_segmentations(
            name=segmentation_name,
            user_id=segmentation_user_id,
            session_id=segmentation_session_id, 
            is_multilabel=is_multilabel,
            voxel_size=voxel_spacing,
        )
        
        if not segs:
            return {"processed": 0, "error": "No segmentation found"}
            
        seg = segs[0]
        
        # Load the volume
        volume = seg.numpy()
        
        if volume.size == 0:
            return {"processed": 0, "error": "Empty volume"}
            
        # Convert volume to mesh
        mesh = volume_to_mesh(volume, voxel_spacing, level, step_size)
        
        if mesh.vertices.size == 0:
            return {"processed": 0, "error": "Empty mesh generated"}
            
        # Create or get mesh object
        existing_meshes = run.get_meshes(
            object_name=mesh_object_name,
            user_id=mesh_user_id,
            session_id=mesh_session_id
        )
        
        if existing_meshes:
            mesh_obj = existing_meshes[0]
        else:
            mesh_obj = run.new_mesh(
                object_name=mesh_object_name,
                user_id=mesh_user_id,
                session_id=mesh_session_id
            )
            
        # Store the mesh
        mesh_obj.store_mesh(mesh)
        
        return {"processed": 1, "vertices_created": len(mesh.vertices), "faces_created": len(mesh.faces)}
        
    except Exception as e:
        return {"processed": 0, "error": str(e)}


def _mesh_from_segmentation_worker(
    run: "CopickRun",
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str,
    mesh_object_name: str,
    mesh_user_id: str,
    mesh_session_id: str,
    voxel_spacing: float,
    is_multilabel: bool,
    level: float,
    step_size: int,
) -> Dict:
    """Worker function for segmentation to mesh conversion."""
    try:           
        result = mesh_from_segmentation(
            run=run,
            segmentation_name=segmentation_name,
            segmentation_user_id=segmentation_user_id,
            segmentation_session_id=segmentation_session_id,
            mesh_object_name=mesh_object_name,
            mesh_user_id=mesh_user_id,
            mesh_session_id=mesh_session_id,
            voxel_spacing=voxel_spacing,
            is_multilabel=is_multilabel,
            level=level,
            step_size=step_size,
        )
        
        return result
        
    except Exception as e:
        return {"processed": 0, "errors": [str(e)]}


def mesh_from_segmentation_batch(
    root: "CopickRoot",
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str,
    mesh_object_name: str,
    mesh_user_id: str,
    mesh_session_id: str,
    voxel_spacing: float,
    is_multilabel: bool = False,
    level: float = 0.5,
    step_size: int = 1,
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Dict]:
    """
    Batch convert segmentations to meshes across multiple runs.
    
    Args:
        root: CopickRoot object  
        segmentation_name: Name of the segmentation
        segmentation_user_id: User ID of the segmentation
        segmentation_session_id: Session ID of the segmentation
        mesh_object_name: Name for the output mesh object
        mesh_user_id: User ID for the output mesh
        mesh_session_id: Session ID for the output mesh
        voxel_spacing: Voxel spacing of the segmentation
        is_multilabel: Whether the segmentation is multilabel
        level: Isosurface level for marching cubes
        step_size: Step size for marching cubes
        run_names: List of run names to process (None for all runs)
        workers: Number of worker processes
        
    Returns:
        Dictionary mapping run names to conversion results
    """
    if run_names is None:
        run_names = [run.name for run in root.runs]
        
    return map_runs(
        callback=_mesh_from_segmentation_worker,
        root=root,
        runs=run_names,
        workers=workers,
        parallelism="process",
        show_progress=True,
        task_desc="Converting segmentations to meshes",
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        segmentation_session_id=segmentation_session_id,
        mesh_object_name=mesh_object_name,
        mesh_user_id=mesh_user_id,
        mesh_session_id=mesh_session_id,
        voxel_spacing=voxel_spacing,
        is_multilabel=is_multilabel,
        level=level,
        step_size=step_size,
    )