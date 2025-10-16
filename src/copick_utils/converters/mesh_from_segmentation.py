"""Convert segmentation volumes to meshes."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger

from copick_utils.converters.converter_common import (
    create_batch_converter,
    create_batch_worker,
    store_mesh_with_stats,
)
from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickRun, CopickSegmentation

logger = get_logger(__name__)


def volume_to_mesh(volume: np.ndarray, voxel_spacing: float, level: float = 0.5, step_size: int = 1) -> tm.Trimesh:
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
        spacing=(voxel_spacing, voxel_spacing, voxel_spacing),
    )

    # Create trimesh object
    mesh = tm.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    return mesh


def mesh_from_segmentation(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    level: float = 0.5,
    step_size: int = 1,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """
    Convert a CopickSegmentation to a mesh.

    Args:
        segmentation: CopickSegmentation object to convert
        run: CopickRun object
        object_name: Name for the output mesh object
        session_id: Session ID for the output mesh
        user_id: User ID for the output mesh
        level: Isosurface level for marching cubes
        step_size: Step size for marching cubes

    Returns:
        Tuple of (CopickMesh object, stats dict) or None if creation failed.
        Stats dict contains 'vertices_created' and 'faces_created'.
    """
    try:
        # Load the volume from the segmentation
        volume = segmentation.numpy()

        if volume is None or volume.size == 0:
            logger.error("Empty or invalid volume")
            return None

        # Get voxel spacing from segmentation
        voxel_spacing = segmentation.voxel_size

        # Convert volume to mesh
        mesh = volume_to_mesh(volume, voxel_spacing, level, step_size)

        if mesh.vertices.size == 0:
            logger.error("Empty mesh generated")
            return None

        # Store mesh and return stats
        return store_mesh_with_stats(run, mesh, object_name, session_id, user_id, "mesh")

    except Exception as e:
        logger.error(f"Error creating mesh: {e}")
        return None


# Create worker function using common infrastructure
_mesh_from_segmentation_worker = create_batch_worker(mesh_from_segmentation, "mesh", "segmentation", min_points=0)


# Create batch converter using common infrastructure
mesh_from_segmentation_batch = create_batch_converter(
    mesh_from_segmentation,
    "Converting segmentations to meshes",
    "mesh",
    "segmentation",
    min_points=0,
)

# Lazy batch converter for new architecture
mesh_from_segmentation_lazy_batch = create_lazy_batch_converter(
    converter_func=mesh_from_segmentation,
    task_description="Converting segmentations to meshes",
)
