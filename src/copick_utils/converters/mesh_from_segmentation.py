"""Convert segmentation volumes to meshes."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger

from copick_utils.converters.converter_common import (
    store_mesh_with_stats,
)
from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickRun, CopickSegmentation

logger = get_logger(__name__)


def volume_to_mesh(volume: np.ndarray, voxel_spacing: float, level: float = 0.5, step_size: int = 1) -> tm.Trimesh:
    """
    Convert a binary volume to a mesh using marching cubes.

    Vertices are (x, y, z) in physical units, with voxel index i at i * voxel_spacing, like copick's other meshes
    and picks. The volume is padded with a voxel of background first, so the surface is closed where the
    segmentation touches the edge of the volume.

    Args:
        volume: Binary volume array with shape (z, y, x)
        voxel_spacing: Spacing between voxels in physical units
        level: Isosurface level for marching cubes
        step_size: Step size for marching cubes (higher = coarser mesh)

    Returns:
        Trimesh object representing the mesh (outward-facing triangles)
    """
    from skimage import measure

    # Generate mesh using marching cubes
    verts, faces, normals, values = measure.marching_cubes(
        np.pad(volume.astype(float), 1),
        level=level,
        step_size=step_size,
        spacing=(voxel_spacing, voxel_spacing, voxel_spacing),
    )

    # marching_cubes works in the array's (z, y, x) order: reverse to (x, y, z) and undo the padding. Reversing the
    # axes mirrors the mesh, which also turns its triangles to face outward, so the faces stay as they are.
    mesh = tm.Trimesh(vertices=verts[:, ::-1] - voxel_spacing, faces=faces, vertex_normals=normals[:, ::-1])

    return mesh


def label_meshes(
    volume: np.ndarray,
    voxel_spacing: float,
    level: float = 0.5,
    step_size: int = 1,
) -> Dict[int, tm.Trimesh]:
    """
    Convert each non-zero label of a volume to its own mesh (see :func:`volume_to_mesh`).

    Each label is meshed from its own mask, cropped to the label's bounding box.

    Args:
        volume: Label volume with shape (z, y, x)
        voxel_spacing: Spacing between voxels in physical units
        level: Isosurface level for marching cubes (of each label's binary mask)
        step_size: Step size for marching cubes

    Returns:
        Meshes by label value, in increasing label order
    """
    from scipy import ndimage

    if np.issubdtype(volume.dtype, np.integer) and volume.size and volume.min() >= 0:
        labels = volume
        values = None
    else:
        # Map the values to consecutive indices, 0 staying background.
        values, labels = np.unique(volume, return_inverse=True)
        labels = labels.reshape(volume.shape)
        if values[0] != 0:
            values = np.concatenate([[0], values])
            labels = labels + 1

    meshes = {}
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        mesh = volume_to_mesh(labels[box] == index, voxel_spacing, level, step_size)
        # The crop starts at (z, y, x) = box starts: move the mesh there.
        mesh.apply_translation(np.array([s.start for s in box][::-1]) * voxel_spacing)
        meshes[int(index if values is None else values[index])] = mesh
    return meshes


def mesh_from_segmentation(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    level: float = 0.5,
    step_size: int = 1,
    voxel_spacing: Optional[float] = None,
    individual_meshes: bool = False,
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """
    Convert a CopickSegmentation to a mesh.

    Args:
        segmentation: CopickSegmentation object to convert
        run: CopickRun object
        object_name: Name for the output mesh object
        session_id: Session ID for the output mesh; with ``individual_meshes``, a template whose ``{instance_id}``
            is replaced by each label (without it, ``-{label}`` is appended)
        user_id: User ID for the output mesh
        level: Isosurface level for marching cubes
        step_size: Step size for marching cubes
        individual_meshes: Store one mesh per non-zero label instead of one mesh of the whole segmentation

    Returns:
        Tuple of (CopickMesh object, stats dict) or None if creation failed. With ``individual_meshes``, the first
        label's mesh and the totals. Stats dict contains 'vertices_created' and 'faces_created'.
    """
    try:
        # Load the volume from the segmentation
        volume = segmentation.numpy()

        if volume is None or volume.size == 0:
            logger.error("Empty or invalid volume")
            return None

        # Get voxel spacing from segmentation
        voxel_spacing = segmentation.voxel_size

        if individual_meshes:
            meshes = label_meshes(volume, voxel_spacing, level, step_size)
            if not meshes:
                logger.error("No labels to mesh")
                return None
            first = None
            stats = {"vertices_created": 0, "faces_created": 0}
            for label, mesh in meshes.items():
                if "{instance_id}" in session_id:
                    label_session_id = session_id.replace("{instance_id}", str(label))
                else:
                    label_session_id = f"{session_id}-{label}"
                copick_mesh, mesh_stats = store_mesh_with_stats(
                    run,
                    mesh,
                    object_name,
                    label_session_id,
                    user_id,
                    "mesh",
                )
                if first is None:
                    first = copick_mesh
                for key in stats:
                    stats[key] += mesh_stats[key]
            return first, stats

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


# Lazy batch converter for new architecture
mesh_from_segmentation_lazy_batch = create_lazy_batch_converter(
    converter_func=mesh_from_segmentation,
    task_description="Converting segmentations to meshes",
)
