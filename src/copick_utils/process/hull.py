"""Compute various hull operations on meshes."""
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import trimesh as tm
from copick.util.log import get_logger

from copick_utils.converters.converter_common import create_batch_converter, store_mesh_with_stats
from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickRun

logger = get_logger(__name__)


def compute_hull(
    mesh: "CopickMesh",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    hull_type: str = "convex",
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """
    Compute hull of a CopickMesh object.

    Args:
        mesh: CopickMesh object to compute hull for
        run: CopickRun object
        object_name: Name for the output mesh
        session_id: Session ID for the output mesh
        user_id: User ID for the output mesh
        hull_type: Type of hull to compute ('convex')
        **kwargs: Additional keyword arguments

    Returns:
        Tuple of (CopickMesh object, stats dict) or None if operation failed.
        Stats dict contains 'vertices_created' and 'faces_created'.
    """
    try:
        # Get trimesh object
        trimesh_obj = mesh.mesh

        if trimesh_obj is None:
            logger.error("Could not load mesh data")
            return None

        # Ensure we have proper Trimesh object
        if isinstance(trimesh_obj, tm.Scene):
            if len(trimesh_obj.geometry) == 0:
                logger.error("Mesh is empty")
                return None
            trimesh_obj = trimesh_obj.to_mesh()

        if not isinstance(trimesh_obj, tm.Trimesh):
            logger.error(f"Expected Trimesh object, got {type(trimesh_obj)}")
            return None

        # Compute hull based on type
        if hull_type == "convex":
            hull_mesh = trimesh_obj.convex_hull
            shape_name = "convex hull"
        else:
            raise ValueError(f"Unknown hull type: {hull_type}")

        if hull_mesh is None:
            logger.error(f"Failed to compute {hull_type} hull")
            return None

        if hull_mesh.vertices.shape[0] == 0:
            logger.error(f"{hull_type.capitalize()} hull resulted in empty mesh")
            return None

        # Store the result
        copick_mesh, stats = store_mesh_with_stats(
            run=run,
            mesh=hull_mesh,
            object_name=object_name,
            session_id=session_id,
            user_id=user_id,
            shape_name=shape_name,
        )

        logger.info(f"Created {hull_type} hull mesh with {stats['vertices_created']} vertices")
        return copick_mesh, stats

    except Exception as e:
        logger.error(f"Error computing {hull_type} hull: {e}")
        return None


# Create batch converter
hull_from_mesh_batch = create_batch_converter(
    compute_hull,
    "Computing hull from meshes",
    "mesh",
    "mesh",
    min_points=0,
)

# Lazy batch converter for new architecture
hull_lazy_batch = create_lazy_batch_converter(
    converter_func=compute_hull,
    task_description="Computing convex hulls",
)
