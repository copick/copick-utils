"""Mesh operations (union, intersection, difference, exclusion, concatenate)."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import trimesh as tm
from copick.util.log import get_logger

from copick_utils.converters.converter_common import (
    create_batch_converter,
    create_batch_worker,
    store_mesh_with_stats,
)
from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickRun

logger = get_logger(__name__)


def _perform_mesh_boolean_operation(mesh1: tm.Trimesh, mesh2: tm.Trimesh, operation: str) -> Optional[tm.Trimesh]:
    """
    Perform boolean operation between two meshes.

    Args:
        mesh1: First mesh
        mesh2: Second mesh
        operation: Type of boolean operation ('union', 'difference', 'intersection', 'exclusion', 'concatenate')

    Returns:
        Result mesh or None if operation failed
    """
    try:
        if operation == "union":
            result = mesh1.union(mesh2)
        elif operation == "difference":
            result = mesh1.difference(mesh2)
        elif operation == "intersection":
            result = mesh1.intersection(mesh2)
        elif operation == "exclusion":
            # Exclusion = (A union B) - (A intersection B)
            union_mesh = mesh1.union(mesh2)
            intersection_mesh = mesh1.intersection(mesh2)
            result = union_mesh.difference(intersection_mesh)
        elif operation == "concatenate":
            # Simple concatenation without boolean operations
            result = tm.util.concatenate([mesh1, mesh2])
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Handle the case where result might be a Scene or empty
        if isinstance(result, tm.Scene):
            if len(result.geometry) == 0:
                logger.warning(f"{operation.capitalize()} operation resulted in empty geometry")
                return None
            # Concatenate all geometries in the scene
            result = tm.util.concatenate(list(result.geometry.values()))
        elif isinstance(result, tm.Trimesh):
            if result.vertices.shape[0] == 0:
                logger.warning(f"{operation.capitalize()} operation resulted in empty mesh")
                return None
        else:
            logger.warning(f"{operation.capitalize()} operation returned unexpected type: {type(result)}")
            return None

        return result

    except Exception as e:
        logger.error(f"{operation.capitalize()} operation failed: {e}")
        return None


def mesh_boolean_operation(
    mesh1: "CopickMesh",
    mesh2: "CopickMesh",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    operation: str,
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """
    Perform boolean operation between two CopickMesh objects.

    Args:
        mesh1: First CopickMesh object
        mesh2: Second CopickMesh object
        run: CopickRun object
        object_name: Name for the output mesh
        session_id: Session ID for the output mesh
        user_id: User ID for the output mesh
        operation: Type of operation ('union', 'difference', 'intersection', 'exclusion', 'concatenate')
        **kwargs: Additional keyword arguments

    Returns:
        Tuple of (CopickMesh object, stats dict) or None if operation failed.
        Stats dict contains 'vertices_created' and 'faces_created'.
    """
    try:
        # Get trimesh objects
        trimesh1 = mesh1.mesh
        trimesh2 = mesh2.mesh

        if trimesh1 is None or trimesh2 is None:
            logger.error("Could not load mesh data")
            return None

        # Ensure we have proper Trimesh objects
        if isinstance(trimesh1, tm.Scene):
            if len(trimesh1.geometry) == 0:
                logger.error("First mesh is empty")
                return None
            trimesh1 = tm.util.concatenate(list(trimesh1.geometry.values()))

        if isinstance(trimesh2, tm.Scene):
            if len(trimesh2.geometry) == 0:
                logger.error("Second mesh is empty")
                return None
            trimesh2 = tm.util.concatenate(list(trimesh2.geometry.values()))

        # Perform boolean operation
        result_mesh = _perform_mesh_boolean_operation(trimesh1, trimesh2, operation)

        if result_mesh is None:
            logger.error(f"Boolean {operation} operation failed")
            return None

        # Store the result
        copick_mesh, stats = store_mesh_with_stats(
            run=run,
            mesh=result_mesh,
            object_name=object_name,
            session_id=session_id,
            user_id=user_id,
            shape_name=f"{operation} result",
        )

        logger.info(f"Created {operation} mesh with {stats['vertices_created']} vertices")
        return copick_mesh, stats

    except Exception as e:
        logger.error(f"Error performing {operation}: {e}")
        return None


# Individual operation functions
def mesh_union(
    mesh1: "CopickMesh",
    mesh2: "CopickMesh",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """Union of two meshes."""
    return mesh_boolean_operation(mesh1, mesh2, run, object_name, session_id, user_id, "union", **kwargs)


def mesh_difference(
    mesh1: "CopickMesh",
    mesh2: "CopickMesh",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """Difference of two meshes (mesh1 - mesh2)."""
    return mesh_boolean_operation(mesh1, mesh2, run, object_name, session_id, user_id, "difference", **kwargs)


def mesh_intersection(
    mesh1: "CopickMesh",
    mesh2: "CopickMesh",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """Intersection of two meshes."""
    return mesh_boolean_operation(mesh1, mesh2, run, object_name, session_id, user_id, "intersection", **kwargs)


def mesh_exclusion(
    mesh1: "CopickMesh",
    mesh2: "CopickMesh",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """Exclusive or (XOR) of two meshes."""
    return mesh_boolean_operation(mesh1, mesh2, run, object_name, session_id, user_id, "exclusion", **kwargs)


def mesh_concatenate(
    mesh1: "CopickMesh",
    mesh2: "CopickMesh",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """Concatenate two meshes without boolean operations."""
    return mesh_boolean_operation(mesh1, mesh2, run, object_name, session_id, user_id, "concatenate", **kwargs)


def mesh_multi_union(
    meshes: List["CopickMesh"],
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """
    Perform N-way boolean union of multiple meshes.

    Args:
        meshes: List of CopickMesh objects (N≥2)
        run: CopickRun object
        object_name: Name for output mesh
        session_id: Session ID for output
        user_id: User ID for output
        **kwargs: Additional arguments

    Returns:
        Tuple of (CopickMesh, stats) or None if failed
    """
    try:
        if len(meshes) < 2:
            logger.error("Need at least 2 meshes for N-way union")
            return None

        # Load all meshes and convert to Trimesh objects
        trimeshes = []
        for i, mesh in enumerate(meshes):
            trimesh_obj = mesh.mesh
            if trimesh_obj is None:
                logger.error(f"Could not load mesh {i+1} (session: {mesh.session_id})")
                return None

            # Handle Scene objects
            if isinstance(trimesh_obj, tm.Scene):
                if len(trimesh_obj.geometry) == 0:
                    logger.error(f"Mesh {i+1} is empty")
                    return None
                trimesh_obj = tm.util.concatenate(list(trimesh_obj.geometry.values()))

            trimeshes.append(trimesh_obj)

        # Perform cumulative boolean union
        result = trimeshes[0]
        for i, trimesh_obj in enumerate(trimeshes[1:], start=2):
            try:
                result = result.union(trimesh_obj)
                # Handle Scene result
                if isinstance(result, tm.Scene):
                    if len(result.geometry) == 0:
                        logger.error(f"Union failed at mesh {i}: empty result")
                        return None
                    result = tm.util.concatenate(list(result.geometry.values()))
            except Exception as e:
                logger.error(f"Union failed at mesh {i}: {e}")
                return None

        # Store the result
        copick_mesh, stats = store_mesh_with_stats(
            run=run,
            mesh=result,
            object_name=object_name,
            session_id=session_id,
            user_id=user_id,
            shape_name=f"{len(meshes)}-way union result",
        )

        logger.info(f"Created {len(meshes)}-way union with {stats['vertices_created']} vertices")
        return copick_mesh, stats

    except Exception as e:
        logger.error(f"Error in N-way mesh union: {e}")
        return None


def mesh_multi_concatenate(
    meshes: List["CopickMesh"],
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """
    Concatenate N meshes without boolean operations.

    Args:
        meshes: List of CopickMesh objects (N≥2)
        run: CopickRun object
        object_name: Name for output mesh
        session_id: Session ID for output
        user_id: User ID for output
        **kwargs: Additional arguments

    Returns:
        Tuple of (CopickMesh, stats) or None if failed
    """
    try:
        if len(meshes) < 2:
            logger.error("Need at least 2 meshes for N-way concatenation")
            return None

        # Load all meshes
        trimeshes = []
        for i, mesh in enumerate(meshes):
            trimesh_obj = mesh.mesh
            if trimesh_obj is None:
                logger.error(f"Could not load mesh {i+1} (session: {mesh.session_id})")
                return None

            # Handle Scene objects
            if isinstance(trimesh_obj, tm.Scene):
                if len(trimesh_obj.geometry) == 0:
                    logger.error(f"Mesh {i+1} is empty")
                    return None
                trimesh_obj = tm.util.concatenate(list(trimesh_obj.geometry.values()))

            trimeshes.append(trimesh_obj)

        # Concatenate all meshes
        result = tm.util.concatenate(trimeshes)

        # Store the result
        copick_mesh, stats = store_mesh_with_stats(
            run=run,
            mesh=result,
            object_name=object_name,
            session_id=session_id,
            user_id=user_id,
            shape_name=f"{len(meshes)}-mesh concatenation",
        )

        logger.info(f"Concatenated {len(meshes)} meshes: {stats['vertices_created']} vertices")
        return copick_mesh, stats

    except Exception as e:
        logger.error(f"Error in N-way concatenation: {e}")
        return None


# Create batch workers for each operation
_mesh_union_worker = create_batch_worker(mesh_union, "mesh", "mesh", min_points=0)
_mesh_difference_worker = create_batch_worker(mesh_difference, "mesh", "mesh", min_points=0)
_mesh_intersection_worker = create_batch_worker(mesh_intersection, "mesh", "mesh", min_points=0)
_mesh_exclusion_worker = create_batch_worker(mesh_exclusion, "mesh", "mesh", min_points=0)
_mesh_concatenate_worker = create_batch_worker(mesh_concatenate, "mesh", "mesh", min_points=0)

# Create batch converters
mesh_union_batch = create_batch_converter(
    mesh_union,
    "Computing mesh unions",
    "mesh",
    "mesh",
    min_points=0,
    dual_input=True,
)

mesh_difference_batch = create_batch_converter(
    mesh_difference,
    "Computing mesh differences",
    "mesh",
    "mesh",
    min_points=0,
    dual_input=True,
)

mesh_intersection_batch = create_batch_converter(
    mesh_intersection,
    "Computing mesh intersections",
    "mesh",
    "mesh",
    min_points=0,
    dual_input=True,
)

mesh_exclusion_batch = create_batch_converter(
    mesh_exclusion,
    "Computing mesh exclusions",
    "mesh",
    "mesh",
    min_points=0,
    dual_input=True,
)

mesh_concatenate_batch = create_batch_converter(
    mesh_concatenate,
    "Computing mesh concatenations",
    "mesh",
    "mesh",
    min_points=0,
    dual_input=True,
)

# Lazy batch converters for new architecture
mesh_union_lazy_batch = create_lazy_batch_converter(
    converter_func=mesh_union,
    task_description="Computing mesh unions",
)

mesh_difference_lazy_batch = create_lazy_batch_converter(
    converter_func=mesh_difference,
    task_description="Computing mesh differences",
)

mesh_intersection_lazy_batch = create_lazy_batch_converter(
    converter_func=mesh_intersection,
    task_description="Computing mesh intersections",
)

mesh_exclusion_lazy_batch = create_lazy_batch_converter(
    converter_func=mesh_exclusion,
    task_description="Computing mesh exclusions",
)

mesh_concatenate_lazy_batch = create_lazy_batch_converter(
    converter_func=mesh_concatenate,
    task_description="Computing mesh concatenations",
)

# Lazy batch converters for N-way operations
mesh_multi_union_lazy_batch = create_lazy_batch_converter(
    converter_func=mesh_multi_union,
    task_description="Computing N-way mesh unions",
)

mesh_multi_concatenate_lazy_batch = create_lazy_batch_converter(
    converter_func=mesh_multi_concatenate,
    task_description="Computing N-way mesh concatenations",
)
