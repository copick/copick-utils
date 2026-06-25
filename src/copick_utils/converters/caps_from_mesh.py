"""Extract the top/bottom surfaces ("caps") of a closed slab box mesh.

A boundary slab (e.g. the ``valid-sample`` mesh) is a closed box: a top surface, a parallel
bottom surface, and 4 side walls that seal them into a watertight volume. For distance-based
particle selection (``copick logical clippicks``) one usually wants distance to the *top/bottom*
of the specimen only -- the side walls would otherwise contaminate the distance field for picks
near the lateral edges.

This module extracts only the cap faces (the near-horizontal top/bottom surfaces) as an open mesh,
discarding the near-vertical side walls. Classification is purely geometric (face-normal
orientation relative to the slab axis), so it is robust to the input being a re-triangulated
boolean result (the real ``valid-sample`` mesh is ``valid-area`` INTERSECT ``sample``, which
destroys any original face ordering).
"""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger

from copick_utils.converters.converter_common import store_mesh_with_stats
from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickRun

logger = get_logger(__name__)

# Mesh vertices/normals are in physical (x, y, z) order, so the beam/slab axis "z" is index 2.
# (This is the opposite of array/segmentation space, where z is axis 0.)
_AXIS_VEC = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def _resolve_axis_vector(mesh: tm.Trimesh, axis: str, auto_axis: bool) -> np.ndarray:
    """Resolve the unit slab-normal vector, oriented toward +global-z.

    With ``auto_axis`` the slab normal is taken as the direction of smallest oriented-bounding-box
    extent (robust for slabs tilted past the cap threshold). Otherwise it is the fixed ``axis`` unit
    vector. The result is oriented so its z-component is non-negative, so "top" (positive normal dot)
    consistently means the upper surface for near-horizontal slabs.
    """
    vec = None
    if auto_axis:
        try:
            obb = mesh.bounding_box_oriented
            extents = np.asarray(obb.primitive.extents, dtype=float)
            rotation = np.asarray(obb.primitive.transform, dtype=float)[:3, :3]
            vec = rotation[:, int(np.argmin(extents))]
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"auto-axis failed ({e}); falling back to fixed axis '{axis}'")
            vec = None

    if vec is None:
        vec = _AXIS_VEC[axis].copy()

    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        vec = _AXIS_VEC[axis].copy()
        norm = 1.0
    vec = vec / norm

    # Deterministic orientation so top/bottom labelling is stable.
    if vec[2] < 0:
        vec = -vec
    return vec


def extract_slab_caps(
    mesh: tm.Trimesh,
    axis: str = "z",
    angle_threshold: float = 45.0,
    which: str = "both",
    auto_axis: bool = False,
) -> Optional[tm.Trimesh]:
    """Extract the top/bottom cap faces of a slab box mesh as an open mesh.

    Faces whose normal is within ``angle_threshold`` degrees of the slab axis are treated as caps
    (top = normal pointing along +axis, bottom = -axis); faces near-perpendicular to the axis are
    side walls and are discarded. Degenerate (zero-area) faces have a zero normal and are therefore
    never selected as caps.

    Args:
        mesh: Input slab mesh (ideally watertight; classification is done on its faces).
        axis: Slab-normal axis in physical mesh coordinates: 'x', 'y', or 'z' (default 'z').
        angle_threshold: Max angle (degrees) between a face normal and the axis for a face to count
            as a cap. Default 45 (the natural midpoint between horizontal caps and vertical walls).
        which: Which caps to return: 'both' (default), 'top', or 'bottom'.
        auto_axis: If True, infer the slab normal from the mesh's thinnest oriented-bounding-box
            extent instead of using ``axis`` (useful for strongly tilted slabs).

    Returns:
        An OPEN ``trimesh.Trimesh`` of the selected cap faces (vertices compacted, winding
        preserved -- it is NOT re-normal-fixed), or ``None`` if no cap faces are found.
    """
    if which not in ("both", "top", "bottom"):
        raise ValueError(f"Invalid 'which': {which!r}. Must be 'both', 'top', or 'bottom'.")
    if axis not in _AXIS_VEC:
        raise ValueError(f"Invalid 'axis': {axis!r}. Must be 'x', 'y', or 'z'.")
    if not 0.0 < angle_threshold < 90.0:
        raise ValueError(f"angle_threshold must be in (0, 90) degrees, got {angle_threshold}.")

    if mesh.faces.shape[0] == 0:
        logger.warning("Input mesh has no faces; nothing to extract.")
        return None

    axis_vec = _resolve_axis_vector(mesh, axis, auto_axis)

    # face_normals are unit vectors (zero for degenerate faces); dot gives cos(angle-to-axis).
    dots = np.asarray(mesh.face_normals) @ axis_vec
    cos_threshold = np.cos(np.deg2rad(angle_threshold))
    is_cap = np.abs(dots) >= cos_threshold

    if which == "both":
        selected = is_cap
    elif which == "top":
        selected = is_cap & (dots > 0)
    else:  # bottom
        selected = is_cap & (dots < 0)

    face_indices = np.nonzero(selected)[0]
    if face_indices.size == 0:
        logger.warning(
            f"No '{which}' cap faces found (axis={axis}, angle_threshold={angle_threshold} deg). "
            "The slab may be tilted beyond the threshold -- try --auto-axis or a larger "
            "--angle-threshold.",
        )
        return None

    # submesh compacts vertices to those referenced and preserves face winding.
    caps = mesh.submesh([face_indices], append=True)
    return caps


def caps_from_mesh(
    mesh: "CopickMesh",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    axis: str = "z",
    angle_threshold: float = 45.0,
    which: str = "both",
    auto_axis: bool = False,
    voxel_spacing: Optional[float] = None,
    **kwargs,
) -> Optional[Tuple["CopickMesh", Dict[str, int]]]:
    """Converter: extract slab caps from a CopickMesh and store the result as a new mesh.

    Args:
        mesh: CopickMesh to extract caps from.
        run: Copick run object.
        object_name: Name for the output mesh object.
        session_id: Session ID for the output mesh.
        user_id: User ID for the output mesh.
        axis: Slab-normal axis ('x', 'y', 'z').
        angle_threshold: Max angle (degrees) from the axis for a face to count as a cap.
        which: Which caps to keep ('both', 'top', 'bottom').
        auto_axis: Infer the slab normal from the mesh's thinnest OBB extent.
        voxel_spacing: Unused (absorbs the value the lazy task dict carries for mesh inputs).
        **kwargs: Additional arguments (ignored).

    Returns:
        Tuple of (CopickMesh, stats dict) or None if extraction failed. Stats contain
        'vertices_created' and 'faces_created'.
    """
    try:
        trimesh_obj = mesh.mesh
        if trimesh_obj is None:
            logger.error("Could not load mesh data")
            return None

        # Handle Scene objects (mirrors picks_from_mesh / segmentation_from_mesh).
        if isinstance(trimesh_obj, tm.Scene):
            if len(trimesh_obj.geometry) == 0:
                logger.error("Mesh is empty")
                return None
            trimesh_obj = tm.util.concatenate(list(trimesh_obj.geometry.values()))

        if not trimesh_obj.is_watertight:
            logger.warning(
                "Input mesh is not watertight; normal-based cap classification may be less reliable.",
            )
        elif which != "both":
            # Consistent outward winding makes the normal-sign top/bottom split reliable. Only
            # needed when distinguishing top vs bottom; harmless but skipped for the 'both' path.
            trimesh_obj.fix_normals()

        caps = extract_slab_caps(
            trimesh_obj,
            axis=axis,
            angle_threshold=angle_threshold,
            which=which,
            auto_axis=auto_axis,
        )
        if caps is None or caps.faces.shape[0] == 0:
            return None

        return store_mesh_with_stats(run, caps, object_name, session_id, user_id, f"{which}-caps")

    except Exception as e:
        logger.error(f"Error extracting slab caps: {e}")
        return None


# Lazy batch converter for the parallel discovery/processing architecture.
caps_from_mesh_lazy_batch = create_lazy_batch_converter(
    converter_func=caps_from_mesh,
    task_description="Extracting slab caps from meshes",
)
