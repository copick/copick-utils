"""Smooth segmentations using mesh-based methods."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import trimesh
import trimesh.smoothing
from copick.util.log import get_logger
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from skimage.measure import marching_cubes

from copick_utils.converters.lazy_converter import create_lazy_batch_converter

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickSegmentation

logger = get_logger(__name__)


# ── Pure algorithm functions ──────────────────────────────────────────────────


def _smooth_via_mesh(
    binary_seg: np.ndarray,
    voxel_size: tuple = (1.0, 1.0, 1.0),
    taubin_lambda: float = 0.5,
    taubin_nu: float = -0.53,
    iterations: int = 10,
) -> np.ndarray:
    """
    Mesh-based tangential smoothing.

    1. Marching cubes to extract surface mesh in physical coordinates.
    2. Taubin smoothing (lambda/nu algorithm — tangential, volume-preserving).
    3. Re-rasterize onto the original voxel grid using efficient ray-casting.

    Args:
        binary_seg: Binary segmentation array (z, y, x).
        voxel_size: Physical voxel size (z, y, x) in any consistent unit (must be isotropic).
        taubin_lambda: Positive shrink factor (0 < lambda < 1).
        taubin_nu: Negative inflate factor (nu < -lambda, typically ~ -lambda - 0.03).
        iterations: Number of Taubin lambda/nu steps.

    Returns:
        Smoothed binary segmentation array with same shape as input.
    """
    import time

    seg = binary_seg.astype(np.float32)
    logger.debug(
        f"Input shape: {binary_seg.shape}, voxel_size: {voxel_size}, foreground voxels: {np.count_nonzero(binary_seg)}",
    )

    # 1. Extract surface mesh
    t0 = time.perf_counter()
    verts, faces, *_ = marching_cubes(seg, level=0.5, spacing=voxel_size)
    # marching_cubes returns vertices in (z, y, x) order; mesh_to_volume expects (x, y, z)
    verts = verts[:, ::-1]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    logger.debug(f"Marching cubes: {time.perf_counter() - t0:.2f}s — {len(verts)} vertices, {len(faces)} faces")

    # 2. Taubin smoothing
    t0 = time.perf_counter()
    trimesh.smoothing.filter_taubin(
        mesh,
        lamb=taubin_lambda,
        nu=taubin_nu,
        iterations=iterations,
    )
    logger.debug(f"Taubin smoothing ({iterations} iters): {time.perf_counter() - t0:.2f}s")

    # 3. Re-rasterize using trimesh voxelization + morphological fill
    t0 = time.perf_counter()
    shape = binary_seg.shape  # (z, y, x)
    voxel_spacing = voxel_size[0]  # isotropic

    vg = mesh.voxelized(pitch=voxel_spacing).fill()

    # Map VoxelGrid back into full-size output array
    result = np.zeros(shape, dtype=bool)

    # vg.transform[:3, 3] = origin in physical (x, y, z) coords
    origin = vg.transform[:3, 3]
    offset_x = int(round(origin[0] / voxel_spacing))
    offset_y = int(round(origin[1] / voxel_spacing))
    offset_z = int(round(origin[2] / voxel_spacing))

    # vg.matrix is indexed as (i, j, k) corresponding to (x, y, z)
    # Transpose to (z, y, x) to match output array
    mat_zyx = vg.matrix.transpose(2, 1, 0)
    sz, sy, sx = mat_zyx.shape

    # Compute overlapping region with bounds clipping
    src_z0 = max(-offset_z, 0)
    dst_z0 = max(offset_z, 0)
    src_y0 = max(-offset_y, 0)
    dst_y0 = max(offset_y, 0)
    src_x0 = max(-offset_x, 0)
    dst_x0 = max(offset_x, 0)
    cz = min(sz - src_z0, shape[0] - dst_z0)
    cy = min(sy - src_y0, shape[1] - dst_y0)
    cx = min(sx - src_x0, shape[2] - dst_x0)

    if cz > 0 and cy > 0 and cx > 0:
        result[dst_z0 : dst_z0 + cz, dst_y0 : dst_y0 + cy, dst_x0 : dst_x0 + cx] = mat_zyx[
            src_z0 : src_z0 + cz,
            src_y0 : src_y0 + cy,
            src_x0 : src_x0 + cx,
        ]

    logger.debug(
        f"Re-rasterize (voxelize+fill): {time.perf_counter() - t0:.2f}s — "
        f"voxel grid shape: {vg.matrix.shape}, offset: ({offset_x}, {offset_y}, {offset_z}), "
        f"output foreground voxels: {np.count_nonzero(result)}",
    )

    return result


def _smooth_membrane_via_mesh(
    membrane_seg: np.ndarray,
    voxel_size: tuple = (1.0, 1.0, 1.0),
    dilation_voxels: int = 3,
    taubin_lambda: float = 0.5,
    taubin_nu: float = -0.53,
    iterations: int = 10,
) -> np.ndarray:
    """
    Smoothing for thin (~1-3 voxel) membrane segmentations.

    Dilates the membrane to ensure marching cubes sees a closed surface,
    applies mesh-based smoothing, then re-thins back to original thickness.

    Args:
        membrane_seg: Binary membrane segmentation array (z, y, x).
        voxel_size: Physical voxel size (z, y, x) in any consistent unit.
        dilation_voxels: Number of voxels to dilate before smoothing.
        taubin_lambda: Positive shrink factor (0 < lambda < 1).
        taubin_nu: Negative inflate factor (nu < -lambda).
        iterations: Number of Taubin lambda/nu steps.

    Returns:
        Smoothed membrane segmentation array with same shape as input.
    """
    import time

    logger.debug(f"Membrane input shape: {membrane_seg.shape}, dilation_voxels: {dilation_voxels}")

    t0 = time.perf_counter()
    struct = generate_binary_structure(3, 1)
    dilated = binary_dilation(membrane_seg, structure=struct, iterations=dilation_voxels)
    logger.debug(
        f"Dilation ({dilation_voxels} iters): {time.perf_counter() - t0:.2f}s — dilated foreground: {np.count_nonzero(dilated)}",
    )

    smoothed_volume = _smooth_via_mesh(
        dilated,
        voxel_size=voxel_size,
        taubin_lambda=taubin_lambda,
        taubin_nu=taubin_nu,
        iterations=iterations,
    )

    # Keep only the surface shell at the original thickness
    t0 = time.perf_counter()
    eroded = binary_erosion(smoothed_volume, structure=struct, iterations=dilation_voxels)
    result = smoothed_volume & ~eroded
    logger.debug(
        f"Erosion + shell extraction: {time.perf_counter() - t0:.2f}s — shell foreground: {np.count_nonzero(result)}",
    )

    return result


# ── Copick wrappers ───────────────────────────────────────────────────────────


def smooth_mesh_segmentation(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    taubin_lambda: float = 0.5,
    taubin_nu: float = -0.53,
    iterations: int = 10,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Smooth a segmentation using Taubin mesh smoothing.

    Args:
        segmentation: Input CopickSegmentation object.
        run: CopickRun object.
        object_name: Name for the output segmentation.
        session_id: Session ID for the output segmentation.
        user_id: User ID for the output segmentation.
        taubin_lambda: Taubin smoothing lambda parameter.
        taubin_nu: Taubin smoothing nu parameter.
        iterations: Number of Taubin smoothing iterations.
        **kwargs: Additional keyword arguments from lazy converter.

    Returns:
        Tuple of (CopickSegmentation, stats dict) or None on failure.
    """
    try:
        seg_array = segmentation.numpy()

        if seg_array is None:
            logger.error("Could not load segmentation data")
            return None

        if seg_array.size == 0:
            logger.error("Empty segmentation data")
            return None

        if segmentation.is_multilabel:
            logger.error("Smoothing does not support multilabel segmentations. Use single-label input.")
            return None

        if not np.any(seg_array):
            logger.warning("Segmentation is entirely empty, skipping")
            return None

        voxel_spacing = segmentation.voxel_size
        voxel_size = (voxel_spacing, voxel_spacing, voxel_spacing)

        voxels_before = int(np.count_nonzero(seg_array))

        result_array = _smooth_via_mesh(
            seg_array.astype(bool),
            voxel_size=voxel_size,
            taubin_lambda=taubin_lambda,
            taubin_nu=taubin_nu,
            iterations=iterations,
        )

        voxels_after = int(np.count_nonzero(result_array))

        output_seg = run.new_segmentation(
            name=object_name,
            user_id=user_id,
            session_id=session_id,
            is_multilabel=False,
            voxel_size=voxel_spacing,
            exist_ok=True,
        )
        output_seg.from_numpy(result_array.astype(seg_array.dtype))

        stats = {
            "voxels_before": voxels_before,
            "voxels_after": voxels_after,
            "voxels_changed": abs(voxels_after - voxels_before),
        }
        logger.info(
            f"Smoothed (mesh): {stats['voxels_before']} -> {stats['voxels_after']} voxels "
            f"(delta={voxels_after - voxels_before}, iterations={iterations})",
        )
        return output_seg, stats

    except Exception as e:
        logger.error(f"Error smoothing segmentation: {e}")
        return None


def smooth_membrane_segmentation(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    taubin_lambda: float = 0.5,
    taubin_nu: float = -0.53,
    iterations: int = 10,
    dilation_voxels: int = 3,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Smooth a thin membrane segmentation using dilate-smooth-thin.

    Args:
        segmentation: Input CopickSegmentation object.
        run: CopickRun object.
        object_name: Name for the output segmentation.
        session_id: Session ID for the output segmentation.
        user_id: User ID for the output segmentation.
        taubin_lambda: Taubin smoothing lambda parameter.
        taubin_nu: Taubin smoothing nu parameter.
        iterations: Number of Taubin smoothing iterations.
        dilation_voxels: Number of voxels to dilate before smoothing.
        **kwargs: Additional keyword arguments from lazy converter.

    Returns:
        Tuple of (CopickSegmentation, stats dict) or None on failure.
    """
    try:
        seg_array = segmentation.numpy()

        if seg_array is None:
            logger.error("Could not load segmentation data")
            return None

        if seg_array.size == 0:
            logger.error("Empty segmentation data")
            return None

        if segmentation.is_multilabel:
            logger.error("Smoothing does not support multilabel segmentations. Use single-label input.")
            return None

        if not np.any(seg_array):
            logger.warning("Segmentation is entirely empty, skipping")
            return None

        voxel_spacing = segmentation.voxel_size
        voxel_size = (voxel_spacing, voxel_spacing, voxel_spacing)

        voxels_before = int(np.count_nonzero(seg_array))

        result_array = _smooth_membrane_via_mesh(
            seg_array.astype(bool),
            voxel_size=voxel_size,
            dilation_voxels=dilation_voxels,
            taubin_lambda=taubin_lambda,
            taubin_nu=taubin_nu,
            iterations=iterations,
        )

        voxels_after = int(np.count_nonzero(result_array))

        output_seg = run.new_segmentation(
            name=object_name,
            user_id=user_id,
            session_id=session_id,
            is_multilabel=False,
            voxel_size=voxel_spacing,
            exist_ok=True,
        )
        output_seg.from_numpy(result_array.astype(seg_array.dtype))

        stats = {
            "voxels_before": voxels_before,
            "voxels_after": voxels_after,
            "voxels_changed": abs(voxels_after - voxels_before),
        }
        logger.info(
            f"Smoothed (membrane): {stats['voxels_before']} -> {stats['voxels_after']} voxels "
            f"(delta={voxels_after - voxels_before}, dilation={dilation_voxels}, iterations={iterations})",
        )
        return output_seg, stats

    except Exception as e:
        logger.error(f"Error smoothing membrane segmentation: {e}")
        return None


# ── Lazy batch converters ─────────────────────────────────────────────────────

smooth_mesh_lazy_batch = create_lazy_batch_converter(
    converter_func=smooth_mesh_segmentation,
    task_description="Smoothing segmentations (mesh)",
)

smooth_membrane_lazy_batch = create_lazy_batch_converter(
    converter_func=smooth_membrane_segmentation,
    task_description="Smoothing segmentations (membrane)",
)
