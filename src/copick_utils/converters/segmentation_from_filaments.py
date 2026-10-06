"""Paint tubes around filament centrelines: an instance segmentation whose IDs are the filaments' IDs."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import zarr
from copick.util.log import get_logger

from copick_utils.converters.lazy_converter import create_lazy_batch_converter
from copick_utils.process.filament_tracing import paint_tubes

if TYPE_CHECKING:
    from copick.models import CopickFilaments, CopickRun, CopickSegmentation

logger = get_logger(__name__)


def _densify(points: np.ndarray, step: float) -> np.ndarray:
    """Insert points along each segment so consecutive points are at most ``step`` apart."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points
    out = [points[:1]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    return np.concatenate(out)


def segmentation_from_filaments(
    filaments: "CopickFilaments",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    voxel_spacing: float,
    tomo_type: str = "wbp",
    radius: Optional[float] = None,
    output_segmentation_type: Optional[str] = None,
    **kwargs,
) -> Optional[Tuple["CopickSegmentation", Dict[str, int]]]:
    """
    Paint a tube around each filament into an instance segmentation.

    Each filament's tube radius is its own ``radius`` if it has one, otherwise ``radius``, otherwise the object's
    radius. Voxels inside a tube hold the filament's ID; where tubes overlap, the nearest centreline wins. The
    volume's shape is that of the run's ``tomo_type`` tomogram at ``voxel_spacing``.

    Args:
        filaments: Input CopickFilaments.
        run: CopickRun object.
        object_name: Object name of the output segmentation.
        session_id: Session ID of the output segmentation.
        user_id: User ID of the output segmentation.
        voxel_spacing: Voxel spacing of the output segmentation, in Angstrom.
        tomo_type: Tomogram type whose shape the output takes.
        radius: Tube radius in Angstrom for filaments without their own radius.
        output_segmentation_type: The type the output URI names; only an instance segmentation can be written.
        **kwargs: Additional keyword arguments from the lazy converter.

    Returns:
        Tuple of (CopickSegmentation, stats dict) or None if the operation failed.
    """
    from copick_utils.util.segmentations import new_segmentation

    try:
        if output_segmentation_type not in (None, "instance"):
            raise ValueError(f"fil2seg writes instance segmentations, not {output_segmentation_type} ones.")
        voxel_spacing = float(voxel_spacing)
        vs = run.get_voxel_spacing(voxel_spacing)
        tomograms = vs.get_tomograms(tomo_type) if vs is not None else []
        tomogram = tomograms[0] if tomograms else None
        if tomogram is None:
            raise ValueError(f"No {tomo_type} tomogram at {voxel_spacing} Å in {run.name}.")
        shape = zarr.open(tomogram.zarr(), "r")["0"].shape

        obj = run.root.get_object(filaments.pickable_object_name)
        fallback = radius if radius is not None else (obj.radius if obj is not None else None)
        polylines, ids, radii = [], [], []
        for filament in filaments.filaments:
            tube = filament.radius if filament.radius is not None else fallback
            if tube is None:
                raise ValueError(
                    f"Filament {filament.instance_id} has no radius; give one with --radius.",
                )
            polylines.append(_densify(np.asarray(filament.points, dtype=float), voxel_spacing / 2.0))
            ids.append(int(filament.instance_id))
            radii.append(float(tube))

        volume = paint_tubes(polylines, ids, radii, shape, voxel_spacing)
        output = new_segmentation(run, voxel_spacing, object_name, session_id, user_id, "instance")
        output.from_numpy(volume)
        stats = {"filaments_painted": len(ids), "voxels_painted": int(np.count_nonzero(volume))}
        logger.info(f"Painted {len(ids)} filament tubes ({stats['voxels_painted']} voxels) in {run.name}")
        return output, stats

    except Exception as e:
        logger.error(f"Error painting filaments in {run.name}: {e}")
        return None


segmentation_from_filaments_lazy_batch = create_lazy_batch_converter(
    converter_func=segmentation_from_filaments,
    task_description="Painting filaments",
)
