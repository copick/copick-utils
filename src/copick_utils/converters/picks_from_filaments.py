"""Sample picks along filaments at a fixed spacing, oriented along the filament and carrying its ID."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
from copick.util.log import get_logger

from copick_utils.converters.lazy_converter import create_lazy_batch_converter
from copick_utils.process.filament_tracing import sample_curve

if TYPE_CHECKING:
    from copick.models import CopickFilaments, CopickPicks, CopickRun

logger = get_logger(__name__)


def picks_from_filaments(
    filaments: "CopickFilaments",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    spacing: float,
    anchor: str = "center",
    roll: str = "parallel",
    seed: Optional[int] = None,
    length_unit: str = "angstrom",
    **kwargs,
) -> Optional[Tuple["CopickPicks", Dict[str, int]]]:
    """
    Sample each filament at a fixed spacing along its curve.

    A filament's stored curve is used when it still describes the filament (a traced ``bspline``, or ``catmull-rom``
    control points from an editor); otherwise copick derives a curve from its points. Each pick's rotation has its +Z
    axis along the filament in point order, its translation is 0, and its ``instance_id`` and ``score`` are the
    filament's. Picks are written grouped by filament and in order along it.

    Args:
        filaments: Input CopickFilaments.
        run: CopickRun object.
        object_name: Object name of the output picks.
        session_id: Session ID of the output picks.
        user_id: User ID of the output picks.
        spacing: Distance between picks along a filament, in Angstrom. Required: no spacing suits every filament.
        anchor: ``center`` or ``start`` (see ``sample_curve``).
        roll: ``parallel`` or ``random`` (see ``sample_curve``).
        seed: Random seed for ``roll="random"`` (combined with each filament's ID).
        length_unit: ``angstrom``, or ``voxel`` for a spacing in voxels of the filaments' voxel spacing.
        **kwargs: Additional keyword arguments from the lazy converter.

    Returns:
        Tuple of (CopickPicks, stats dict) or None if the operation failed.
    """
    try:
        if spacing is None or spacing <= 0:
            raise ValueError("A positive spacing is required.")
        if length_unit == "voxel":
            if not filaments.voxel_spacing:
                raise ValueError("A spacing in voxels needs the filaments' voxel spacing, which this file lacks.")
            spacing = spacing * float(filaments.voxel_spacing)
        positions, transforms, ids, scores = [], [], [], []
        for filament in filaments.filaments:
            curve = filament.editable_curve().model_dump()
            filament_seed = None if seed is None else int(seed) + int(filament.instance_id)
            p, r, _ = sample_curve(curve, spacing, anchor=anchor, roll=roll, seed=filament_seed)
            t = np.tile(np.eye(4), (len(p), 1, 1))
            t[:, :3, :3] = r
            positions.append(p)
            transforms.append(t)
            ids.append(np.full(len(p), filament.instance_id, dtype=np.int64))
            scores.append(np.full(len(p), filament.score, dtype=float))

        picks = run.new_picks(object_name, session_id, user_id, exist_ok=True)
        if positions:
            picks.from_numpy(
                np.concatenate(positions),
                np.concatenate(transforms),
                instance_ids=np.concatenate(ids),
                scores=np.concatenate(scores),
            )
        else:
            picks.points = []
            picks.store()

        stats = {"filaments_sampled": len(positions), "points_created": int(sum(len(p) for p in positions))}
        logger.info(
            f"Sampled {stats['points_created']} picks from {stats['filaments_sampled']} filaments "
            f"every {spacing} Å in {run.name}",
        )
        return picks, stats

    except Exception as e:
        logger.error(f"Error sampling filaments in {run.name}: {e}")
        return None


picks_from_filaments_lazy_batch = create_lazy_batch_converter(
    converter_func=picks_from_filaments,
    task_description="Sampling filaments",
)
