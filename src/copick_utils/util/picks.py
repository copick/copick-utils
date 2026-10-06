"""Helpers that keep a pick's identity intact while tools filter, paint or measure picks.

A pick's particle centre is its ``location`` plus the translation of its ``transformation`` (copick's geometry
documentation, "Shifts"). Tools that place or test particles use that centre, not ``location`` alone.

Tools that keep a subset of picks copy the surviving points themselves instead of rebuilding them from arrays, so each
point keeps its transform (rotation and shift), ``instance_id`` and ``score``, and the points keep their order. For
filament objects the order is the order along the filament, and ``instance_id`` is the filament.
"""

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from copick.models import CopickPicks, CopickPoint, CopickRun, CopickSegmentation


def point_centres(points: Sequence["CopickPoint"]) -> np.ndarray:
    """The particle centres of picks, in Angstrom.

    Args:
        points: Copick points.

    Returns:
        (N, 3) array of ``location + transformation[:3, 3]`` in (x, y, z) order.
    """
    if len(points) == 0:
        return np.zeros((0, 3), dtype=float)
    locations = np.array([[p.location.x, p.location.y, p.location.z] for p in points], dtype=float)
    shifts = np.array([np.asarray(p.transformation, dtype=float)[:3, 3] for p in points], dtype=float)
    return locations + shifts


def pick_centres(picks: "CopickPicks") -> np.ndarray:
    """The particle centres of a pick set, in Angstrom.

    Args:
        picks: Copick pick set.

    Returns:
        (N, 3) array of ``location + transformation[:3, 3]`` in (x, y, z) order.
    """
    return point_centres(picks.points or [])


def store_pick_subset(
    points: Sequence["CopickPoint"],
    keep: np.ndarray,
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
) -> "CopickPicks":
    """Store the kept points, unchanged and in their original order, as a pick set.

    An empty selection stores an empty pick set, so an earlier result in the same place does not survive a run that
    kept nothing.

    Args:
        points: The input points.
        keep: Boolean mask over ``points``.
        run: Run to write into.
        object_name: Object name of the output pick set.
        session_id: Session ID of the output pick set.
        user_id: User ID of the output pick set.

    Returns:
        The stored pick set.
    """
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (len(points),):
        raise ValueError(f"The selection has shape {keep.shape}, but there are {len(points)} points.")
    kept = [p.model_copy(deep=True) for p, k in zip(points, keep) if k]
    output = run.new_picks(object_name, session_id, user_id, exist_ok=True)
    output.points = kept
    output.store()
    return output


def segmentation_label_volume(segmentation: "CopickSegmentation") -> np.ndarray:
    """The volume a tool tests positions against: the label channel of a panoptic segmentation, otherwise the
    segmentation itself (binary, multilabel or instance; non-zero means inside).

    Args:
        segmentation: Copick segmentation.

    Returns:
        (Z, Y, X) array.
    """
    if getattr(segmentation, "is_panoptic", False):
        return segmentation.numpy(channel="label")
    return segmentation.numpy()


def voxel_indices(positions: np.ndarray, voxel_spacing: float) -> np.ndarray:
    """The (z, y, x) indices of the voxels holding positions, under copick's mapping of voxel ``i`` to ``i * s``.

    Args:
        positions: (N, 3) positions in Angstrom, (x, y, z) order.
        voxel_spacing: Voxel spacing in Angstrom.

    Returns:
        (N, 3) integer array in (z, y, x) order.
    """
    return np.round(np.asarray(positions, dtype=float) / float(voxel_spacing)).astype(int)[:, ::-1]
