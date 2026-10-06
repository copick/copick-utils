"""3D spline fitting to skeleton volumes for pick generation with orientations.

``fit-spline`` fits a smoothing spline to every filament of a skeleton (or segmentation) volume and samples picks
along each one, in voxel units. It is built on the filament tracer (``filament_tracing.py``): the skeleton is split
into chains between ends and junctions, continuing straight through junctions, and each chain gets its own spline.
For new work, ``copick convert seg2fil`` and ``copick convert fil2picks`` separate tracing from sampling and work in
angstroms.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from copick.util.log import get_logger
from scipy.interpolate import splev

from copick_utils.converters.lazy_converter import create_lazy_batch_converter
from copick_utils.process.filament_tracing import (
    evaluate_spline,
    fit_spline,
    rotation_minimizing_frames,
    skeleton_chains,
)

if TYPE_CHECKING:
    from copick.models import CopickPicks, CopickRun, CopickSegmentation

logger = get_logger(__name__)

#: Chain-splitting settings for skeleton input, in voxels (the input has no thickness to derive them from).
_PRUNE_VOXELS = 3.0
_MERGE_VOXELS = 3.0
_MAX_BEND = 45.0
_WINDOW_VOXELS = 6.0


class SkeletonSplineFitter:
    """3D spline fitting to skeleton coordinates with point sampling and orientation computation.

    Coordinates are voxel indices: skeleton coordinates in (z, y, x) order, as ``np.argwhere`` returns them, and
    spline points in (x, y, z) order.
    """

    def __init__(self):
        self.spline_tck = None
        self.smoothing = None
        self.skeleton_coords = None
        self.ordered_coords = None
        self.sampled_points = None
        self.t_sampled = None
        self.tangents = None
        self._dense = None

    def extract_skeleton_coordinates(self, binary_volume: np.ndarray) -> np.ndarray:
        """Extract (z, y, x) coordinates of the skeleton voxels."""
        self.skeleton_coords = np.argwhere(binary_volume)
        return self.skeleton_coords

    def order_skeleton_points_longest_path(self, coords: np.ndarray, connectivity_radius: float = 2.0) -> np.ndarray:
        """Order skeleton points along the longest filament chain.

        The skeleton is split into chains between ends and junctions, continuing straight through junctions, and the
        longest chain is returned in order.

        Args:
            coords: (N, 3) skeleton voxel coordinates (z, y, x).
            connectivity_radius: Unused; skeleton voxels are connected to their 26 neighbours.

        Returns:
            (M, 3) ordered coordinates (z, y, x) of the longest chain.
        """
        chains = _chains_from_coords(coords)
        if not chains:
            return np.zeros((0, 3))
        self.ordered_coords = max(chains, key=_chain_length)
        return self.ordered_coords

    def fit_regularized_spline(
        self,
        coords: np.ndarray,
        smoothing_factor: Optional[float] = None,
        degree: int = 3,
    ) -> Any:
        """Fit a smoothing spline to ordered (z, y, x) coordinates.

        Args:
            coords: (N, 3) ordered coordinates (z, y, x).
            smoothing_factor: scipy ``splprep`` smoothing factor ``s`` in voxels squared; None uses scipy's default
                ``m - sqrt(2 m)`` for ``m`` points.
            degree: Spline degree (1-5).

        Returns:
            The spline as scipy's ``(knots, [cx, cy, cz], degree)`` in (x, y, z) voxel coordinates.
        """
        points = np.asarray(coords, dtype=float)[:, ::-1]
        m = len(points)
        self.smoothing = float(m - np.sqrt(2 * m)) if smoothing_factor is None else float(smoothing_factor)
        self.spline_tck = fit_spline(points, s=self.smoothing, degree=degree)
        if self.spline_tck is None:
            raise ValueError("Not enough distinct points to fit a spline")
        self._dense = None
        return self.spline_tck

    def sample_points_along_spline(self, spacing_distance: float) -> np.ndarray:
        """Sample points along the spline at exactly ``spacing_distance`` voxels, from its start.

        Args:
            spacing_distance: Distance between consecutive points along the spline, in voxels.

        Returns:
            (N, 3) sampled points (x, y, z) in voxel coordinates.
        """
        if self.spline_tck is None:
            raise ValueError("Must fit spline first")
        if spacing_distance <= 0:
            raise ValueError("spacing_distance must be positive")
        points, tangents, arc = self._dense_curve()
        along = np.arange(0.0, arc[-1] + 1e-9, spacing_distance)
        self.t_sampled = along
        self.sampled_points = np.column_stack([np.interp(along, arc, points[:, i]) for i in range(3)])
        sampled_tangents = np.column_stack([np.interp(along, arc, tangents[:, i]) for i in range(3)])
        self.tangents = sampled_tangents / np.maximum(np.linalg.norm(sampled_tangents, axis=1, keepdims=True), 1e-12)
        return self.sampled_points

    def compute_transforms(self) -> np.ndarray:
        """Compute 4x4 transformation matrices for each sampled point.

        The rotation's +Z axis is the spline's tangent (its exact derivative) at the point, in the direction of the
        sampling. The rotation about the axis is rotation-minimizing along the spline, starting from the coordinate
        axis least aligned with the first tangent, so it never flips. The translation is 0.

        Returns:
            np.ndarray: [N, 4, 4] array of transformation matrices
        """
        if self.sampled_points is None:
            raise ValueError("Must sample points first before computing transforms")
        points, tangents, arc = self._dense_curve()
        frames = rotation_minimizing_frames(points, tangents)
        nearest = np.clip(np.searchsorted(arc, self.t_sampled), 0, len(arc) - 1)
        transforms = np.tile(np.eye(4), (len(self.sampled_points), 1, 1))
        for i, (k, z) in enumerate(zip(nearest, self.tangents)):
            x = frames[k][:, 0] - np.dot(frames[k][:, 0], z) * z
            x /= np.linalg.norm(x)
            transforms[i, :3, :3] = np.column_stack([x, np.cross(z, x), z])
        return transforms

    def get_spline_properties(self) -> Dict[str, Any]:
        """Properties of the fitted spline: length, sampled point count and spacing, and curvature (1/voxel)."""
        if self.spline_tck is None:
            return {}
        points, _, arc = self._dense_curve()
        knots, coefficients, k = self.spline_tck
        u = np.linspace(knots[k], knots[-k - 1], max(64, len(points)))
        d1 = np.array(splev(u, self.spline_tck, der=1)).T
        d2 = np.array(splev(u, self.spline_tck, der=2)).T if k >= 2 else np.zeros_like(d1)
        speed = np.maximum(np.linalg.norm(d1, axis=1), 1e-12)
        curvature = np.linalg.norm(np.cross(d1, d2), axis=1) / speed**3
        n = len(self.sampled_points) if self.sampled_points is not None else 0
        return {
            "total_length": float(arc[-1]),
            "n_sampled_points": n,
            "average_spacing": float(arc[-1] / (n - 1)) if n > 1 else 0.0,
            "max_curvature": float(curvature.max()) if len(curvature) else 0.0,
            "mean_curvature": float(curvature.mean()) if len(curvature) else 0.0,
            "smoothing_factor": self.smoothing,
        }

    def max_turn(self) -> float:
        """The largest sine of the turning angle between consecutive sampled segments (the --curvature-threshold
        quantity; it depends on the sampling spacing)."""
        if self.sampled_points is None or len(self.sampled_points) < 3:
            return 0.0
        v = np.diff(self.sampled_points, axis=0)
        v1, v2 = v[:-1], v[1:]
        cross = np.linalg.norm(np.cross(v1, v2), axis=1)
        norms = np.maximum(np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-12)
        return float(np.max(cross / norms))

    def detect_high_curvature_outliers(self, coords: np.ndarray, curvature_threshold: float = 0.2) -> np.ndarray:
        """Deprecated: returns ``coords`` unchanged.

        Earlier versions deleted skeleton points around sharp bends, which also removed real curves. Sharp bends are
        now handled by raising the spline's smoothing (``fit_spline_to_skeleton``).
        """
        logger.warning("detect_high_curvature_outliers is deprecated and no longer removes points")
        return coords

    def _dense_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._dense is None:
            self._dense = evaluate_spline(self.spline_tck, step=0.25)
        return self._dense


def _chain_length(coords: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))) if len(coords) > 1 else 0.0


def _chains_from_coords(coords: np.ndarray) -> List[np.ndarray]:
    """Ordered (z, y, x) chains of a skeleton given as voxel coordinates."""
    coords = np.asarray(coords, dtype=int)
    if len(coords) == 0:
        return []
    low = coords.min(axis=0)
    volume = np.zeros(tuple(coords.max(axis=0) - low + 1), dtype=bool)
    volume[tuple((coords - low).T)] = True
    return [chain.coords + low for chain in _skeleton_chains(volume)]


def _skeleton_chains(volume: np.ndarray):
    from skimage.morphology import skeletonize

    skeleton = skeletonize(np.pad(np.asarray(volume, dtype=bool), 1))[1:-1, 1:-1, 1:-1]
    if not skeleton.any():
        skeleton = np.asarray(volume, dtype=bool)  # already as thin as it gets
    chains, _ = skeleton_chains(
        skeleton,
        prune_length=_PRUNE_VOXELS,
        junction_merge=_MERGE_VOXELS,
        max_bend=_MAX_BEND,
        direction_window=_WINDOW_VOXELS,
    )
    return chains


def _fit_chain(
    coords: np.ndarray,
    spacing_distance: float,
    smoothing_factor: Optional[float],
    degree: int,
    compute_transforms: bool,
    curvature_threshold: float,
    max_iterations: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], SkeletonSplineFitter, Dict[str, Any]]:
    """Fit one ordered chain; if the sampled curve turns more sharply than ``curvature_threshold``, raise the
    smoothing (doubling it, up to ``max_iterations`` times) instead of removing skeleton points."""
    fitter = SkeletonSplineFitter()
    fitter.ordered_coords = coords
    fitter.fit_regularized_spline(coords, smoothing_factor=smoothing_factor, degree=degree)
    fitter.sample_points_along_spline(spacing_distance)
    smoothing = max(fitter.smoothing, float(len(coords)))
    for _ in range(max_iterations):
        if fitter.max_turn() <= curvature_threshold:
            break
        smoothing *= 2.0
        fitter.fit_regularized_spline(coords, smoothing_factor=smoothing, degree=degree)
        fitter.sample_points_along_spline(spacing_distance)
    transforms = fitter.compute_transforms() if compute_transforms else None
    return fitter.sampled_points, transforms, fitter, fitter.get_spline_properties()


def fit_spline_to_skeleton(
    binary_volume: np.ndarray,
    spacing_distance: float,
    smoothing_factor: Optional[float] = None,
    degree: int = 3,
    connectivity_radius: float = 2.0,
    compute_transforms: bool = True,
    curvature_threshold: float = 0.2,
    max_iterations: int = 5,
) -> Tuple[np.ndarray, Optional[np.ndarray], SkeletonSplineFitter, Dict[str, Any]]:
    """
    Fit a smoothing spline to the longest filament of a skeleton and sample points along it.

    Args:
        binary_volume: 3D binary volume where skeleton is True/1
        spacing_distance: Distance between consecutive sampled points along the spline, in voxels
        smoothing_factor: Smoothing parameter for spline fitting (scipy's ``s``, voxels squared; auto if None)
        degree: Degree of the spline (1-5)
        connectivity_radius: Unused; skeleton voxels are connected to their 26 neighbours
        compute_transforms: Whether to compute 4x4 transformation matrices for each point
        curvature_threshold: Largest sine of the turning angle between consecutive samples; a sharper fit is
            smoothed further
        max_iterations: Maximum number of smoothing increases

    Returns:
        Tuple of (sampled_points, transforms, spline_fitter, properties):
            - sampled_points: Nx3 array of points (x, y, z, voxels) exactly ``spacing_distance`` apart
            - transforms: [N, 4, 4] array of transformation matrices (or None if compute_transforms=False)
            - spline_fitter: SkeletonSplineFitter object for further analysis
            - properties: dict with spline properties
    """
    chains = _skeleton_chains(binary_volume)
    if not chains:
        raise ValueError("No skeleton points found in binary volume")
    coords = max((chain.coords for chain in chains), key=_chain_length)
    if len(coords) < 2:
        raise ValueError("Not enough ordered points for spline fitting")
    return _fit_chain(
        coords,
        spacing_distance,
        smoothing_factor,
        degree,
        compute_transforms,
        curvature_threshold,
        max_iterations,
    )


def fit_spline_to_segmentation(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    spacing_distance: float,
    smoothing_factor: Optional[float] = None,
    degree: int = 3,
    connectivity_radius: float = 2.0,
    compute_transforms: bool = True,
    curvature_threshold: float = 0.2,
    max_iterations: int = 5,
    voxel_spacing: float = 1.0,
    label: Optional[int] = None,
    filaments_object_name: Optional[str] = None,
    filaments_user_id: Optional[str] = None,
    filaments_session_id: Optional[str] = None,
    **kwargs,
) -> Optional[Tuple["CopickPicks", Dict[str, int]]]:
    """
    Fit a spline to every filament of a segmentation (skeleton) volume and create picks with orientations.

    Matches the lazy converter signature:
        (segmentation, run, object_name, session_id, user_id, **tool_kwargs)

    Every filament gets its own spline. The picks of all filaments go into one pick set, grouped by filament and in
    order along it, with the filament's ID (1..K by length, longest first) as ``instance_id``.

    Args:
        segmentation: Input segmentation containing skeleton to fit spline to
        run: CopickRun object
        object_name: Name for the output pick object
        session_id: Session ID for output picks
        user_id: User ID for output picks
        spacing_distance: Distance between consecutive sampled points along the spline, in voxels
        smoothing_factor: Smoothing parameter for spline fitting (scipy's ``s``, voxels squared; auto if None)
        degree: Degree of the spline (1-5)
        connectivity_radius: Unused; skeleton voxels are connected to their 26 neighbours
        compute_transforms: Whether to compute orientations for picks
        curvature_threshold: Largest sine of the turning angle between consecutive samples
        max_iterations: Maximum number of smoothing increases
        voxel_spacing: Voxel spacing for coordinate scaling (the segmentation's own voxel size is used when known)
        label: Label to fit in a multilabel segmentation (default: every non-zero voxel)
        filaments_object_name: Also store the fitted splines as copick Filaments under this object name
        filaments_user_id: User ID of the Filaments
        filaments_session_id: Session ID of the Filaments ({input_session_id} is replaced)
        **kwargs: Additional keyword arguments from the lazy converter

    Returns:
        Tuple of (CopickPicks object, stats dict) or None if failed.
        Stats dict contains 'picks_created' and 'filaments_fitted'.
    """
    volume = segmentation.numpy()
    if volume is None:
        logger.error(f"Could not load segmentation data for {run.name}")
        return None

    logger.info(f"Fitting splines to segmentation {segmentation.session_id} in run {run.name}")

    try:
        voxel_spacing = float(segmentation.voxel_size or voxel_spacing)
        mask = volume == label if label is not None else volume > 0
        chains = [chain.coords for chain in _skeleton_chains(mask)]
        chains = sorted((c for c in chains if len(c) >= 2), key=_chain_length, reverse=True)

        positions, transforms, ids, curves = [], [], [], []
        for instance_id, coords in enumerate(chains, start=1):
            points, rotations, fitter, _ = _fit_chain(
                _start_at_smaller_end(coords),
                spacing_distance,
                smoothing_factor,
                degree,
                True,
                curvature_threshold,
                max_iterations,
            )
            positions.append(points * voxel_spacing)
            transforms.append(rotations if compute_transforms else np.tile(np.eye(4), (len(points), 1, 1)))
            ids.append(np.full(len(points), instance_id, dtype=np.int64))
            curves.append((fitter.spline_tck, fitter.smoothing))

        output_picks = run.new_picks(object_name=object_name, session_id=session_id, user_id=user_id, exist_ok=True)
        if positions:
            output_picks.from_numpy(
                np.concatenate(positions),
                np.concatenate(transforms),
                instance_ids=np.concatenate(ids),
            )
        else:
            output_picks.points = []
            output_picks.store()

        if filaments_object_name:
            _store_filaments(
                run,
                segmentation,
                curves,
                voxel_spacing,
                filaments_object_name,
                filaments_user_id or user_id,
                (filaments_session_id or session_id).replace("{input_session_id}", segmentation.session_id),
                {
                    "spacing_distance": spacing_distance,
                    "smoothing_factor": smoothing_factor,
                    "degree": degree,
                    "curvature_threshold": curvature_threshold,
                    "max_iterations": max_iterations,
                },
            )

        stats = {"picks_created": int(sum(len(p) for p in positions)), "filaments_fitted": len(positions)}
        logger.info(
            f"Created {stats['picks_created']} picks on {stats['filaments_fitted']} filaments "
            f"with session_id: {session_id}",
        )
        return output_picks, stats

    except Exception as e:
        logger.error(f"Error fitting spline to segmentation: {e}")
        return None


def _start_at_smaller_end(coords: np.ndarray) -> np.ndarray:
    """Order a chain so that it starts at the end with the smaller (z, y, x) voxel."""
    start, end = np.round(coords[0]).astype(int), np.round(coords[-1]).astype(int)
    return coords[::-1] if tuple(end) < tuple(start) else coords


def _store_filaments(run, segmentation, curves, voxel_spacing, object_name, user_id, session_id, flags) -> None:
    """Store fitted splines (voxel coordinates) as copick Filaments with ``bspline`` curves."""
    from copick.models import CopickFilamentCurve

    import copick_utils

    source = f"{segmentation.name}:{segmentation.user_id}/{segmentation.session_id}@{voxel_spacing}"
    fitted = [
        CopickFilamentCurve.from_tck(tck, step=voxel_spacing, smoothing=smoothing, scale=voxel_spacing)
        for tck, smoothing in curves
    ]
    metadata = [
        {
            "fit": {
                "method": "splprep",
                "coordinates": "voxels",
                "tool": "copick-utils fit-spline",
                "version": copick_utils.__version__,
                "source": source,
                **flags,
            },
        }
        for _ in curves
    ]
    filaments = run.new_filaments(object_name, session_id, user_id, exist_ok=True)
    filaments.from_curves(fitted, voxel_spacing=voxel_spacing, metadata=metadata)


# Lazy batch converter for the lazy task discovery architecture
fit_spline_lazy_batch = create_lazy_batch_converter(
    converter_func=fit_spline_to_segmentation,
    task_description="Fitting splines to segmentations",
)
