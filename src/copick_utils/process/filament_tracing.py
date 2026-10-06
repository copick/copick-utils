"""Filament tracing: centrelines from segmentations, picks along centrelines, and tube masks around them.

The functions here work on numpy arrays and do not read or write copick objects; the converters
(``converters/filaments_from_segmentation.py``, ``picks_from_filaments.py``, ``segmentation_from_filaments.py``) and
``fit-spline`` call them. Curves are evaluated with copick's reference implementation (``copick.util.filaments``), so
a curve is sampled the same way copick regenerates its points.

Conventions (copick ``docs/geometry.md``):

- Volumes are indexed (z, y, x); voxel ``i`` sits at ``i * voxel_size`` Angstrom. Positions are (x, y, z) Angstrom.
- A filament's points are ordered along it; its picks carry its ID as ``instance_id``, and each pick's rotation has
  its +Z axis along the filament (the tangent, in point order). The rotation about the axis (roll) is documented by
  ``sample_curve``.

Tracing a semantic segmentation (``trace_centrelines``):

1. Connected components (26-connected) smaller than ``min_volume`` are dropped.
2. Each remaining component is skeletonized (Lee) and turned into a skeleton graph (``skeleton_graph.py``); spurs
   shorter than ``prune_length`` are pruned, and junctions joined by a bridge shorter than ``junction_merge`` are
   merged (two crossing filaments often skeletonize as two forks joined by a short bridge).
3. At every junction, branches are paired by straightest continuation: the pair whose directions deviate least from
   a straight line is joined first, as long as the deviation is at most ``max_bend`` degrees. A crossing therefore
   gives two filaments, and a Y gives one filament through the junction and one that ends there.
4. Paired branches are chained into filaments. At a free end, the last label radius of skeleton is replaced by the
   continuation of the filament's direction up to the edge of the label (thinning shortens a rod's skeleton by about
   one radius at each end and often bends its last voxels).
5. Filaments shorter than ``min_length``, shorter than ``min_aspect`` label diameters (blobs), or thinner than
   ``min_radius`` (slivers of segmentation noise, much thinner than the real filaments) are rejected.
6. Each filament is fitted with a smoothing B-spline (scipy ``splprep``) whose RMS deviation from the skeleton is
   about ``smoothing`` Angstrom. The fit is returned exactly (``tck``), so copick can store it as a ``bspline`` curve.
7. Every label voxel is assigned to the nearest filament of its component, which gives the instance segmentation.

Tracing an instance segmentation traces each instance on its own and keeps its ID; an instance that branches keeps
its longest chain.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

from copick_utils.process.skeleton_graph import (
    Topology,
    decompose,
    merge_junctions,
    node_centre,
    prune_spurs,
    skeleton_graph,
    splice,
)

#: Version of the tracing algorithm, recorded in each filament's ``metadata["fit"]``.
TRACER_VERSION = 1


@dataclass
class TraceParameters:
    """Tracing parameters. Lengths are in Angstrom and volumes in cubic Angstrom; ``None`` derives a value from the
    label itself, as noted, so the defaults carry over between objects and voxel sizes.

    Attributes:
        min_volume: Drop connected components smaller than this. None: no volume filter.
        min_length: Reject filaments shorter than this. None: no length filter.
        min_aspect: Reject filaments shorter than this many label diameters (compact blobs). Basis: on dataset 10521
            (easymode microtubule labels, 58 tomograms) every microtubule-thick piece this rejects is under 44 nm,
            and every piece of 46 nm or more passes.
        min_radius: Reject filaments whose median label radius is below this (thin slivers of segmentation noise).
            None: no radius filter. Basis: on dataset 10521, real microtubules have a label radius of 40-76 Å, while
            slivers and stubs of segmentation noise have 10-35 Å; the converters default to a third of the object's
            tube radius (40 Å there).
        prune_length: Prune skeleton spurs shorter than this. None: one label diameter.
        junction_merge: Merge junctions joined by a bridge up to this long. None: one label diameter.
        max_bend: Largest deviation from a straight line, in degrees, for two branches to continue through a junction.
        smoothing: RMS deviation of the fitted spline from the skeleton. None: half a voxel.
        extend_ends: Extend free ends along their direction to the edge of the label.
        direction_window: Length over which a branch's direction at a junction is measured. None: two label
            diameters.
        fill_lumen: Before skeletonizing, fill holes up to this radius in every axis-aligned slice of the label. A
            label of a tube's wall alone (an empty lumen) otherwise skeletonizes to a mesh over the wall. None: no
            filling; the converters default to the object's tube radius. Basis: on dataset 10521, ten tomograms have
            stretches of hollow label; without filling, three of them traced no microtubule at all.
    """

    min_volume: Optional[float] = None
    min_length: Optional[float] = None
    min_aspect: float = 3.0
    min_radius: Optional[float] = None
    prune_length: Optional[float] = None
    junction_merge: Optional[float] = None
    max_bend: float = 45.0
    smoothing: Optional[float] = None
    extend_ends: bool = True
    direction_window: Optional[float] = None
    fill_lumen: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        """The parameters as a plain dict (for ``metadata["fit"]``)."""
        return dict(self.__dict__)


@dataclass
class Centreline:
    """A traced filament.

    Attributes:
        instance_id: The filament's ID (>= 1).
        tck: The fitted spline as scipy's ``(knots, [cx, cy, cz], degree)``, in Angstrom (x, y, z).
        smoothing: The ``splprep`` smoothing factor ``s`` of the fit (Angstrom squared).
        points: Dense points along the fit, (N, 3) Angstrom (x, y, z), at most one voxel apart.
        length: Length along the fit, in Angstrom.
        radius: Median label radius along the skeleton (distance to the background), in Angstrom.
        score: Fraction of the fitted centreline that lies inside the label.
        component: The connected component (or input instance ID) the filament came from.
        metadata: Details for ``metadata["fit"]`` (junctions, extended ends, closed, ...).
    """

    instance_id: int
    tck: Tuple[np.ndarray, List[np.ndarray], int]
    smoothing: float
    points: np.ndarray
    length: float
    radius: Optional[float]
    score: float
    component: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceReport:
    """Counts from one tracing run."""

    components: int = 0
    components_removed_volume: int = 0
    filaments: int = 0
    rejected_length: int = 0
    rejected_aspect: int = 0
    rejected_radius: int = 0
    junctions_resolved: int = 0
    total_length: float = 0.0
    dropped_branch_length: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        """The counts as a dict (numeric, so batch runs can sum them)."""
        return dict(self.__dict__)


# ---------------------------------------------------------------------------------------------------------------------
# Skeleton chains
# ---------------------------------------------------------------------------------------------------------------------


def _branch_end(topology: Topology, branch_index: int, at_start: bool) -> np.ndarray:
    """The voxel position (z, y, x) where a branch meets its node."""
    branch = topology.branches[branch_index]
    return topology.graph.coords[branch.path[0] if at_start else branch.path[-1]].astype(float)


def _end_direction(topology: Topology, branch_index: int, at_start: bool, window: float) -> np.ndarray:
    """Unit direction (z, y, x) pointing from a junction into a branch, measured over ``window`` voxels of path from
    the branch's own end (not the junction's centre, which for a cluster of merged junctions can lie far from it)."""
    branch = topology.branches[branch_index]
    path = branch.path if at_start else branch.path[::-1]
    coords = topology.graph.coords[path].astype(float)
    steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    reach = np.concatenate([[0.0], np.cumsum(steps)])
    far = coords[min(int(np.searchsorted(reach, window)), len(coords) - 1)]
    direction = far - coords[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        direction = far - node_centre(topology, branch.start if at_start else branch.end)
        norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-9 else np.zeros(3)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two unit vectors."""
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _pair_junction(
    topology: Topology,
    node: int,
    max_bend: float,
    window: float,
    gap_tolerance: float = 2.0,
) -> List[Tuple[Tuple[int, bool], Tuple[int, bool]]]:
    """Pair the branch ends at a junction by straightest continuation, greedily, within ``max_bend`` degrees.

    Continuing from one branch into another means arriving along the first, crossing the junction from its end to
    the second's, and leaving along the second. Where the two ends are more than ``gap_tolerance`` voxels apart (a
    cluster of merged junctions), the crossing has to continue straight as well, so a pair that turns back inside
    the junction is never chosen.
    """
    ends = topology.incident(node)
    directions = [_end_direction(topology, b, at_start, window) for b, at_start in ends]
    positions = [_branch_end(topology, b, at_start) for b, at_start in ends]
    candidates = []
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            arrive, leave = -directions[i], directions[j]  # symmetric: reversing both gives the same bend
            bend = _angle(arrive, leave)
            gap = positions[j] - positions[i]
            distance = np.linalg.norm(gap)
            if distance > gap_tolerance:
                gap = gap / distance
                bend = max(bend, _angle(arrive, gap), _angle(gap, leave))
            candidates.append((bend, i, j))
    pairs, used = [], set()
    for bend, i, j in sorted(candidates):
        if bend > max_bend:
            break
        if i in used or j in used:
            continue
        used.update((i, j))
        pairs.append((ends[i], ends[j]))
    return pairs


@dataclass
class Chain:
    """An ordered skeleton path through one or more branches.

    Attributes:
        coords: (N, 3) voxel positions (z, y, x), in order.
        free_ends: Whether the chain's start and end are free (endpoints), as opposed to ending at a junction.
        closed: Whether the chain closes on itself.
        junctions: Number of junctions the chain passes through.
    """

    coords: np.ndarray
    free_ends: Tuple[bool, bool]
    closed: bool = False
    junctions: int = 0


def skeleton_chains(
    skeleton: np.ndarray,
    prune_length: float,
    junction_merge: float,
    max_bend: float,
    direction_window: float,
    junction_trim: float = 0.0,
) -> Tuple[List[Chain], int]:
    """Split a skeleton into filament chains, continuing straight through junctions.

    Args:
        skeleton: 3D boolean skeleton (one connected component, or several).
        prune_length: Prune spurs shorter than this many voxels.
        junction_merge: Merge junctions joined by a bridge up to this many voxels.
        max_bend: Largest deviation from straight, in degrees, to continue through a junction.
        direction_window: Voxels of path over which a branch's direction at a junction is measured.
        junction_trim: Drop this many voxels of path where a branch meets a junction. Thinning bends each branch
            into the junction's centre; without those voxels, a chain through a junction joins its branches directly,
            and a chain that ends at a junction ends straight.

    Returns:
        Tuple of (chains, number of junctions resolved).
    """
    topology = splice(decompose(skeleton_graph(skeleton)))
    if prune_length > 0:
        topology = prune_spurs(topology, prune_length)
    if junction_merge > 0:
        topology = splice(merge_junctions(topology, junction_merge))

    # Links between branch ends that continue through a junction
    link: Dict[Tuple[int, bool], Tuple[int, bool]] = {}
    junctions = topology.junctions()
    for node in junctions:
        for a, b in _pair_junction(topology, node, max_bend, direction_window):
            link[a] = b
            link[b] = a

    coords = topology.graph.coords
    visited = set()
    chains: List[Chain] = []

    def doubles_back(parts: List[np.ndarray], branch: int, at_start: bool) -> bool:
        """Whether a branch, beyond the junction it starts from, runs back along the chain so far."""
        if not parts or junction_trim <= 0:
            return False
        path = coords[topology.branches[branch].path if at_start else topology.branches[branch].path[::-1]]
        steps = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        beyond = path[steps > 4.0 * junction_trim].astype(float)
        if len(beyond) == 0:
            return False
        earlier = np.concatenate(parts)
        steps_back = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(earlier[::-1], axis=0), axis=1))])
        earlier = earlier[::-1][steps_back > 4.0 * junction_trim]
        if len(earlier) == 0:
            return False
        distance, _ = cKDTree(earlier).query(beyond)
        return bool(np.any(distance < 2.0 * junction_trim))

    def follow(start_branch: int, enter_at_start: bool) -> Chain:
        parts, n_junctions = [], 0
        branch, at_start = start_branch, enter_at_start
        first_end = (branch, at_start)
        closed = False
        while True:
            visited.add(branch)
            current = topology.branches[branch]
            path = coords[current.path if at_start else current.path[::-1]].astype(float)
            entry_node = current.start if at_start else current.end
            exit_node = current.end if at_start else current.start
            if junction_trim > 0 and topology.node_kind.get(entry_node) == "junction":
                path = _trim(path, junction_trim, at_start=True)
            if junction_trim > 0 and topology.node_kind.get(exit_node) == "junction":
                path = _trim(path, junction_trim, at_start=False)
            parts.append(path)
            exit_end = (branch, not at_start)
            if exit_end not in link:
                break
            nxt = link[exit_end]
            if nxt == first_end or nxt[0] in visited:
                closed = nxt == first_end
                n_junctions += int(closed)
                break
            if doubles_back(parts, *nxt):
                break
            n_junctions += 1
            branch, at_start = nxt
        free_start = first_end not in link
        free_end = not closed and (branch, not at_start) not in link
        start_kind = topology.node_kind.get(
            topology.branches[first_end[0]].start if first_end[1] else topology.branches[first_end[0]].end,
        )
        last = topology.branches[branch]
        end_kind = topology.node_kind.get(last.end if at_start else last.start)
        return Chain(
            coords=np.concatenate(parts),
            free_ends=(free_start and start_kind == "endpoint", free_end and end_kind == "endpoint"),
            closed=closed or (len(parts) == 1 and topology.branches[first_end[0]].closed),
            junctions=n_junctions,
        )

    # Open chains start at an end that continues nowhere
    for b in range(len(topology.branches)):
        if b in visited:
            continue
        for at_start in (True, False):
            if (b, at_start) not in link:
                chains.append(follow(b, at_start))
                break
    # What is left are closed loops of linked branches
    for b in range(len(topology.branches)):
        if b not in visited:
            chains.append(follow(b, True))
    return chains, len(link) // 2


# ---------------------------------------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------------------------------------


def _dedupe(points: np.ndarray) -> np.ndarray:
    """Drop consecutive duplicate points."""
    if len(points) < 2:
        return points
    keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9])
    return points[keep]


def fit_spline(points: np.ndarray, s: float, degree: int = 3) -> Optional[Tuple[np.ndarray, List[np.ndarray], int]]:
    """A smoothing B-spline through ordered points (scipy ``splprep``, chord-length parameterisation).

    Args:
        points: (N, 3) ordered points.
        s: ``splprep`` smoothing factor (sum of squared deviations, in the points' units squared).
        degree: Spline degree (lowered for very short inputs).

    Returns:
        ``(knots, [cx, cy, cz], degree)`` with clamped knots, or None for fewer than two distinct points.
    """
    points = _dedupe(np.asarray(points, dtype=float))
    if len(points) < 2:
        return None
    k = int(min(degree, len(points) - 1))
    chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    u = chord / chord[-1]
    tck, _ = splprep(points.T, u=u, s=float(s), k=k)
    knots, coefficients, k = tck
    count = len(knots) - int(k) - 1  # scipy may pad the coefficient arrays past the spline's own count
    return np.asarray(knots, dtype=float), [np.asarray(c, dtype=float)[:count] for c in coefficients], int(k)


def evaluate_spline(
    tck: Tuple[np.ndarray, List[np.ndarray], int],
    step: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense points, unit tangents and arc length along a B-spline.

    Args:
        tck: ``(knots, [cx, cy, cz], degree)``.
        step: Largest spacing of the returned points (the spline's units).

    Returns:
        Tuple of (points (N, 3), tangents (N, 3), arc length at each point (N,)).
    """
    knots, coefficients, k = tck
    u0, u1 = knots[k], knots[-k - 1]
    coarse = np.array(splev(np.linspace(u0, u1, 64), (knots, coefficients, k))).T
    rough = float(np.sum(np.linalg.norm(np.diff(coarse, axis=0), axis=1)))
    n = max(16, int(np.ceil(4 * rough / step)) + 1)
    u = np.linspace(u0, u1, n)
    points = np.array(splev(u, (knots, coefficients, k))).T
    tangents = np.array(splev(u, (knots, coefficients, k), der=1)).T
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    return points, tangents, arc


def _trim(coords: np.ndarray, length: float, at_start: bool) -> np.ndarray:
    """Remove ``length`` voxels of path from one end of a chain, keeping at least half of it."""
    path = coords[::-1] if at_start else coords
    if len(path) < 4:
        return coords
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    remaining = np.concatenate([[0.0], np.cumsum(steps[::-1])])[::-1]  # path length from each point to the end
    keep = max(int(np.searchsorted(-remaining, -length)), len(path) // 2)
    keep = min(keep, len(path))
    trimmed = path[:keep] if keep >= 2 else path
    return trimmed[::-1] if at_start else trimmed


def _extend(
    coords: np.ndarray,
    mask: np.ndarray,
    window: float,
    limit: float,
    at_start: bool,
) -> Tuple[np.ndarray, bool]:
    """Extend a chain end (voxel coords) along its direction while inside ``mask``, by at most ``limit`` voxels."""
    path = coords[::-1] if at_start else coords
    if len(path) < 2:
        return coords, False
    steps = np.linalg.norm(np.diff(path[::-1], axis=0), axis=1)
    reach = np.concatenate([[0.0], np.cumsum(steps)])
    back = path[::-1][min(int(np.searchsorted(reach, window)), len(path) - 1)]
    direction = path[-1] - back
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return coords, False
    direction /= norm
    added, shape = [], np.array(mask.shape)
    for t in np.arange(0.5, limit + 1e-9, 0.5):
        point = path[-1] + t * direction
        voxel = np.round(point).astype(int)
        if np.any(voxel < 0) or np.any(voxel >= shape) or not mask[tuple(voxel)]:
            break
        added.append(point)
    if not added:
        return coords, False
    added = np.array(added)
    return (np.concatenate([added[::-1], coords]) if at_start else np.concatenate([coords, added])), True


# ---------------------------------------------------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------------------------------------------------


def fill_lumen(mask: np.ndarray, max_radius: float) -> np.ndarray:
    """Fill the holes of a 3D mask that are enclosed within a slice along any axis and no larger than a disc of
    ``max_radius`` voxels, such as the lumen of a tube whose wall alone is labelled. Larger enclosed regions (a loop
    of several filaments) are left open.

    Args:
        mask: 3D boolean mask.
        max_radius: Radius, in voxels, of the largest disc-sized hole to fill.

    Returns:
        The filled mask (a copy).
    """
    max_area = np.pi * max_radius**2
    filled = mask.copy()
    for axis in range(3):
        for index in range(mask.shape[axis]):
            plane = np.take(mask, index, axis=axis)
            holes = ndimage.binary_fill_holes(plane) & ~plane
            if not holes.any():
                continue
            labels, _ = ndimage.label(holes)
            small = np.bincount(labels.ravel()) <= max_area
            small[0] = False
            target = [slice(None)] * 3
            target[axis] = index
            filled[tuple(target)] |= small[labels]
    return filled


def _trace_mask(
    mask: np.ndarray,
    offset: np.ndarray,
    voxel_size: float,
    params: TraceParameters,
    single: bool,
    component: int,
    report: TraceReport,
) -> List[Centreline]:
    """Trace the filaments of one component (or instance) mask, cropped at ``offset`` (z, y, x)."""
    from skimage.morphology import skeletonize

    padded = np.pad(mask, 1)
    skeleton = skeletonize(padded)
    if not skeleton.any():
        # Thinning removes compact specks and blobs entirely: they have no length.
        report.rejected_length += 1
        return []
    edt = ndimage.distance_transform_edt(padded)
    on_skeleton = edt[skeleton]
    radius = float(np.median(on_skeleton)) if on_skeleton.size else 0.5  # voxels
    diameter = 2.0 * radius

    def voxels(value: Optional[float], default: float) -> float:
        return default if value is None else float(value) / voxel_size

    chains, resolved = skeleton_chains(
        skeleton,
        prune_length=voxels(params.prune_length, diameter),
        junction_merge=voxels(params.junction_merge, diameter),
        max_bend=params.max_bend,
        direction_window=voxels(params.direction_window, 2.0 * diameter),
        junction_trim=radius,
    )
    report.junctions_resolved += resolved
    if single and len(chains) > 1:
        lengths = [np.sum(np.linalg.norm(np.diff(c.coords, axis=0), axis=1)) for c in chains]
        keep = int(np.argmax(lengths))
        report.dropped_branch_length += float((sum(lengths) - lengths[keep]) * voxel_size)
        chains = [chains[keep]]

    sigma = voxels(params.smoothing, 0.5) * voxel_size  # Angstrom
    window = voxels(params.direction_window, 2.0 * diameter)
    centrelines = []
    for chain in chains:
        coords = chain.coords.astype(float)
        extended = []
        if params.extend_ends and not chain.closed:
            for at_start, free in zip((True, False), chain.free_ends):
                if free:
                    # Thinning bends the last radius or so of a skeleton (hooks at ends, and where the label meets
                    # the volume's edge), so drop it and continue the filament's own direction to the label's edge.
                    coords = _trim(coords, radius, at_start)
                    coords, did = _extend(coords, padded, window, 2.5 * radius + 1.0, at_start)
                    extended.append(did)
        coords = coords - 1.0 + offset  # undo the padding, back to the volume's (z, y, x)
        points = coords[:, ::-1] * voxel_size  # (x, y, z) Angstrom
        length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) if len(points) > 1 else 0.0
        local = edt[tuple(np.clip(np.round(chain.coords).astype(int), 0, np.array(padded.shape) - 1).T)]
        chain_radius = float(np.median(local)) * voxel_size if local.size else None
        if params.min_length is not None and length < params.min_length:
            report.rejected_length += 1
            continue
        if params.min_radius is not None and (chain_radius or 0.0) < params.min_radius:
            report.rejected_radius += 1
            continue
        if chain_radius and length < params.min_aspect * 2.0 * chain_radius:
            report.rejected_aspect += 1
            continue
        points = _dedupe(points)
        s = len(points) * sigma**2
        tck = fit_spline(points, s=s)
        if tck is None:
            report.rejected_length += 1
            continue
        tck, s, hooks = _fit_without_end_hooks(points, tck, sigma, diameter * voxel_size, params.max_bend)
        dense, _, arc = evaluate_spline(tck, step=voxel_size)
        inside = _inside(dense, padded, offset, voxel_size)
        centrelines.append(
            Centreline(
                instance_id=0,
                tck=tck,
                smoothing=s,
                points=dense,
                length=float(arc[-1]),
                radius=chain_radius,
                score=float(inside),
                component=component,
                metadata={
                    "junctions": chain.junctions,
                    "ends_extended": int(sum(extended)),
                    "end_hooks_removed": hooks,
                    "closed": bool(chain.closed),
                },
            ),
        )
    return centrelines


def _end_bend(tck: Tuple[np.ndarray, List[np.ndarray], int], diameter: float) -> Tuple[float, float]:
    """The angle (degrees) at each end between the curve's end tangent and its direction from one to three label
    diameters in; a large angle is a hook at that end."""
    points, tangents, arc = evaluate_spline(tck, step=max(diameter / 8.0, 1e-6))
    length = arc[-1]
    if length < 4.0 * diameter:
        return 0.0, 0.0

    def direction(a: float, b: float) -> np.ndarray:
        i, j = np.searchsorted(arc, a), np.searchsorted(arc, b)
        d = points[min(j, len(points) - 1)] - points[min(i, len(points) - 1)]
        return d / max(np.linalg.norm(d), 1e-12)

    start = np.degrees(np.arccos(np.clip(np.dot(tangents[0], direction(diameter, 3.0 * diameter)), -1.0, 1.0)))
    inner_end = direction(length - 3.0 * diameter, length - diameter)
    end = np.degrees(np.arccos(np.clip(np.dot(tangents[-1], inner_end), -1.0, 1.0)))
    return float(start), float(end)


def _fit_without_end_hooks(
    points: np.ndarray,
    tck: Tuple[np.ndarray, List[np.ndarray], int],
    sigma: float,
    diameter: float,
    max_bend: float,
    rounds: int = 3,
) -> Tuple[Tuple[np.ndarray, List[np.ndarray], int], float, int]:
    """Refit without the end points that make the curve hook at an end (thinning artefacts where a filament meets
    another structure or the edge of the volume). An end is trimmed by one label diameter each round, at most
    ``rounds`` times.

    Returns:
        Tuple of (spline, its smoothing factor, number of end trims).
    """
    trims = 0
    s = len(points) * sigma**2
    for _ in range(rounds):
        start_bend, end_bend = _end_bend(tck, diameter)
        if start_bend <= max_bend and end_bend <= max_bend:
            break
        chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
        keep = np.ones(len(points), dtype=bool)
        if start_bend > max_bend:
            keep &= chord >= diameter
            trims += 1
        if end_bend > max_bend:
            keep &= chord <= chord[-1] - diameter
            trims += 1
        if keep.sum() < 4:
            break
        points = points[keep]
        s = len(points) * sigma**2
        refit = fit_spline(points, s=s)
        if refit is None:
            break
        tck = refit
    return tck, s, trims


def _inside(points: np.ndarray, padded: np.ndarray, offset: np.ndarray, voxel_size: float) -> float:
    """Fraction of points (x, y, z Angstrom) inside a padded mask cropped at ``offset``."""
    zyx = np.round(points[:, ::-1] / voxel_size - offset + 1.0).astype(int)
    valid = np.all((zyx >= 0) & (zyx < np.array(padded.shape)), axis=1)
    hits = np.zeros(len(zyx), dtype=bool)
    hits[valid] = padded[tuple(zyx[valid].T)]
    return float(hits.mean()) if len(hits) else 0.0


def _orient(centreline: Centreline, voxel_size: float) -> Centreline:
    """Order a filament so that it starts at the end with the smaller (z, y, x) voxel; reverse the fit if needed."""
    start = np.round(centreline.points[0][::-1] / voxel_size).astype(int)
    end = np.round(centreline.points[-1][::-1] / voxel_size).astype(int)
    if tuple(end) < tuple(start):
        knots, coefficients, k = centreline.tck
        u0, u1 = knots[0], knots[-1]
        centreline.tck = (u0 + u1 - knots[::-1], [c[::-1] for c in coefficients], k)
        centreline.points = centreline.points[::-1]
    return centreline


def trace_centrelines(
    volume: np.ndarray,
    voxel_size: float,
    params: Optional[TraceParameters] = None,
    instances: bool = False,
) -> Tuple[List[Centreline], np.ndarray, TraceReport]:
    """Trace the filaments of a segmentation.

    Args:
        volume: (Z, Y, X) segmentation. Without ``instances``, any non-zero voxel is filament. With ``instances``, each
            positive value is one filament (an instance segmentation).
        voxel_size: Voxel size in Angstrom.
        params: Tracing parameters (defaults: ``TraceParameters()``).
        instances: Trace each instance on its own and keep its ID.

    Returns:
        Tuple of (centrelines, assignment, report):
            - centrelines: The filaments, with IDs 1..K by length (longest first) for a semantic input, or the input's
              IDs for an instance input. Points start at the end with the smaller (z, y, x).
            - assignment: (Z, Y, X) instance segmentation: each label voxel holds the ID of the nearest filament of its
              component; voxels of components without a filament are 0. For an instance input, the kept instances.
              With ``fill_lumen``, the filled holes are part of the label.
            - report: Counts.
    """
    params = params or TraceParameters()
    report = TraceReport()
    volume = np.asarray(volume)
    if params.fill_lumen:
        volume = _fill_label(volume, params.fill_lumen / voxel_size, instances)
    if instances:
        labeled = volume.astype(np.int64, copy=False)
        ids = np.unique(labeled[labeled > 0])
    else:
        labeled, n = ndimage.label(volume > 0, structure=np.ones((3, 3, 3)))
        ids = np.arange(1, n + 1)
    counts = np.bincount(labeled.ravel()) if labeled.size else np.zeros(1, dtype=int)
    boxes = ndimage.find_objects(labeled)
    report.components = int(len(ids))

    found: List[Centreline] = []
    for value in ids:
        value = int(value)
        if params.min_volume is not None and counts[value] * voxel_size**3 < params.min_volume:
            report.components_removed_volume += 1
            continue
        box = boxes[value - 1]
        if box is None:
            continue
        offset = np.array([sl.start for sl in box])
        mask = labeled[box] == value
        if instances:
            mask, _ = _largest_piece(mask)
        traced = _trace_mask(mask, offset, voxel_size, params, single=instances, component=value, report=report)
        for centreline in traced:
            if instances:
                centreline.instance_id = value
            found.append(_orient(centreline, voxel_size))

    if not instances:
        found.sort(key=lambda c: -c.length)
        for i, centreline in enumerate(found, start=1):
            centreline.instance_id = i
    report.filaments = len(found)
    report.total_length = float(sum(c.length for c in found))
    for centreline in found:
        centreline.metadata.update(
            {
                "method": "splprep",
                "degree": int(centreline.tck[2]),
                "smoothing": float(centreline.smoothing),
                "component": int(centreline.component),
                "label_radius": centreline.radius,
                "tracer_version": TRACER_VERSION,
                "parameters": params.as_dict(),
            },
        )

    if instances:
        kept = np.zeros(int(labeled.max()) + 1 if labeled.size else 1, dtype=bool)
        kept[[c.instance_id for c in found]] = True
        assignment = np.where(kept[labeled], labeled, 0)
    else:
        assignment = _assign(labeled, found, boxes, voxel_size)
    return found, assignment, report


def _fill_label(volume: np.ndarray, max_radius: float, instances: bool) -> np.ndarray:
    """The label with its lumen-sized holes filled (see ``fill_lumen``): as a whole for a semantic label, and one
    instance at a time, into empty voxels only, for an instance segmentation."""
    if not instances:
        return fill_lumen(volume > 0, max_radius)
    filled = volume.copy()
    for value, box in enumerate(ndimage.find_objects(volume.astype(np.int64, copy=False)), start=1):
        if box is None:
            continue
        mask = fill_lumen(volume[box] == value, max_radius)
        region = filled[box]
        region[mask & (region == 0)] = value
    return filled


def _largest_piece(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """The largest 26-connected piece of a mask, and the number of pieces."""
    pieces, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return mask, n
    sizes = np.bincount(pieces.ravel())
    sizes[0] = 0
    return pieces == int(np.argmax(sizes)), n


def _assign(
    labeled: np.ndarray,
    centrelines: Sequence[Centreline],
    boxes: Sequence[Optional[Tuple[slice, ...]]],
    voxel_size: float,
) -> np.ndarray:
    """Give each voxel of a component the ID of the nearest filament traced from that component."""
    dtype = np.uint16 if len(centrelines) < 2**16 else np.uint32
    assignment = np.zeros(labeled.shape, dtype=dtype)
    by_component: Dict[int, List[Centreline]] = {}
    for centreline in centrelines:
        by_component.setdefault(centreline.component, []).append(centreline)
    for component, members in by_component.items():
        box = boxes[component - 1]
        local = labeled[box] == component
        offset = np.array([sl.start for sl in box])
        voxels_zyx = np.argwhere(local)
        points = np.concatenate([c.points for c in members])
        owner = np.concatenate([np.full(len(c.points), c.instance_id) for c in members])
        _, nearest = cKDTree(points).query((voxels_zyx + offset)[:, ::-1] * voxel_size)
        sub = assignment[box]
        sub[tuple(voxels_zyx.T)] = owner[nearest]
    return assignment


# ---------------------------------------------------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------------------------------------------------


def evaluate_curve_with_tangents(curve: Dict[str, Any], step: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense points, unit tangents and arc length along a copick filament curve.

    Args:
        curve: A copick curve's fields: ``kind`` (``bspline``, ``catmull-rom`` or ``linear``), ``control_points``
            (Angstrom), and ``degree``/``knots`` (``bspline``) or ``alpha`` (``catmull-rom``).
        step: Largest spacing of the returned points, in Angstrom.

    Returns:
        Tuple of (points (N, 3), tangents (N, 3), arc length (N,)). A ``bspline`` has its exact derivative as
        tangent; the other kinds take the direction of the curve, evaluated by copick at ``step``.
    """
    kind = curve["kind"]
    control = np.asarray(curve["control_points"], dtype=float)
    if kind == "bspline":
        coefficients = [control[:, i] for i in range(3)]
        return evaluate_spline((np.asarray(curve["knots"], dtype=float), coefficients, int(curve["degree"])), step)
    from copick.util.filaments import evaluate_curve

    points = evaluate_curve(control, step=step, kind=kind, alpha=curve.get("alpha"))
    tangents = np.gradient(points, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    return points, tangents, arc


def _least_aligned_normal(tangent: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to ``tangent``, from the coordinate axis least aligned with it."""
    axis = np.eye(3)[int(np.argmin(np.abs(tangent)))]
    normal = axis - np.dot(axis, tangent) * tangent
    return normal / np.linalg.norm(normal)


def rotation_minimizing_frames(points: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    """Rotation-minimizing frames along a curve (double reflection; Wang et al. 2008).

    Args:
        points: (N, 3) points along the curve.
        tangents: (N, 3) unit tangents.

    Returns:
        (N, 3, 3) rotations whose columns are (x, y, z) with z the tangent. The first frame's x axis is the coordinate
        axis least aligned with the first tangent, made perpendicular to it; later frames turn as little as possible.
    """
    n = len(points)
    frames = np.zeros((n, 3, 3))
    x = _least_aligned_normal(tangents[0])
    for i in range(n):
        if i > 0:
            v1 = points[i] - points[i - 1]
            c1 = np.dot(v1, v1)
            if c1 > 1e-18:
                x_l = x - (2.0 / c1) * np.dot(v1, x) * v1
                t_l = tangents[i - 1] - (2.0 / c1) * np.dot(v1, tangents[i - 1]) * v1
                v2 = tangents[i] - t_l
                c2 = np.dot(v2, v2)
                x = x_l - (2.0 / c2) * np.dot(v2, x_l) * v2 if c2 > 1e-18 else x_l
            x = x - np.dot(x, tangents[i]) * tangents[i]
            norm = np.linalg.norm(x)
            x = x / norm if norm > 1e-12 else _least_aligned_normal(tangents[i])
        z = tangents[i]
        y = np.cross(z, x)
        frames[i] = np.column_stack([x, y, z])
    return frames


def sample_curve(
    curve: Dict[str, Any],
    spacing: float,
    anchor: str = "center",
    roll: str = "parallel",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Picks along a filament curve at a fixed spacing, with +Z along the filament.

    Args:
        curve: A copick curve's fields (see ``evaluate_curve_with_tangents``).
        spacing: Distance between consecutive picks along the curve, in Angstrom.
        anchor: ``center`` splits the length left over after the last full step evenly between the two ends;
            ``start`` places the first pick at the start of the curve.
        roll: ``parallel``: the frames are rotation-minimizing along the curve, starting from the coordinate axis
            least aligned with the first tangent (no frame flips, in any direction). ``random``: each pick gets a
            uniformly random rotation about the axis.
        seed: Random seed for ``roll="random"``.

    Returns:
        Tuple of (positions (M, 3) Angstrom, rotations (M, 3, 3), distance along the curve (M,)). A curve shorter
        than ``spacing`` gives one pick at its middle.
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if anchor not in ("center", "start"):
        raise ValueError("anchor must be 'center' or 'start'")
    if roll not in ("parallel", "random"):
        raise ValueError("roll must be 'parallel' or 'random'")
    fine = min(float(spacing), float(curve.get("step") or spacing)) / 4.0
    points, tangents, arc = evaluate_curve_with_tangents(curve, fine)
    length = float(arc[-1])
    if length < spacing:
        along = np.array([length / 2.0])
    else:
        count = int(np.floor(length / spacing + 1e-9)) + 1
        margin = (length - (count - 1) * spacing) / 2.0 if anchor == "center" else 0.0
        along = margin + spacing * np.arange(count)

    positions = np.column_stack([np.interp(along, arc, points[:, i]) for i in range(3)])
    sample_tangents = np.column_stack([np.interp(along, arc, tangents[:, i]) for i in range(3)])
    sample_tangents /= np.maximum(np.linalg.norm(sample_tangents, axis=1, keepdims=True), 1e-12)

    frames = rotation_minimizing_frames(points, tangents)
    nearest = np.clip(np.searchsorted(arc, along), 0, len(arc) - 1)
    rotations = np.zeros((len(along), 3, 3))
    for i, (k, z) in enumerate(zip(nearest, sample_tangents)):
        x = frames[k][:, 0] - np.dot(frames[k][:, 0], z) * z
        x /= np.linalg.norm(x)
        rotations[i] = np.column_stack([x, np.cross(z, x), z])
    if roll == "random":
        angles = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, len(along))
        for i, angle in enumerate(angles):
            c, s = np.cos(angle), np.sin(angle)
            rotations[i] = rotations[i] @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return positions, rotations, along


# ---------------------------------------------------------------------------------------------------------------------
# Painting
# ---------------------------------------------------------------------------------------------------------------------


def paint_tubes(
    polylines: Sequence[np.ndarray],
    ids: Sequence[int],
    radii: Sequence[float],
    shape: Tuple[int, int, int],
    voxel_size: float,
) -> np.ndarray:
    """An instance segmentation of tubes around filament centrelines.

    Args:
        polylines: (N, 3) ordered points per filament, in Angstrom (x, y, z), at most a voxel apart.
        ids: Filament IDs (>= 1).
        radii: Tube radius per filament, in Angstrom.
        shape: (Z, Y, X) of the output.
        voxel_size: Voxel size in Angstrom.

    Returns:
        (Z, Y, X) array: voxels within a tube hold its filament's ID; where tubes overlap, the nearest centreline wins.
    """
    dtype = np.uint16 if (max(ids) if len(ids) else 0) < 2**16 else np.uint32
    out = np.zeros(shape, dtype=dtype)
    trees: Dict[int, cKDTree] = {}
    shape_arr = np.array(shape)
    for polyline, fid, radius in zip(polylines, ids, radii):
        polyline = np.asarray(polyline, dtype=float)
        if len(polyline) == 0 or radius <= 0:
            continue
        tree = trees[int(fid)] = cKDTree(polyline)
        zyx = polyline[:, ::-1] / voxel_size
        r = radius / voxel_size
        low = np.maximum(np.floor(zyx.min(axis=0) - r).astype(int), 0)
        high = np.minimum(np.ceil(zyx.max(axis=0) + r).astype(int) + 1, shape_arr)
        if np.any(high <= low):
            continue
        grid = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(low, high)], indexing="ij"), axis=-1)
        grid = grid.reshape(-1, 3)
        distance, _ = tree.query(grid[:, ::-1] * voxel_size, distance_upper_bound=radius)
        inside = np.isfinite(distance)
        cells, distance = grid[inside], distance[inside]
        current = out[tuple(cells.T)]
        conflict = current > 0
        if np.any(conflict):
            other = np.full(conflict.sum(), np.inf)
            for owner in np.unique(current[conflict]):
                sel = current[conflict] == owner
                other[sel], _ = trees[int(owner)].query(cells[conflict][sel][:, ::-1] * voxel_size)
            take = np.ones(len(cells), dtype=bool)
            take[np.flatnonzero(conflict)] = distance[conflict] < other
            cells = cells[take]
        out[tuple(cells.T)] = fid
    return out
