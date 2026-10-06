"""Skeleton graphs: a 3D skeleton's voxels as a graph, split into branches between endpoints and junctions.

Shared by ``skeletonize`` (spur pruning), ``seg-stats --skeleton`` (skeleton length) and the filament tracer
(``seg2fil``, ``fit-spline``). Coordinates are voxel indices in (z, y, x) order and lengths are in voxels, unless a
function says otherwise. Everything here is numpy and scipy; nothing reads or writes copick objects.

The graph joins 26-connected skeleton voxels. A thin 26-connected skeleton can still contain staircase corners, where
three voxels are pairwise adjacent; the diagonal edge of such a triangle duplicates the two shorter steps around it and
would make every voxel of a staircase look like a junction. ``skeleton_graph`` therefore drops an edge when a common
neighbour reaches both of its ends by strictly shorter steps.

A topology (``decompose``) is a list of branches, each an ordered voxel path between two nodes. A node is a junction
(a cluster of adjacent voxels with three or more neighbours), an endpoint (one neighbour, or an isolated voxel), or the
start of a closed loop.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

#: Half of the 26 neighbour offsets (lexicographically positive), so each edge is generated once.
_OFFSETS = np.array(
    [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dz, dy, dx) > (0, 0, 0)],
)


@dataclass
class SkeletonGraph:
    """The voxels of a skeleton and the edges between 26-neighbours.

    Attributes:
        coords: (N, 3) voxel indices (z, y, x), in C order.
        adjacency: (N, N) symmetric sparse matrix of edge lengths (1, sqrt(2) or sqrt(3)).
    """

    coords: np.ndarray
    adjacency: sparse.csr_matrix

    @property
    def degree(self) -> np.ndarray:
        """The number of neighbours of each voxel."""
        return np.diff(self.adjacency.indptr)

    def neighbours(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """The neighbours of voxel ``i`` and the lengths of the edges to them."""
        a, b = self.adjacency.indptr[i], self.adjacency.indptr[i + 1]
        return self.adjacency.indices[a:b], self.adjacency.data[a:b]

    def subgraph(self, keep: np.ndarray) -> "SkeletonGraph":
        """The graph of the kept voxels (a boolean mask over the voxels)."""
        keep = np.asarray(keep, dtype=bool)
        return SkeletonGraph(self.coords[keep], self.adjacency[keep][:, keep].tocsr())


def skeleton_graph(skeleton: np.ndarray) -> SkeletonGraph:
    """The graph of a 3D skeleton's voxels, without the redundant diagonals of staircase corners.

    Args:
        skeleton: 3D boolean (or non-zero) array.

    Returns:
        The skeleton graph.
    """
    coords = np.argwhere(skeleton)
    n = len(coords)
    if n == 0:
        return SkeletonGraph(coords.reshape(0, 3), sparse.csr_matrix((0, 0)))
    shape = np.asarray(skeleton.shape)
    linear = np.ravel_multi_index(coords.T, skeleton.shape)  # ascending: argwhere returns C order
    rows, cols, lengths = [], [], []
    for offset in _OFFSETS:
        moved = coords + offset
        inside = np.all((moved >= 0) & (moved < shape), axis=1)
        source = np.flatnonzero(inside)
        target_linear = np.ravel_multi_index(moved[inside].T, skeleton.shape)
        position = np.minimum(np.searchsorted(linear, target_linear), n - 1)
        hit = linear[position] == target_linear
        rows.append(source[hit])
        cols.append(position[hit])
        lengths.append(np.full(int(hit.sum()), float(np.linalg.norm(offset))))
    rows, cols, lengths = np.concatenate(rows), np.concatenate(cols), np.concatenate(lengths)
    keep = _non_redundant(rows, cols, lengths, n)
    rows, cols, lengths = rows[keep], cols[keep], lengths[keep]
    adjacency = sparse.coo_matrix(
        (np.r_[lengths, lengths], (np.r_[rows, cols], np.r_[cols, rows])),
        shape=(n, n),
    ).tocsr()
    return SkeletonGraph(coords, adjacency)


def _non_redundant(rows: np.ndarray, cols: np.ndarray, lengths: np.ndarray, n: int) -> np.ndarray:
    """Which edges to keep: an edge is redundant when a common neighbour reaches both its ends by shorter steps."""
    keep = np.ones(len(rows), dtype=bool)
    long_edges = np.flatnonzero(lengths > 1.0)
    if len(long_edges) == 0:
        return keep
    full = sparse.coo_matrix(
        (np.r_[lengths, lengths], (np.r_[rows, cols], np.r_[cols, rows])),
        shape=(n, n),
    ).tocsr()
    for e in long_edges:
        i, j, length = rows[e], cols[e], lengths[e]
        ni, li = full.indices[full.indptr[i] : full.indptr[i + 1]], full.data[full.indptr[i] : full.indptr[i + 1]]
        nj, lj = full.indices[full.indptr[j] : full.indptr[j + 1]], full.data[full.indptr[j] : full.indptr[j + 1]]
        shorter_i = set(ni[li < length].tolist())
        if any(k in shorter_i for k in nj[lj < length].tolist()):
            keep[e] = False
    return keep


@dataclass
class Branch:
    """An ordered voxel path between two topology nodes.

    Attributes:
        path: Voxel indices into the graph, in order, both end voxels included.
        start: Topology node at ``path[0]``.
        end: Topology node at ``path[-1]``.
        length: Length along the path, in voxels.
    """

    path: np.ndarray
    start: int
    end: int
    length: float

    @property
    def closed(self) -> bool:
        """Whether the branch starts and ends at the same node."""
        return self.start == self.end


@dataclass
class Topology:
    """A skeleton as branches between nodes.

    Attributes:
        graph: The skeleton graph.
        branches: The branches.
        node_voxels: Voxel indices of each node (several for a junction cluster).
        node_kind: ``"junction"``, ``"endpoint"`` or ``"loop"`` per node.
    """

    graph: SkeletonGraph
    branches: List[Branch] = field(default_factory=list)
    node_voxels: Dict[int, np.ndarray] = field(default_factory=dict)
    node_kind: Dict[int, str] = field(default_factory=dict)

    def incident(self, node: int) -> List[Tuple[int, bool]]:
        """The branch ends at a node, as (branch index, True if the branch starts there). A closed branch at the
        node appears twice."""
        ends = []
        for b, branch in enumerate(self.branches):
            if branch.start == node:
                ends.append((b, True))
            if branch.end == node:
                ends.append((b, False))
        return ends

    def junctions(self) -> List[int]:
        """The junction nodes."""
        return [node for node, kind in self.node_kind.items() if kind == "junction"]

    @property
    def length(self) -> float:
        """The total length of the branches, in voxels."""
        return float(sum(b.length for b in self.branches))


def decompose(graph: SkeletonGraph, min_loop_length: float = 3.0) -> Topology:
    """Split a skeleton graph into branches between endpoints and junction clusters.

    Args:
        graph: The skeleton graph.
        min_loop_length: A branch that leaves a junction and returns to it is kept only when it is at least this long
            (voxels); shorter ones are corners of a junction cluster, not loops.

    Returns:
        The topology.
    """
    n = len(graph.coords)
    topology = Topology(graph)
    if n == 0:
        return topology
    degree = graph.degree
    junction_voxels = np.flatnonzero(degree >= 3)
    cluster_of = np.full(n, -1)
    if len(junction_voxels):
        sub = graph.adjacency[junction_voxels][:, junction_voxels]
        _, labels = connected_components(sub, directed=False)
        cluster_of[junction_voxels] = labels

    node_of = np.full(n, -1)
    next_node = 0
    for cluster in range(int(cluster_of.max()) + 1 if len(junction_voxels) else 0):
        members = junction_voxels[cluster_of[junction_voxels] == cluster]
        node_of[members] = next_node
        topology.node_voxels[next_node] = members
        topology.node_kind[next_node] = "junction"
        next_node += 1
    for i in np.flatnonzero(degree <= 1):
        node_of[i] = next_node
        topology.node_voxels[next_node] = np.array([i])
        topology.node_kind[next_node] = "endpoint"
        next_node += 1

    visited = np.zeros(n, dtype=bool)
    used_edges = set()
    for i in np.flatnonzero(node_of >= 0):
        visited[i] = True
        if degree[i] == 0:
            topology.branches.append(Branch(np.array([i]), int(node_of[i]), int(node_of[i]), 0.0))
            continue
        neighbours, lengths = graph.neighbours(i)
        for j, length in zip(neighbours, lengths):
            if node_of[j] >= 0 and node_of[j] == node_of[i]:
                continue  # an edge inside a junction cluster
            if (min(i, j), max(i, j)) in used_edges:
                continue
            path, total = _walk(graph, i, int(j), float(length), node_of)
            used_edges.add((min(path[0], path[1]), max(path[0], path[1])))
            used_edges.add((min(path[-2], path[-1]), max(path[-2], path[-1])))
            visited[path] = True
            start, end = int(node_of[path[0]]), int(node_of[path[-1]])
            if start == end and topology.node_kind[start] == "junction" and total < min_loop_length:
                continue
            topology.branches.append(Branch(np.asarray(path), start, end, total))

    # Closed loops with no junction or endpoint: every voxel has two neighbours.
    for i in np.flatnonzero(~visited):
        if visited[i]:
            continue
        node_of[i] = next_node
        topology.node_voxels[next_node] = np.array([i])
        topology.node_kind[next_node] = "loop"
        neighbours, lengths = graph.neighbours(i)
        path, total = _walk(graph, int(i), int(neighbours[0]), float(lengths[0]), node_of)
        visited[path] = True
        topology.branches.append(Branch(np.asarray(path), next_node, next_node, total))
        next_node += 1
    return topology


def _walk(graph: SkeletonGraph, start: int, step: int, length: float, node_of: np.ndarray) -> Tuple[List[int], float]:
    """Follow two-neighbour voxels from ``start`` through ``step`` until a node voxel (or back to ``start``)."""
    path = [start, step]
    previous, current = start, step
    while node_of[current] < 0:
        neighbours, lengths = graph.neighbours(current)
        others = [(k, w) for k, w in zip(neighbours, lengths) if k != previous]
        if not others:
            break
        k, w = others[0]
        previous, current = current, int(k)
        path.append(current)
        length += float(w)
        if current == start:
            break
    return path, length


def splice(topology: Topology) -> Topology:
    """Join the two branches at every junction that has exactly two branch ends (a kink, not a junction), and turn
    a junction with one branch end into an endpoint."""
    changed = True
    while changed:
        changed = False
        for node in list(topology.junctions()):
            ends = topology.incident(node)
            if len(ends) == 1:
                topology.node_kind[node] = "endpoint"
                changed = True
            elif len(ends) == 2 and ends[0][0] != ends[1][0]:
                (a, a_starts), (b, b_starts) = ends
                first, second = topology.branches[a], topology.branches[b]
                path_a = first.path[::-1] if a_starts else first.path  # ends at the node
                path_b = second.path if b_starts else second.path[::-1]  # starts at the node
                start = first.end if a_starts else first.start
                end = second.end if b_starts else second.start
                gap = np.linalg.norm(topology.graph.coords[path_a[-1]] - topology.graph.coords[path_b[0]])
                path = np.concatenate([path_a, path_b[1:] if path_a[-1] == path_b[0] else path_b])
                joined = Branch(path, start, end, first.length + second.length + float(gap))
                topology.branches = [br for k, br in enumerate(topology.branches) if k not in (a, b)] + [joined]
                del topology.node_kind[node]
                del topology.node_voxels[node]
                changed = True
                break
            elif len(ends) == 0:
                del topology.node_kind[node]
                del topology.node_voxels[node]
                changed = True
                break
    return topology


def prune_spurs(topology: Topology, min_length: float) -> Topology:
    """Remove terminal branches (one end an endpoint, the other a junction) shorter than ``min_length`` voxels.

    Repeats until no spur is left, splicing the junctions that pruning reduces to two branch ends. A junction whose
    branches are all short spurs keeps its longest one. Branches between two endpoints are never removed: they are a
    whole piece, not a spur.

    Args:
        topology: The topology (modified in place and returned).
        min_length: Spurs shorter than this (voxels) are removed.

    Returns:
        The pruned topology.
    """
    while True:
        spurs = []
        for b, branch in enumerate(topology.branches):
            kinds = (topology.node_kind.get(branch.start), topology.node_kind.get(branch.end))
            if sorted(kinds) == ["endpoint", "junction"] and branch.length < min_length:
                spurs.append(b)
        if not spurs:
            return topology
        spur_set = set(spurs)
        for node in topology.junctions():
            ends = topology.incident(node)
            if ends and all(b in spur_set for b, _ in ends):
                spur_set.discard(max((b for b, _ in ends), key=lambda b: topology.branches[b].length))
        if not spur_set:
            return topology
        for b in spur_set:
            branch = topology.branches[b]
            for node in (branch.start, branch.end):
                if topology.node_kind.get(node) == "endpoint":
                    topology.node_kind.pop(node)
                    topology.node_voxels.pop(node)
        topology.branches = [br for k, br in enumerate(topology.branches) if k not in spur_set]
        splice(topology)


def merge_junctions(topology: Topology, max_length: float) -> Topology:
    """Contract branches shorter than ``max_length`` voxels that join two different junctions, so two nearby
    junctions (two filaments crossing, seen as two forks joined by a short bridge) become one. Short branches that
    then start and end at the merged junction (bubbles of the skeleton, where the label is thick) are absorbed too.

    Args:
        topology: The topology (modified in place and returned).
        max_length: Bridges up to this length (voxels) are contracted.

    Returns:
        The topology.
    """
    parent = {node: node for node in topology.node_kind}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    bridges = []
    for b, branch in enumerate(topology.branches):
        if (
            branch.start != branch.end
            and topology.node_kind.get(branch.start) == "junction"
            and topology.node_kind.get(branch.end) == "junction"
            and branch.length <= max_length
        ):
            bridges.append(b)
            parent[find(branch.start)] = find(branch.end)
    if not bridges:
        return topology
    bridge_set = set(bridges)
    merged_voxels: Dict[int, List[np.ndarray]] = {}
    for node, voxels in topology.node_voxels.items():
        merged_voxels.setdefault(find(node), []).append(voxels)
    for b in bridges:
        merged_voxels[find(topology.branches[b].start)].append(topology.branches[b].path)
    kept = []
    for b, branch in enumerate(topology.branches):
        if b in bridge_set:
            continue
        branch.start, branch.end = find(branch.start), find(branch.end)
        if branch.closed and topology.node_kind.get(branch.start) == "junction" and branch.length <= max_length:
            merged_voxels[branch.start].append(branch.path)  # a bubble inside the merged junction
            continue
        kept.append(branch)
    topology.branches = kept
    topology.node_kind = {find(node): kind for node, kind in topology.node_kind.items()}
    topology.node_voxels = {node: np.unique(np.concatenate(parts)) for node, parts in merged_voxels.items()}
    return topology


def node_centre(topology: Topology, node: int) -> np.ndarray:
    """The mean voxel position (z, y, x) of a node."""
    return topology.graph.coords[topology.node_voxels[node]].mean(axis=0)


def pruned_skeleton(skeleton: np.ndarray, min_length: float) -> np.ndarray:
    """A skeleton with its spurs shorter than ``min_length`` voxels removed (see ``prune_spurs``).

    Args:
        skeleton: 3D boolean (or non-zero) array.
        min_length: Spur length threshold, in voxels.

    Returns:
        Boolean array of the same shape.
    """
    graph = skeleton_graph(skeleton)
    topology = prune_spurs(decompose(graph), min_length)
    keep = np.zeros(len(graph.coords), dtype=bool)
    for branch in topology.branches:
        keep[branch.path] = True
    for node, voxels in topology.node_voxels.items():
        if topology.node_kind[node] == "junction":
            keep[voxels] = True
    out = np.zeros(skeleton.shape, dtype=bool)
    out[tuple(graph.coords[keep].T)] = True
    return out


def skeleton_length(skeleton: np.ndarray, voxel_size: float = 1.0) -> float:
    """The total length of a skeleton's branches, in the unit of ``voxel_size`` (voxels by default)."""
    return decompose(skeleton_graph(skeleton)).length * float(voxel_size)


@dataclass
class SkeletonSummary:
    """Skeleton measurements of one component or instance.

    Attributes:
        length: Total branch length (voxels).
        n_branches: Number of branches.
        n_junctions: Number of junctions.
        n_endpoints: Number of endpoints.
        radius: Median distance from the skeleton to the background (voxels), or None without a skeleton.
    """

    length: float
    n_branches: int
    n_junctions: int
    n_endpoints: int
    radius: Optional[float]


def summarize(mask: np.ndarray, prune_length: float = 0.0) -> SkeletonSummary:
    """Skeletonize a binary mask (Lee) and measure its skeleton.

    Args:
        mask: 3D boolean array of one component or instance.
        prune_length: Remove spurs shorter than this (voxels) before measuring; 0 keeps every branch.

    Returns:
        The measurements.
    """
    from scipy import ndimage
    from skimage.morphology import skeletonize

    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    skeleton = skeletonize(padded)
    if not skeleton.any():
        return SkeletonSummary(0.0, 0, 0, 0, None)
    topology = decompose(skeleton_graph(skeleton))
    if prune_length > 0:
        topology = prune_spurs(topology, prune_length)
    edt = ndimage.distance_transform_edt(padded)
    on_skeleton = topology.graph.coords
    radius = float(np.median(edt[tuple(on_skeleton.T)])) if len(on_skeleton) else None
    kinds = list(topology.node_kind.values())
    return SkeletonSummary(
        length=topology.length,
        n_branches=len(topology.branches),
        n_junctions=kinds.count("junction"),
        n_endpoints=kinds.count("endpoint"),
        radius=radius,
    )
