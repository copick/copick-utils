"""The skeleton graph splits skeletons into branches between endpoints and junctions."""

import numpy as np
from copick_utils.process.skeleton_graph import (
    decompose,
    merge_junctions,
    prune_spurs,
    pruned_skeleton,
    skeleton_graph,
    splice,
    summarize,
)

SHAPE = (20, 40, 40)


def _kinds(topology):
    return sorted(topology.node_kind.values())


def test_line_and_staircase_are_single_branches():
    line = np.zeros(SHAPE, bool)
    line[10, 20, 2:30] = True
    topology = decompose(skeleton_graph(line))
    assert len(topology.branches) == 1 and _kinds(topology) == ["endpoint", "endpoint"]
    assert topology.length == 27.0

    # A staircase has pairwise-adjacent corner voxels; their diagonals must not read as junctions.
    stairs = np.zeros(SHAPE, bool)
    y = x = 5
    for k in range(20):
        stairs[10, y, x] = True
        if k % 2:
            y += 1
        else:
            x += 1
    topology = decompose(skeleton_graph(stairs))
    assert len(topology.branches) == 1 and _kinds(topology) == ["endpoint", "endpoint"]


def test_junctions_loops_and_isolated_voxels():
    fork = np.zeros(SHAPE, bool)
    fork[10, 20, 2:20] = True
    for k in range(12):
        fork[10, 20 + k, 20 + k] = True
        fork[10, 20 - k, 20 + k] = True
    topology = splice(decompose(skeleton_graph(fork)))
    assert len(topology.branches) == 3 and _kinds(topology).count("junction") == 1

    ring = np.zeros(SHAPE, bool)
    theta = np.linspace(0, 2 * np.pi, 400)
    ring[10, np.round(20 + 10 * np.sin(theta)).astype(int), np.round(20 + 10 * np.cos(theta)).astype(int)] = True
    from skimage.morphology import skeletonize

    topology = splice(decompose(skeleton_graph(skeletonize(ring))))
    assert len(topology.branches) == 1 and topology.branches[0].closed

    dot = np.zeros(SHAPE, bool)
    dot[1, 1, 1] = True
    topology = decompose(skeleton_graph(dot))
    assert len(topology.branches) == 1 and _kinds(topology) == ["endpoint"]


def test_spurs_are_pruned_and_close_junctions_merge():
    spur = np.zeros(SHAPE, bool)
    spur[10, 20, 2:30] = True
    spur[10, 21:24, 15] = True
    topology = prune_spurs(decompose(skeleton_graph(spur)), min_length=5)
    assert len(topology.branches) == 1 and topology.length == 27.0
    assert pruned_skeleton(spur, 5).sum() == 28

    # Two forks joined by a short bridge, as two crossing filaments often skeletonize
    cross = np.zeros(SHAPE, bool)
    cross[10, 20, 15:25] = True
    for k in range(1, 10):
        cross[10, 20 - k, 15 - k] = cross[10, 20 + k, 15 - k] = True
        cross[10, 20 - k, 24 + k] = cross[10, 20 + k, 24 + k] = True
    topology = splice(decompose(skeleton_graph(cross)))
    assert _kinds(topology).count("junction") == 2
    topology = merge_junctions(topology, max_length=12)
    assert [len(topology.incident(node)) for node in topology.junctions()] == [4]


def test_summary_of_a_rod():
    rod = np.zeros((11, 11, 60), bool)
    rod[3:8, 3:8, 10:50] = True
    summary = summarize(rod)
    assert summary.n_branches == 1 and summary.n_junctions == 0
    assert 30 <= summary.length <= 40
    assert summary.radius == 3.0
