"""Shared utilities for constructing closed slab meshes from two regular grids of points."""

from typing import Tuple

import numpy as np
import trimesh as tm


def triangulate_rect_grid(points: np.ndarray, grid_dims: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate a regular grid of points into a surface mesh.

    Args:
        points: Nx3 array of points arranged on a regular grid (row-major order).
        grid_dims: (rows, cols) dimensions of the grid.

    Returns:
        Tuple of (vertices, faces) arrays.
    """
    grid = points.reshape(grid_dims[0], grid_dims[1], 3)

    vertices = []
    tris = []

    for i in range(grid_dims[0] - 1):
        for j in range(grid_dims[1] - 1):
            v1 = grid[i, j, :]
            v2 = grid[i + 1, j, :]
            v3 = grid[i + 1, j + 1, :]
            v4 = grid[i, j + 1, :]

            vertices.extend([v1, v2, v3, v4])
            lmax = len(vertices)
            tris.append([lmax - 2, lmax - 3, lmax - 4])
            tris.append([lmax - 4, lmax - 1, lmax - 2])

    return np.array(vertices), np.array(tris)


def fill_sides(
    top_points: np.ndarray,
    bottom_points: np.ndarray,
    grid_dims: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create side-wall triangles connecting two surface grids along their boundaries.

    Args:
        top_points: Nx3 array of top surface points (row-major order).
        bottom_points: Nx3 array of bottom surface points (row-major order).
        grid_dims: (rows, cols) dimensions of the grid.

    Returns:
        Tuple of (vertices, faces) arrays for the side walls.
    """
    arr1 = top_points.reshape(grid_dims[0], grid_dims[1], 3)
    arr2 = bottom_points.reshape(grid_dims[0], grid_dims[1], 3)

    vertices = []
    tris = []

    rows, cols = grid_dims

    # Connect along row boundaries (first and last row)
    for i in range(cols - 1):
        for j in [0, rows - 1]:
            v1 = arr1[j, i]
            v2 = arr2[j, i]
            v3 = arr2[j, i + 1]
            v4 = arr1[j, i + 1]

            vertices.extend([v1, v2, v3, v4])
            lmax = len(vertices)
            tris.append([lmax - 2, lmax - 3, lmax - 4])
            tris.append([lmax - 4, lmax - 1, lmax - 2])

    # Connect along column boundaries (first and last column)
    for i in range(rows - 1):
        for j in [0, cols - 1]:
            v1 = arr1[i, j]
            v2 = arr2[i, j]
            v3 = arr2[i + 1, j]
            v4 = arr1[i + 1, j]

            vertices.extend([v1, v2, v3, v4])
            lmax = len(vertices)
            tris.append([lmax - 2, lmax - 3, lmax - 4])
            tris.append([lmax - 4, lmax - 1, lmax - 2])

    return np.array(vertices), np.array(tris)


def triangulate_box(
    top_points: np.ndarray,
    bottom_points: np.ndarray,
    grid_dims: Tuple[int, int],
) -> tm.Trimesh:
    """Assemble a closed slab mesh from top and bottom surface grids.

    Creates a watertight mesh by triangulating the top surface, bottom surface,
    and connecting side walls between them.

    Args:
        top_points: Nx3 array of top surface points (row-major order).
        bottom_points: Nx3 array of bottom surface points (row-major order).
        grid_dims: (rows, cols) dimensions of the grid.

    Returns:
        A trimesh.Trimesh object representing the closed slab.
    """
    m1v, m1f = triangulate_rect_grid(top_points, grid_dims)
    m2v, m2f = triangulate_rect_grid(bottom_points, grid_dims)
    sv, sf = fill_sides(top_points, bottom_points, grid_dims)

    full = tm.util.append_faces(vertices_seq=[m1v, m2v, sv], faces_seq=[m1f, m2f, sf])
    mesh = tm.Trimesh(vertices=full[0], faces=full[1])
    mesh.fix_normals()

    return mesh
