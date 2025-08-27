"""Test configuration and fixtures for copick-utils tests."""

import tempfile
from pathlib import Path
from typing import Generator, List, Tuple

import copick
import numpy as np
import pytest


def create_hollow_spheres_volume() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Creates a 256^3 uint8 volume with 3 hollow spheres at random locations.
    Also returns points that lie in the center of each shell.
    """
    # Set random seed for reproducibility
    np.random.seed(43)

    # Create 256^3 volume of zeros
    volume_size = 256
    volume = np.zeros((volume_size, volume_size, volume_size), dtype=np.uint8)

    # Generate 3 random sphere centers
    # Keep them away from edges to avoid clipping
    margin = 40
    centers = np.random.randint(margin, volume_size - margin, size=(3, 3))

    # Generate random outer radii for each sphere
    min_outer_radius = 15
    max_outer_radius = 35
    outer_radii = np.random.randint(min_outer_radius, max_outer_radius, size=3)

    # Generate inner radii (70-85% of outer radius for good hollow effect)
    inner_ratio = np.random.uniform(0.7, 0.85, size=3)
    inner_radii = (outer_radii * inner_ratio).astype(int)

    # Create coordinate grids for the entire volume
    x, y, z = np.meshgrid(
        np.arange(volume_size),
        np.arange(volume_size),
        np.arange(volume_size),
        indexing="ij",
    )

    # Create each hollow sphere
    for i in range(3):
        cx, cy, cz = centers[i]
        outer_r = outer_radii[i]
        inner_r = inner_radii[i]

        # Calculate distance from sphere center for all voxels
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)

        # Create hollow sphere mask: points between inner and outer radius
        hollow_sphere_mask = (distance <= outer_r) & (distance >= inner_r)

        # Set hollow sphere voxels to 255 (maximum uint8 value)
        volume[hollow_sphere_mask] = 1

    # Generate points in the center of each shell
    shell_center_points = []

    for i in range(3):
        cx, cy, cz = centers[i]
        outer_r = outer_radii[i]
        inner_r = inner_radii[i]

        # Calculate middle radius (center of shell)
        middle_radius = (outer_r + inner_r) / 2.0

        # Generate points on sphere at middle radius
        # Use spherical coordinates to generate evenly distributed points
        num_points = 50  # Number of points per sphere (adjust as needed)

        # Generate random spherical coordinates
        theta = np.random.uniform(0, 2 * np.pi, num_points)  # azimuthal angle
        phi = np.random.uniform(0, np.pi, num_points)  # polar angle

        # Convert to cartesian coordinates
        x_points = middle_radius * np.sin(phi) * np.cos(theta) + cx
        y_points = middle_radius * np.sin(phi) * np.sin(theta) + cy
        z_points = middle_radius * np.cos(phi) + cz

        # Stack coordinates into (N, 3) array
        sphere_points = np.column_stack((x_points, y_points, z_points))
        shell_center_points.append(sphere_points)

    return volume, centers, outer_radii, inner_radii, shell_center_points


def create_test_points() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create different types of point sets for testing."""
    np.random.seed(42)

    # Sphere points - clustered around origin
    sphere_points = np.random.randn(30, 3) * 10 + np.array([50, 50, 50])

    # Plane points - roughly planar
    plane_points = np.random.randn(25, 3)
    plane_points[:, 2] *= 2  # Make z-variation smaller
    plane_points += np.array([100, 100, 100])

    # Ellipsoid points - elongated in one direction
    ellipsoid_points = np.random.randn(40, 3)
    ellipsoid_points[:, 0] *= 20  # Elongate in x
    ellipsoid_points[:, 1] *= 10  # Medium in y
    ellipsoid_points[:, 2] *= 5  # Narrow in z
    ellipsoid_points += np.array([150, 150, 150])

    # Surface points - distributed on a curved surface
    t = np.linspace(0, 2 * np.pi, 35)
    s = np.linspace(0, np.pi, 20)
    t_grid, s_grid = np.meshgrid(t[:20], s[:20])
    surface_x = 15 * np.sin(s_grid) * np.cos(t_grid) + 200
    surface_y = 15 * np.sin(s_grid) * np.sin(t_grid) + 200
    surface_z = 15 * np.cos(s_grid) + 200
    surface_points = np.column_stack([surface_x.flatten(), surface_y.flatten(), surface_z.flatten()])

    # Mesh points - for convex hull/alpha shape
    mesh_points = np.array(
        [
            [0, 0, 0],
            [10, 0, 0],
            [5, 10, 0],
            [5, 5, 10],
            [2, 2, 2],
            [8, 2, 2],
            [5, 8, 2],
            [5, 5, 8],
        ],
        dtype=float,
    ) + np.array([250, 250, 250])

    return sphere_points, plane_points, ellipsoid_points, surface_points, mesh_points


@pytest.fixture
def temp_copick_project() -> Generator[Tuple[copick.models.CopickRoot, Path], None, None]:
    """Create a temporary copick project with test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        project_path = temp_path / "project"
        project_path.mkdir()

        # Create copick config
        config = {
            "name": "test-picks-to-mesh",
            "description": "Test project for picks to mesh conversion",
            "version": "1.0.0",
            "pickable_objects": [
                {
                    "name": "sphere-points",
                    "is_particle": True,
                    "label": 1,
                    "color": [255, 204, 153, 255],
                    "radius": 10.0,
                },
                {
                    "name": "sphere",
                    "is_particle": False,
                    "label": 2,
                    "color": [0, 92, 49, 255],
                    "radius": 160.0,
                },
                {
                    "name": "ellipsoid-points",
                    "is_particle": True,
                    "label": 3,
                    "color": [146, 96, 191, 255],
                    "radius": 50.0,
                },
                {
                    "name": "ellipsoid",
                    "is_particle": False,
                    "label": 4,
                    "color": [200, 100, 100, 255],
                    "radius": 50.0,
                },
                {
                    "name": "plane-points",
                    "is_particle": True,
                    "label": 5,
                    "color": [200, 100, 100, 255],
                    "radius": 50.0,
                },
                {
                    "name": "plane",
                    "is_particle": False,
                    "label": 6,
                    "color": [200, 100, 100, 255],
                    "radius": 50.0,
                },
                {
                    "name": "surface-points",
                    "is_particle": False,
                    "label": 7,
                    "color": [200, 100, 100, 255],
                    "radius": 50.0,
                },
                {
                    "name": "surface",
                    "is_particle": False,
                    "label": 8,
                    "color": [200, 100, 100, 255],
                    "radius": 50.0,
                },
                {
                    "name": "mesh-points",
                    "is_particle": True,
                    "label": 9,
                    "color": [100, 200, 100, 255],
                    "radius": 20.0,
                },
            ],
            "config_type": "filesystem",
            "overlay_root": f"local://{project_path}",
            "overlay_fs_args": {
                "auto_mkdir": True,
            },
        }

        config_path = temp_path / "config.json"
        import json

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Create copick root
        root = copick.from_file(str(config_path))

        # Create test run
        run = root.new_run("test_run")

        # Create test data
        sphere_points, plane_points, ellipsoid_points, surface_points, mesh_points = create_test_points()

        # Create picks for different point types
        picks_data = {
            "sphere-points": [
                ("single", sphere_points),
                ("cluster-0", sphere_points[:15]),
                ("cluster-1", sphere_points[15:]),
            ],
            "plane-points": [
                ("single", plane_points),
            ],
            "ellipsoid-points": [
                ("single", ellipsoid_points),
            ],
            "surface-points": [
                ("single", surface_points),
            ],
            "mesh-points": [
                ("single", mesh_points),
                ("alpha", mesh_points + np.random.randn(*mesh_points.shape) * 2),
            ],
        }

        for object_name, sessions in picks_data.items():
            for session_id, points in sessions:
                picks = run.new_picks(
                    object_name=object_name,
                    user_id="test",
                    session_id=session_id,
                    exist_ok=True,
                )
                # Convert to ZYX order and scale by voxel size
                points_zyx = points[:, [2, 1, 0]] * 3.7
                picks.from_numpy(points_zyx)

        # Also create the hollow spheres data
        volume, centers, outer_radii, inner_radii, shell_center_points = create_hollow_spheres_volume()

        for idx, pointset in enumerate(shell_center_points):
            # Flip the order to match (z, y, x) indexing
            pointset = pointset[:, [2, 1, 0]]
            picks = run.new_picks(
                object_name="sphere-points",
                user_id="sim",
                session_id=f"sphere-{idx}",
                exist_ok=True,
            )
            picks.from_numpy(np.array(pointset) * 3.7)

        # Create combined sphere points
        all_spheres = np.concatenate(shell_center_points, axis=0)
        all_spheres = all_spheres[:, [2, 1, 0]]
        picks = run.new_picks(
            object_name="sphere-points",
            user_id="sim",
            session_id="all-spheres",
            exist_ok=True,
        )
        picks.from_numpy(np.array(all_spheres) * 3.7)

        yield root, temp_path


@pytest.fixture
def sample_points():
    """Provide sample point sets for testing."""
    return create_test_points()
