"""Tests for picks to mesh converter functions."""

import numpy as np
import trimesh as tm
from copick_utils.converters.ellipsoid_from_picks import (
    create_ellipsoid_mesh,
    deduplicate_ellipsoids,
    ellipsoid_from_picks,
    fit_ellipsoid_to_points,
)
from copick_utils.converters.mesh_from_picks import mesh_from_picks
from copick_utils.converters.plane_from_picks import create_plane_mesh, fit_plane_to_points, plane_from_picks
from copick_utils.converters.sphere_from_picks import (
    create_sphere_mesh,
    deduplicate_spheres,
    fit_sphere_to_points,
    sphere_from_picks,
)
from copick_utils.converters.surface_from_picks import (
    delaunay_surface,
    fit_2d_surface_to_points,
    grid_surface,
    rbf_surface,
    surface_from_picks,
)


class TestMeshFromPicks:
    """Test mesh_from_picks converter functions."""

    def test_convex_hull_mesh(self, temp_copick_project, sample_points):
        """Test convex hull mesh creation."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")
        _, _, _, _, mesh_points = sample_points

        result = mesh_from_picks(mesh_points, run, "test-mesh", "test-session", "test-user", mesh_type="convex_hull")

        assert result is not None
        mesh_obj, stats = result
        assert mesh_obj is not None
        assert stats["vertices_created"] > 0
        assert stats["faces_created"] > 0

        # Check that mesh was stored
        stored_mesh = run.get_mesh("test-mesh", "test-session", "test-user")
        assert stored_mesh is not None
        assert stored_mesh.mesh is not None

    def test_alpha_shape_mesh(self, temp_copick_project, sample_points):
        """Test alpha shape mesh creation."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")
        _, _, _, _, mesh_points = sample_points

        result = mesh_from_picks(
            mesh_points,
            run,
            "test-alpha",
            "test-session",
            "test-user",
            mesh_type="alpha_shape",
            alpha=15.0,
        )

        assert result is not None
        mesh_obj, stats = result
        assert mesh_obj is not None
        assert stats["vertices_created"] > 0
        assert stats["faces_created"] > 0

    def test_mesh_with_clustering(self, temp_copick_project, sample_points):
        """Test mesh creation with clustering."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")
        sphere_points, _, _, _, _ = sample_points

        # Add some noise to create clusters
        clustered_points = np.vstack([sphere_points + np.array([0, 0, 0]), sphere_points + np.array([100, 0, 0])])

        result = mesh_from_picks(
            clustered_points,
            run,
            "test-clustered",
            "test-session",
            "test-user",
            mesh_type="convex_hull",
            use_clustering=True,
            clustering_method="dbscan",
            clustering_params={"eps": 15.0, "min_samples": 5},
            all_clusters=True,
        )

        assert result is not None
        mesh_obj, stats = result
        assert mesh_obj is not None
        assert stats["vertices_created"] > 0

    def test_insufficient_points(self, temp_copick_project):
        """Test handling of insufficient points."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")

        # Only 2 points - insufficient for mesh
        points = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)

        result = mesh_from_picks(points, run, "test-insufficient", "test-session", "test-user", mesh_type="convex_hull")

        assert result is None


class TestSphereFromPicks:
    """Test sphere_from_picks converter functions."""

    def test_fit_sphere_to_points(self, sample_points):
        """Test sphere fitting function."""
        sphere_points, _, _, _, _ = sample_points

        center, radius = fit_sphere_to_points(sphere_points)

        assert center.shape == (3,)
        assert radius > 0
        assert np.allclose(center, [50, 50, 50], atol=20)

    def test_create_sphere_mesh(self):
        """Test sphere mesh creation."""
        center = np.array([10, 20, 30])
        radius = 15.0

        mesh = create_sphere_mesh(center, radius, subdivisions=1)

        assert isinstance(mesh, tm.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        assert mesh.is_valid

    def test_deduplicate_spheres(self):
        """Test sphere deduplication."""
        # Create overlapping spheres
        spheres = [
            (np.array([0, 0, 0]), 10.0),
            (np.array([5, 0, 0]), 12.0),  # Close to first
            (np.array([100, 0, 0]), 8.0),  # Far away
        ]

        deduplicated = deduplicate_spheres(spheres, min_distance=15.0)

        assert len(deduplicated) == 2  # Should merge first two

    def test_sphere_from_picks(self, temp_copick_project, sample_points):
        """Test complete sphere creation from picks."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")
        sphere_points, _, _, _, _ = sample_points

        result = sphere_from_picks(sphere_points, run, "test-sphere", "test-session", "test-user", subdivisions=1)

        assert result is not None
        mesh_obj, stats = result
        assert mesh_obj is not None
        assert stats["vertices_created"] > 0
        assert stats["faces_created"] > 0

    def test_sphere_with_clustering(self, temp_copick_project, sample_points):
        """Test sphere creation with clustering."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")
        sphere_points, _, _, _, _ = sample_points

        # Create multiple clusters
        clustered_points = np.vstack([sphere_points, sphere_points + np.array([200, 0, 0])])

        result = sphere_from_picks(
            clustered_points,
            run,
            "test-sphere-clustered",
            "test-session",
            "test-user",
            use_clustering=True,
            clustering_method="dbscan",
            clustering_params={"eps": 20.0, "min_samples": 5},
            all_clusters=True,
            subdivisions=1,
        )

        assert result is not None
        mesh_obj, stats = result
        assert mesh_obj is not None

    def test_individual_sphere_meshes(self, temp_copick_project, sample_points):
        """Test individual sphere mesh creation."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")
        sphere_points, _, _, _, _ = sample_points

        # Create multiple clusters
        clustered_points = np.vstack([sphere_points, sphere_points + np.array([200, 0, 0])])

        result = sphere_from_picks(
            clustered_points,
            run,
            "test-individual",
            "base-session",
            "test-user",
            use_clustering=True,
            clustering_method="dbscan",
            clustering_params={"eps": 20.0, "min_samples": 5},
            all_clusters=True,
            individual_meshes=True,
            session_id_template="{base_session_id}-{sphere_id}",
            subdivisions=1,
        )

        assert result is not None
        # Should create multiple individual meshes
        mesh_0 = run.get_mesh("test-individual", "base-session-000", "test-user")
        assert mesh_0 is not None


class TestEllipsoidFromPicks:
    """Test ellipsoid_from_picks converter functions."""

    def test_fit_ellipsoid_to_points(self, sample_points):
        """Test ellipsoid fitting function."""
        _, _, ellipsoid_points, _, _ = sample_points

        center, semi_axes, rotation_matrix = fit_ellipsoid_to_points(ellipsoid_points)

        assert center.shape == (3,)
        assert semi_axes.shape == (3,)
        assert rotation_matrix.shape == (3, 3)
        assert np.all(semi_axes > 0)

    def test_create_ellipsoid_mesh(self):
        """Test ellipsoid mesh creation."""
        center = np.array([10, 20, 30])
        semi_axes = np.array([20, 15, 10])
        rotation_matrix = np.eye(3)

        mesh = create_ellipsoid_mesh(center, semi_axes, rotation_matrix, subdivisions=1)

        assert isinstance(mesh, tm.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_deduplicate_ellipsoids(self):
        """Test ellipsoid deduplication."""
        ellipsoids = [
            (np.array([0, 0, 0]), np.array([10, 8, 6]), np.eye(3)),
            (np.array([5, 0, 0]), np.array([12, 9, 7]), np.eye(3)),  # Close
            (np.array([100, 0, 0]), np.array([8, 6, 4]), np.eye(3)),  # Far
        ]

        deduplicated = deduplicate_ellipsoids(ellipsoids, min_distance=15.0)

        assert len(deduplicated) == 2

    def test_ellipsoid_from_picks(self, temp_copick_project, sample_points):
        """Test complete ellipsoid creation from picks."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")
        _, _, ellipsoid_points, _, _ = sample_points

        result = ellipsoid_from_picks(
            ellipsoid_points,
            run,
            "test-ellipsoid",
            "test-session",
            "test-user",
            subdivisions=1,
        )

        assert result is not None
        mesh_obj, stats = result
        assert mesh_obj is not None
        assert stats["vertices_created"] > 0
        assert stats["faces_created"] > 0


class TestPlaneFromPicks:
    """Test plane_from_picks converter functions."""

    def test_fit_plane_to_points(self, sample_points):
        """Test plane fitting function."""
        _, plane_points, _, _, _ = sample_points

        center, normal = fit_plane_to_points(plane_points)

        assert center.shape == (3,)
        assert normal.shape == (3,)
        assert np.allclose(np.linalg.norm(normal), 1.0)

    def test_create_plane_mesh(self, sample_points):
        """Test plane mesh creation."""
        _, plane_points, _, _, _ = sample_points
        center = np.mean(plane_points, axis=0)
        normal = np.array([0, 0, 1])

        mesh = create_plane_mesh(center, normal, plane_points, padding=1.2)

        assert isinstance(mesh, tm.Trimesh)
        assert len(mesh.vertices) == 4  # Plane should have 4 vertices
        assert len(mesh.faces) == 2  # And 2 triangular faces

    def test_plane_from_picks(self, temp_copick_project, sample_points):
        """Test complete plane creation from picks."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")
        _, plane_points, _, _, _ = sample_points

        result = plane_from_picks(plane_points, run, "test-plane", "test-session", "test-user", padding=1.5)

        assert result is not None
        mesh_obj, stats = result
        assert mesh_obj is not None
        assert stats["vertices_created"] > 0
        assert stats["faces_created"] > 0


class TestSurfaceFromPicks:
    """Test surface_from_picks converter functions."""

    def test_delaunay_surface(self, sample_points):
        """Test Delaunay triangulation surface."""
        _, _, _, surface_points, _ = sample_points

        mesh = delaunay_surface(surface_points)

        assert isinstance(mesh, tm.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_rbf_surface(self, sample_points):
        """Test RBF interpolation surface."""
        _, _, _, surface_points, _ = sample_points

        mesh = rbf_surface(surface_points, grid_resolution=10)

        assert isinstance(mesh, tm.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_grid_surface(self, sample_points):
        """Test grid interpolation surface."""
        _, _, _, surface_points, _ = sample_points

        mesh = grid_surface(surface_points, grid_resolution=10)

        assert isinstance(mesh, tm.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_fit_2d_surface_to_points(self, sample_points):
        """Test surface fitting with different methods."""
        _, _, _, surface_points, _ = sample_points

        for method in ["delaunay", "rbf", "grid"]:
            mesh = fit_2d_surface_to_points(surface_points, method=method, grid_resolution=10)

            assert isinstance(mesh, tm.Trimesh)
            assert len(mesh.vertices) > 0
            assert len(mesh.faces) > 0

    def test_surface_from_picks(self, temp_copick_project, sample_points):
        """Test complete surface creation from picks."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")
        _, _, _, surface_points, _ = sample_points

        result = surface_from_picks(
            surface_points,
            run,
            "test-surface",
            "test-session",
            "test-user",
            surface_method="delaunay",
        )

        assert result is not None
        mesh_obj, stats = result
        assert mesh_obj is not None
        assert stats["vertices_created"] > 0
        assert stats["faces_created"] > 0

    def test_surface_with_different_methods(self, temp_copick_project, sample_points):
        """Test surface creation with different methods."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")
        _, _, _, surface_points, _ = sample_points

        methods = ["delaunay", "rbf", "grid"]

        for method in methods:
            result = surface_from_picks(
                surface_points,
                run,
                f"test-{method}",
                "test-session",
                "test-user",
                surface_method=method,
                grid_resolution=10,
            )

            assert result is not None
            mesh_obj, stats = result
            assert mesh_obj is not None


class TestErrorHandling:
    """Test error handling in converter functions."""

    def test_insufficient_points_sphere(self, temp_copick_project):
        """Test sphere with insufficient points."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")

        # Only 2 points - insufficient for sphere (needs 4)
        points = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)

        result = sphere_from_picks(points, run, "test", "test", "test")
        assert result is None

    def test_insufficient_points_ellipsoid(self, temp_copick_project):
        """Test ellipsoid with insufficient points."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")

        # Only 3 points - insufficient for ellipsoid (needs 6)
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)

        result = ellipsoid_from_picks(points, run, "test", "test", "test")
        assert result is None

    def test_insufficient_points_plane(self, temp_copick_project):
        """Test plane with insufficient points."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")

        # Only 2 points - insufficient for plane (needs 3)
        points = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)

        result = plane_from_picks(points, run, "test", "test", "test")
        assert result is None

    def test_insufficient_points_surface(self, temp_copick_project):
        """Test surface with insufficient points."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")

        # Only 2 points - insufficient for surface (needs 3)
        points = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)

        result = surface_from_picks(points, run, "test", "test", "test")
        assert result is None
