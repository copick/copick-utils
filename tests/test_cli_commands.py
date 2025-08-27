"""Tests for picks to mesh CLI commands."""

from click.testing import CliRunner
from copick_utils.cli.picks_to_mesh_commands import (
    picks2ellipsoid,
    picks2mesh,
    picks2plane,
    picks2sphere,
    picks2surface,
)


class TestPicksToMeshCLI:
    """Test picks2mesh CLI command."""

    def test_picks2mesh_convex_hull(self, temp_copick_project):
        """Test picks2mesh with convex hull."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2mesh,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "mesh-points",
                "--pick-user-id",
                "test",
                "--pick-session-id",
                "single",
                "--mesh-type",
                "convex_hull",
                "--mesh-session-id",
                "mesh-convex",
            ],
        )

        assert result.exit_code == 0

        # Check that mesh was created
        run = root.get_run("test_run")
        mesh = run.get_mesh("mesh-points", "mesh-convex", "picks2mesh")
        assert mesh is not None

    def test_picks2mesh_alpha_shape(self, temp_copick_project):
        """Test picks2mesh with alpha shape."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2mesh,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "mesh-points",
                "--pick-user-id",
                "test",
                "--pick-session-id",
                "alpha",
                "--mesh-type",
                "alpha_shape",
                "--alpha",
                "15.0",
                "--mesh-session-id",
                "mesh-alpha",
            ],
        )

        assert result.exit_code == 0

        # Check that mesh was created
        run = root.get_run("test_run")
        mesh = run.get_mesh("mesh-points", "mesh-alpha", "picks2mesh")
        assert mesh is not None

    def test_picks2mesh_with_clustering(self, temp_copick_project):
        """Test picks2mesh with clustering."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2mesh,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "sphere-points",
                "--pick-user-id",
                "sim",
                "--pick-session-id",
                "all-spheres",
                "--mesh-type",
                "convex_hull",
                "--use-clustering",
                "--clustering-method",
                "dbscan",
                "--clustering-eps",
                "30.0",
                "--clustering-min-samples",
                "10",
                "--all-clusters",
                "--mesh-session-id",
                "mesh-clustered",
            ],
        )

        assert result.exit_code == 0

    def test_picks2mesh_pattern_matching(self, temp_copick_project):
        """Test picks2mesh with pattern matching."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2mesh,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "sphere-points",
                "--pick-user-id",
                "sim",
                "--pick-session-id",
                "sphere-.*",
                "--mesh-type",
                "convex_hull",
                "--mesh-session-id",
                "mesh-{input_session_id}",
            ],
        )

        assert result.exit_code == 0

        # Check that multiple meshes were created
        run = root.get_run("test_run")
        for i in range(3):
            mesh = run.get_mesh("sphere-points", f"mesh-sphere-{i}", "picks2mesh")
            assert mesh is not None

    def test_picks2mesh_missing_alpha(self, temp_copick_project):
        """Test picks2mesh fails without alpha for alpha_shape."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2mesh,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "mesh-points",
                "--pick-user-id",
                "test",
                "--pick-session-id",
                "single",
                "--mesh-type",
                "alpha_shape",
                "--mesh-session-id",
                "mesh-alpha",
            ],
        )

        assert result.exit_code != 0
        assert "Alpha parameter is required" in result.output


class TestPicksToSphereCLI:
    """Test picks2sphere CLI command."""

    def test_picks2sphere_basic(self, temp_copick_project):
        """Test basic picks2sphere functionality."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2sphere,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "sphere-points",
                "--pick-user-id",
                "test",
                "--pick-session-id",
                "single",
                "--mesh-session-id",
                "sphere-basic",
                "--subdivisions",
                "1",
            ],
        )

        assert result.exit_code == 0

        # Check that mesh was created
        run = root.get_run("test_run")
        mesh = run.get_mesh("sphere-points", "sphere-basic", "picks2sphere")
        assert mesh is not None

    def test_picks2sphere_with_clustering(self, temp_copick_project):
        """Test picks2sphere with clustering."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2sphere,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "sphere-points",
                "--pick-user-id",
                "sim",
                "--pick-session-id",
                "all-spheres",
                "--use-clustering",
                "--clustering-method",
                "dbscan",
                "--clustering-eps",
                "30.0",
                "--clustering-min-samples",
                "10",
                "--all-clusters",
                "--mesh-session-id",
                "sphere-clustered",
                "--subdivisions",
                "1",
            ],
        )

        assert result.exit_code == 0

    def test_picks2sphere_individual_meshes(self, temp_copick_project):
        """Test picks2sphere with individual meshes."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2sphere,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "sphere-points",
                "--pick-user-id",
                "sim",
                "--pick-session-id",
                "all-spheres",
                "--use-clustering",
                "--clustering-method",
                "dbscan",
                "--clustering-eps",
                "30.0",
                "--clustering-min-samples",
                "10",
                "--all-clusters",
                "--individual-meshes",
                "--mesh-session-id",
                "sphere-{instance_id}",
                "--subdivisions",
                "1",
            ],
        )

        assert result.exit_code == 0

    def test_picks2sphere_deduplication(self, temp_copick_project):
        """Test picks2sphere with sphere deduplication."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2sphere,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "sphere-points",
                "--pick-user-id",
                "sim",
                "--pick-session-id",
                "all-spheres",
                "--use-clustering",
                "--clustering-method",
                "dbscan",
                "--clustering-eps",
                "30.0",
                "--clustering-min-samples",
                "10",
                "--all-clusters",
                "--deduplicate-spheres",
                "--min-sphere-distance",
                "50.0",
                "--mesh-session-id",
                "sphere-dedup",
                "--subdivisions",
                "1",
            ],
        )

        assert result.exit_code == 0


class TestPicksToEllipsoidCLI:
    """Test picks2ellipsoid CLI command."""

    def test_picks2ellipsoid_basic(self, temp_copick_project):
        """Test basic picks2ellipsoid functionality."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2ellipsoid,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "ellipsoid-points",
                "--pick-user-id",
                "test",
                "--pick-session-id",
                "single",
                "--mesh-session-id",
                "ellipsoid-basic",
                "--subdivisions",
                "1",
            ],
        )

        assert result.exit_code == 0

        # Check that mesh was created
        run = root.get_run("test_run")
        mesh = run.get_mesh("ellipsoid-points", "ellipsoid-basic", "picks2ellipsoid")
        assert mesh is not None

    def test_picks2ellipsoid_with_deduplication(self, temp_copick_project):
        """Test picks2ellipsoid with deduplication."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2ellipsoid,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "ellipsoid-points",
                "--pick-user-id",
                "test",
                "--pick-session-id",
                "single",
                "--deduplicate-ellipsoids",
                "--min-ellipsoid-distance",
                "20.0",
                "--mesh-session-id",
                "ellipsoid-dedup",
                "--subdivisions",
                "1",
            ],
        )

        assert result.exit_code == 0


class TestPicksToPlaneCLI:
    """Test picks2plane CLI command."""

    def test_picks2plane_basic(self, temp_copick_project):
        """Test basic picks2plane functionality."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2plane,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "plane-points",
                "--pick-user-id",
                "test",
                "--pick-session-id",
                "single",
                "--mesh-session-id",
                "plane-basic",
            ],
        )

        assert result.exit_code == 0

        # Check that mesh was created
        run = root.get_run("test_run")
        mesh = run.get_mesh("plane-points", "plane-basic", "picks2plane")
        assert mesh is not None

    def test_picks2plane_with_padding(self, temp_copick_project):
        """Test picks2plane with custom padding."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2plane,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "plane-points",
                "--pick-user-id",
                "test",
                "--pick-session-id",
                "single",
                "--padding",
                "2.0",
                "--mesh-session-id",
                "plane-padded",
            ],
        )

        assert result.exit_code == 0


class TestPicksToSurfaceCLI:
    """Test picks2surface CLI command."""

    def test_picks2surface_delaunay(self, temp_copick_project):
        """Test picks2surface with Delaunay triangulation."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2surface,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "surface-points",
                "--pick-user-id",
                "test",
                "--pick-session-id",
                "single",
                "--surface-method",
                "delaunay",
                "--mesh-session-id",
                "surface-delaunay",
            ],
        )

        assert result.exit_code == 0

        # Check that mesh was created
        run = root.get_run("test_run")
        mesh = run.get_mesh("surface-points", "surface-delaunay", "picks2surface")
        assert mesh is not None

    def test_picks2surface_rbf(self, temp_copick_project):
        """Test picks2surface with RBF interpolation."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2surface,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "surface-points",
                "--pick-user-id",
                "test",
                "--pick-session-id",
                "single",
                "--surface-method",
                "rbf",
                "--grid-resolution",
                "10",
                "--mesh-session-id",
                "surface-rbf",
            ],
        )

        assert result.exit_code == 0

    def test_picks2surface_grid(self, temp_copick_project):
        """Test picks2surface with grid interpolation."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2surface,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "surface-points",
                "--pick-user-id",
                "test",
                "--pick-session-id",
                "single",
                "--surface-method",
                "grid",
                "--grid-resolution",
                "10",
                "--mesh-session-id",
                "surface-grid",
            ],
        )

        assert result.exit_code == 0


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_missing_picks(self, temp_copick_project):
        """Test handling of missing picks."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2mesh,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "nonexistent",
                "--pick-user-id",
                "test",
                "--pick-session-id",
                "single",
                "--mesh-type",
                "convex_hull",
                "--mesh-session-id",
                "mesh-missing",
            ],
        )

        # Should complete but with warning about no matching picks
        assert result.exit_code == 0
        assert "No matching picks found" in result.output

    def test_invalid_config(self):
        """Test handling of invalid config file."""
        runner = CliRunner()
        result = runner.invoke(
            picks2mesh,
            [
                "--config",
                "/nonexistent/config.json",
                "--pick-object-name",
                "test",
                "--pick-user-id",
                "test",
                "--pick-session-id",
                "single",
                "--mesh-type",
                "convex_hull",
                "--mesh-session-id",
                "mesh-test",
            ],
        )

        assert result.exit_code != 0

    def test_placeholder_validation_error(self, temp_copick_project):
        """Test placeholder validation error."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        runner = CliRunner()
        result = runner.invoke(
            picks2sphere,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "sphere-points",
                "--pick-user-id",
                "test",
                "--pick-session-id",
                "single",
                "--individual-meshes",
                "--mesh-session-id",
                "invalid-template",  # Missing {instance_id}
            ],
        )

        assert result.exit_code != 0
        assert "instance_id" in result.output


class TestCLIIntegration:
    """Test CLI integration scenarios."""

    def test_full_workflow_sphere_detection(self, temp_copick_project):
        """Test full workflow: pick points -> cluster -> create spheres."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        # Step 1: Create sphere meshes from clustered points
        runner = CliRunner()
        result = runner.invoke(
            picks2sphere,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "sphere-points",
                "--pick-user-id",
                "sim",
                "--pick-session-id",
                "all-spheres",
                "--use-clustering",
                "--clustering-method",
                "dbscan",
                "--clustering-eps",
                "30.0",
                "--clustering-min-samples",
                "10",
                "--all-clusters",
                "--individual-meshes",
                "--mesh-object-name",
                "sphere",
                "--mesh-session-id",
                "detected-{instance_id}",
                "--subdivisions",
                "1",
            ],
        )

        assert result.exit_code == 0

        # Verify multiple sphere meshes were created
        run = root.get_run("test_run")
        sphere_meshes = run.get_meshes(object_name="sphere", user_id="picks2sphere")
        assert len(sphere_meshes) > 0

    def test_pattern_based_batch_processing(self, temp_copick_project):
        """Test pattern-based batch processing of multiple sessions."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        # Process all individual sphere sessions at once
        runner = CliRunner()
        result = runner.invoke(
            picks2sphere,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "sphere-points",
                "--pick-user-id",
                "sim",
                "--pick-session-id",
                "sphere-.*",  # Matches sphere-0, sphere-1, sphere-2
                "--mesh-object-name",
                "sphere",
                "--mesh-session-id",
                "auto-{input_session_id}",
                "--subdivisions",
                "1",
            ],
        )

        assert result.exit_code == 0

        # Verify meshes were created for each matched session
        run = root.get_run("test_run")
        for i in range(3):
            mesh = run.get_mesh("sphere", f"auto-sphere-{i}", "picks2sphere")
            assert mesh is not None

    def test_different_mesh_types_same_points(self, temp_copick_project):
        """Test creating different mesh types from the same points."""
        root, temp_path = temp_copick_project
        config_path = temp_path / "config.json"

        # Create convex hull mesh
        runner = CliRunner()
        result1 = runner.invoke(
            picks2mesh,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "mesh-points",
                "--pick-user-id",
                "test",
                "--pick-session-id",
                "single",
                "--mesh-type",
                "convex_hull",
                "--mesh-session-id",
                "comparison-convex",
            ],
        )

        # Create alpha shape mesh
        result2 = runner.invoke(
            picks2mesh,
            [
                "--config",
                str(config_path),
                "--pick-object-name",
                "mesh-points",
                "--pick-user-id",
                "test",
                "--pick-session-id",
                "single",
                "--mesh-type",
                "alpha_shape",
                "--alpha",
                "15.0",
                "--mesh-session-id",
                "comparison-alpha",
            ],
        )

        assert result1.exit_code == 0
        assert result2.exit_code == 0

        # Verify both meshes exist
        run = root.get_run("test_run")
        convex_mesh = run.get_mesh("mesh-points", "comparison-convex", "picks2mesh")
        alpha_mesh = run.get_mesh("mesh-points", "comparison-alpha", "picks2mesh")

        assert convex_mesh is not None
        assert alpha_mesh is not None
