"""Tests for batch operations and parallel processing."""

from unittest.mock import Mock, patch

from copick_utils.converters.converter_common import create_batch_converter
from copick_utils.converters.sphere_from_picks import sphere_from_picks


class TestBatchOperations:
    """Test batch processing functionality."""

    def test_batch_converter_creation(self):
        """Test that batch converter is created correctly."""

        # Create a mock converter function
        def mock_converter(points, run, object_name, session_id, user_id, **kwargs):
            return Mock(), {"vertices_created": 100, "faces_created": 50}

        batch_func = create_batch_converter(mock_converter, "Test conversion", "test", min_points=4)

        assert callable(batch_func)

    def test_batch_converter_with_real_function(self, temp_copick_project):
        """Test batch converter with real sphere function."""
        root, temp_path = temp_copick_project

        # Create batch converter for sphere_from_picks
        batch_func = create_batch_converter(
            sphere_from_picks,
            "Converting picks to sphere meshes",
            "sphere",
            min_points=4,
        )

        # Create mock conversion tasks
        run = root.get_run("test_run")

        # Get actual picks from the test data
        run.get_picks("sphere-points", "test", "single")[0]

        from copick_utils.cli.input_output_selection import ConversionTask

        tasks = [
            ConversionTask(
                run=run,
                input_pick_object_name="sphere-points",
                input_pick_user_id="test",
                input_pick_session_id="single",
                output_mesh_object_name="sphere",
                output_mesh_user_id="test",
                output_mesh_session_id="batch-test",
            ),
        ]

        # Run batch conversion
        results = batch_func(root=root, conversion_tasks=tasks, run_names=["test_run"], workers=1, subdivisions=1)

        assert len(results) == 1
        assert "test_run" in results
        assert results["test_run"]["processed"] == 1

    def test_batch_error_handling(self, temp_copick_project):
        """Test batch processing error handling."""
        root, temp_path = temp_copick_project

        # Create a converter that always fails
        def failing_converter(points, run, object_name, session_id, user_id, **kwargs):
            raise ValueError("Test error")

        batch_func = create_batch_converter(failing_converter, "Failing conversion", "fail", min_points=1)

        run = root.get_run("test_run")

        from copick_utils.cli.input_output_selection import ConversionTask

        tasks = [
            ConversionTask(
                run=run,
                input_pick_object_name="sphere-points",
                input_pick_user_id="test",
                input_pick_session_id="single",
                output_mesh_object_name="fail",
                output_mesh_user_id="test",
                output_mesh_session_id="fail-test",
            ),
        ]

        # Run batch conversion - should handle errors gracefully
        results = batch_func(root=root, conversion_tasks=tasks, run_names=["test_run"], workers=1)

        assert len(results) == 1
        assert "test_run" in results
        assert results["test_run"]["processed"] == 0
        assert len(results["test_run"]["errors"]) > 0
        assert "Test error" in results["test_run"]["errors"][0]

    def test_empty_task_list(self, temp_copick_project):
        """Test batch processing with empty task list."""
        root, temp_path = temp_copick_project

        batch_func = create_batch_converter(sphere_from_picks, "Empty test", "sphere", min_points=4)

        results = batch_func(root=root, conversion_tasks=[], run_names=["test_run"], workers=1)

        assert len(results) == 1
        assert "test_run" in results
        assert results["test_run"]["processed"] == 0

    def test_multiple_runs_batch(self, temp_copick_project):
        """Test batch processing across multiple runs."""
        root, temp_path = temp_copick_project

        # Create a second run with data
        run2 = root.new_run("test_run_2")

        # Copy some picks to the second run
        original_picks = root.get_run("test_run").get_picks("sphere-points", "test", "single")[0]
        new_picks = run2.new_picks("sphere-points", "test", "single", exist_ok=True)
        new_picks.from_numpy(original_picks.numpy())

        batch_func = create_batch_converter(sphere_from_picks, "Multi-run test", "sphere", min_points=4)

        from copick_utils.cli.input_output_selection import ConversionTask

        tasks = [
            ConversionTask(
                run=root.get_run("test_run"),
                input_pick_object_name="sphere-points",
                input_pick_user_id="test",
                input_pick_session_id="single",
                output_mesh_object_name="sphere",
                output_mesh_user_id="test",
                output_mesh_session_id="multi-1",
            ),
            ConversionTask(
                run=run2,
                input_pick_object_name="sphere-points",
                input_pick_user_id="test",
                input_pick_session_id="single",
                output_mesh_object_name="sphere",
                output_mesh_user_id="test",
                output_mesh_session_id="multi-2",
            ),
        ]

        results = batch_func(
            root=root,
            conversion_tasks=tasks,
            run_names=["test_run", "test_run_2"],
            workers=1,
            subdivisions=1,
        )

        assert len(results) == 2
        assert "test_run" in results
        assert "test_run_2" in results
        assert results["test_run"]["processed"] == 1
        assert results["test_run_2"]["processed"] == 1

    @patch("copick_utils.converters.converter_common.logger")
    def test_batch_logging(self, mock_logger, temp_copick_project):
        """Test that batch operations log appropriately."""
        root, temp_path = temp_copick_project

        batch_func = create_batch_converter(sphere_from_picks, "Logging test", "sphere", min_points=4)

        run = root.get_run("test_run")

        from copick_utils.cli.input_output_selection import ConversionTask

        tasks = [
            ConversionTask(
                run=run,
                input_pick_object_name="sphere-points",
                input_pick_user_id="test",
                input_pick_session_id="single",
                output_mesh_object_name="sphere",
                output_mesh_user_id="test",
                output_mesh_session_id="log-test",
            ),
        ]

        batch_func(root=root, conversion_tasks=tasks, run_names=["test_run"], workers=1, subdivisions=1)

        # Verify logging was called
        assert mock_logger.info.called

        # Check some of the log messages
        log_messages = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Starting" in msg for msg in log_messages)
        assert any("Completed" in msg for msg in log_messages)
