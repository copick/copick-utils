"""Tests for input/output selection functionality."""

import pytest
from copick_utils.cli.input_output_selection import InputOutputSelector, validate_placeholders


class TestInputOutputSelector:
    """Test the InputOutputSelector class."""

    def test_one_to_one_selection(self, temp_copick_project):
        """Test one-to-one selection mode."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")

        selector = InputOutputSelector(
            pick_object_name="sphere-points",
            pick_user_id="test",
            pick_session_id="single",
            mesh_object_name="test-mesh",
            mesh_user_id="test-user",
            mesh_session_id="output-single",
            individual_meshes=False,
        )

        assert not selector._is_regex_pattern(selector.pick_session_id)
        assert selector.get_mode_description() == "one-to-one (single input → single output)"

        tasks = selector.get_conversion_tasks(run)
        assert len(tasks) == 1

        task = tasks[0]
        assert task["input_picks"].object_name == "sphere-points"
        assert task["input_picks"].user_id == "test"
        assert task["input_picks"].session_id == "single"
        assert task["mesh_object_name"] == "test-mesh"
        assert task["mesh_user_id"] == "test-user"
        assert task["mesh_session_id"] == "output-single"
        assert task["individual_meshes"] is False
        assert task["session_id_template"] is None

    def test_one_to_many_selection(self, temp_copick_project):
        """Test one-to-many selection mode."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")

        selector = InputOutputSelector(
            pick_object_name="sphere-points",
            pick_user_id="test",
            pick_session_id="single",
            mesh_object_name="test-mesh",
            mesh_user_id="test-user",
            mesh_session_id="output-{instance_id}",
            individual_meshes=True,
        )

        assert not selector._is_regex_pattern(selector.pick_session_id)
        assert selector.get_mode_description() == "one-to-many (single input → template output with individual meshes)"

        tasks = selector.get_conversion_tasks(run)
        assert len(tasks) == 1

        task = tasks[0]
        assert task["mesh_session_id"] == "output-{instance_id}"
        assert task["individual_meshes"] is True
        assert task["session_id_template"] == "output-{instance_id}"

    def test_many_to_many_selection(self, temp_copick_project):
        """Test many-to-many selection mode."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")

        selector = InputOutputSelector(
            pick_object_name="sphere-points",
            pick_user_id="sim",
            pick_session_id="sphere-.*",
            mesh_object_name="test-mesh",
            mesh_user_id="test-user",
            mesh_session_id="output-{input_session_id}",
            individual_meshes=False,
        )

        assert selector._is_regex_pattern(selector.pick_session_id)
        assert selector.get_mode_description() == "many-to-one (regex input → template output)"

        tasks = selector.get_conversion_tasks(run)
        assert len(tasks) == 3  # Should match sphere-0, sphere-1, sphere-2

        session_ids = [task["mesh_session_id"] for task in tasks]
        expected = ["output-sphere-0", "output-sphere-1", "output-sphere-2"]
        assert sorted(session_ids) == sorted(expected)

    def test_many_to_many_with_individual_meshes(self, temp_copick_project):
        """Test many-to-many with individual meshes."""
        root, temp_path = temp_copick_project
        root.get_run("test_run")

        selector = InputOutputSelector(
            pick_object_name="sphere-points",
            pick_user_id="sim",
            pick_session_id="sphere-.*",
            mesh_object_name="test-mesh",
            mesh_user_id="test-user",
            mesh_session_id="output-{input_session_id}-{instance_id}",
            individual_meshes=True,
        )

        assert selector._is_regex_pattern(selector.pick_session_id)
        assert selector.get_mode_description() == "many-to-many (regex input → template output with individual meshes)"

    def test_default_object_name(self, temp_copick_project):
        """Test default object name behavior."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")

        selector = InputOutputSelector(
            pick_object_name="sphere-points",
            pick_user_id="test",
            pick_session_id="single",
            mesh_object_name=None,  # Should default to pick_object_name
            mesh_user_id="test-user",
            mesh_session_id="output-single",
            individual_meshes=False,
        )

        tasks = selector.get_conversion_tasks(run)
        assert len(tasks) == 1
        assert tasks[0]["mesh_object_name"] == "sphere-points"

    def test_no_matches_found(self, temp_copick_project):
        """Test behavior when no matches are found."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")

        selector = InputOutputSelector(
            pick_object_name="nonexistent",
            pick_user_id="test",
            pick_session_id="single",
            mesh_object_name="test-mesh",
            mesh_user_id="test-user",
            mesh_session_id="output-single",
            individual_meshes=False,
        )

        tasks = selector.get_conversion_tasks(run)
        assert len(tasks) == 0

    def test_pattern_no_matches(self, temp_copick_project):
        """Test pattern that matches nothing."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")

        selector = InputOutputSelector(
            pick_object_name="sphere-points",
            pick_user_id="test",
            pick_session_id="nomatch-.*",
            mesh_object_name="test-mesh",
            mesh_user_id="test-user",
            mesh_session_id="output-{input_session_id}",
            individual_meshes=False,
        )

        tasks = selector.get_conversion_tasks(run)
        assert len(tasks) == 0

    def test_session_id_template_validation(self):
        """Test session ID template validation."""
        # Should pass validation for pattern with input_session_id
        InputOutputSelector(
            pick_object_name="test",
            pick_user_id="test",
            pick_session_id="test-.*",
            mesh_object_name="test",
            mesh_user_id="test",
            mesh_session_id="output-{input_session_id}",
            individual_meshes=False,
        )
        # Should not raise

        # Test with individual meshes - should require instance_id
        InputOutputSelector(
            pick_object_name="test",
            pick_user_id="test",
            pick_session_id="test-.*",
            mesh_object_name="test",
            mesh_user_id="test",
            mesh_session_id="output-{input_session_id}-{instance_id}",
            individual_meshes=True,
        )
        # Should not raise

    def test_invalid_individual_without_instance_id(self):
        """Test invalid individual meshes without instance_id."""
        with pytest.raises(ValueError, match="instance_id"):
            InputOutputSelector(
                pick_object_name="test",
                pick_user_id="test",
                pick_session_id="single",
                mesh_session_id="output",
                individual_meshes=True,
            )

    def test_invalid_pattern_without_input_session_id(self):
        """Test invalid pattern without input_session_id."""
        with pytest.raises(ValueError, match="input_session_id"):
            InputOutputSelector(
                pick_object_name="test",
                pick_user_id="test",
                pick_session_id="pattern-.*",
                mesh_session_id="output",
                individual_meshes=False,
            )

    def test_regex_pattern_detection(self):
        """Test regex pattern detection."""
        selector = InputOutputSelector(
            pick_object_name="test",
            pick_user_id="test",
            pick_session_id="test-.*",
            mesh_session_id="output-{input_session_id}",
            individual_meshes=False,
        )

        # Test various patterns
        assert selector._is_regex_pattern("test-.*")
        assert selector._is_regex_pattern("test[0-9]+")
        assert selector._is_regex_pattern("test?")
        assert selector._is_regex_pattern("test+")
        assert selector._is_regex_pattern("(test|other)")

        # Test non-patterns
        assert not selector._is_regex_pattern("simple")
        assert not selector._is_regex_pattern("test-123")
        assert not selector._is_regex_pattern("test_session")

    def test_session_id_resolution(self, temp_copick_project):
        """Test session ID template resolution."""
        root, temp_path = temp_copick_project
        root.get_run("test_run")

        selector = InputOutputSelector(
            pick_object_name="sphere-points",
            pick_user_id="sim",
            pick_session_id="sphere-.*",
            mesh_session_id="detected-{input_session_id}",
            individual_meshes=False,
        )

        # Test resolution
        resolved = selector._resolve_session_id("sphere-0")
        assert resolved == "detected-sphere-0"

        resolved = selector._resolve_session_id("sphere-1")
        assert resolved == "detected-sphere-1"

    def test_task_structure(self, temp_copick_project):
        """Test that tasks have the correct structure."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")

        selector = InputOutputSelector(
            pick_object_name="sphere-points",
            pick_user_id="test",
            pick_session_id="single",
            mesh_object_name="output-mesh",
            mesh_user_id="output-user",
            mesh_session_id="output-session",
            individual_meshes=True,
        )

        tasks = selector.get_conversion_tasks(run)
        task = tasks[0]

        # Verify all required keys exist
        required_keys = {
            "input_picks",
            "mesh_object_name",
            "mesh_user_id",
            "mesh_session_id",
            "individual_meshes",
            "session_id_template",
        }
        assert set(task.keys()) == required_keys

        # Verify types and values
        assert hasattr(task["input_picks"], "session_id")
        assert isinstance(task["mesh_object_name"], str)
        assert isinstance(task["mesh_user_id"], str)
        assert isinstance(task["mesh_session_id"], str)
        assert isinstance(task["individual_meshes"], bool)


class TestValidatePlaceholders:
    """Test placeholder validation functions."""

    def test_valid_one_to_one(self):
        """Test valid one-to-one configuration."""
        validate_placeholders("single", "output", False)
        # Should not raise

    def test_valid_one_to_many(self):
        """Test valid one-to-many configuration."""
        validate_placeholders("single", "output-{instance_id}", True)
        # Should not raise

    def test_valid_many_to_many(self):
        """Test valid many-to-many configuration."""
        validate_placeholders("pattern-.*", "output-{input_session_id}", False)
        validate_placeholders("pattern-.*", "output-{input_session_id}-{instance_id}", True)
        # Should not raise

    def test_invalid_individual_without_instance_id(self):
        """Test invalid individual meshes without instance_id."""
        with pytest.raises(ValueError, match="instance_id"):
            validate_placeholders("single", "output", True)

    def test_invalid_pattern_without_input_session_id(self):
        """Test invalid pattern without input_session_id."""
        with pytest.raises(ValueError, match="input_session_id"):
            validate_placeholders("pattern-.*", "output", False)

    def test_invalid_individual_pattern_missing_placeholders(self):
        """Test invalid individual + pattern missing placeholders."""
        with pytest.raises(ValueError, match="input_session_id"):
            validate_placeholders("pattern-.*", "output-{instance_id}", True)


class TestComplexScenarios:
    """Test complex selection scenarios."""

    def test_multiple_patterns_pattern_matching(self, temp_copick_project):
        """Test pattern matching works correctly across different pick sets."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")

        # Test pattern that should match multiple sphere sessions
        selector = InputOutputSelector(
            pick_object_name="sphere-points",
            pick_user_id="sim",
            pick_session_id="sphere-[0-2]",  # More specific pattern
            mesh_object_name="detected-sphere",
            mesh_user_id="detector",
            mesh_session_id="detected-{input_session_id}",
            individual_meshes=False,
        )

        tasks = selector.get_conversion_tasks(run)
        assert len(tasks) == 3

        # Check that each task has correct mapping
        input_sessions = {task["input_picks"].session_id for task in tasks}
        output_sessions = {task["mesh_session_id"] for task in tasks}

        expected_input = {"sphere-0", "sphere-1", "sphere-2"}
        expected_output = {"detected-sphere-0", "detected-sphere-1", "detected-sphere-2"}

        assert input_sessions == expected_input
        assert output_sessions == expected_output

    def test_combined_clustering_and_individual_meshes(self, temp_copick_project):
        """Test selection for clustering with individual meshes."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")

        selector = InputOutputSelector(
            pick_object_name="sphere-points",
            pick_user_id="sim",
            pick_session_id="all-spheres",
            mesh_object_name="cluster-sphere",
            mesh_user_id="clusterer",
            mesh_session_id="cluster-{instance_id}",
            individual_meshes=True,
        )

        tasks = selector.get_conversion_tasks(run)
        assert len(tasks) == 1

        task = tasks[0]
        assert task["mesh_session_id"] == "cluster-{instance_id}"
        assert "{instance_id}" in task["mesh_session_id"]  # Template should be preserved
        assert task["session_id_template"] == "cluster-{instance_id}"

    def test_object_name_inheritance(self, temp_copick_project):
        """Test that object names are properly inherited."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")

        # Test with None mesh_object_name (should default to pick_object_name)
        selector = InputOutputSelector(
            pick_object_name="sphere-points",
            pick_user_id="test",
            pick_session_id="single",
            mesh_object_name=None,
            mesh_user_id="test",
            mesh_session_id="test",
            individual_meshes=False,
        )

        tasks = selector.get_conversion_tasks(run)
        assert len(tasks) == 1
        assert tasks[0]["mesh_object_name"] == "sphere-points"

        # Test with explicit mesh_object_name
        selector.mesh_object_name = "explicit-mesh"
        tasks = selector.get_conversion_tasks(run)
        assert len(tasks) == 1
        assert tasks[0]["mesh_object_name"] == "explicit-mesh"

    def test_edge_case_invalid_regex(self):
        """Test edge case with invalid regex pattern."""
        # Invalid regex should be treated as literal string and not trigger input_session_id requirement
        selector = InputOutputSelector(
            pick_object_name="sphere-points",
            pick_user_id="test",
            pick_session_id="test[invalid",  # Invalid regex
            mesh_object_name="test",
            mesh_user_id="test",
            mesh_session_id="output",  # No placeholders needed since it's not a valid regex
            individual_meshes=False,
        )

        # Should not raise because invalid regex is treated as literal string
        assert not selector._is_regex_pattern("test[invalid")

    def test_complex_regex_patterns(self, temp_copick_project):
        """Test complex regex patterns."""
        root, temp_path = temp_copick_project
        run = root.get_run("test_run")

        # Test complex pattern that matches our test data
        selector = InputOutputSelector(
            pick_object_name="sphere-points",
            pick_user_id="sim",
            pick_session_id="sphere-[0-9]+",  # Complex pattern
            mesh_object_name="pattern-sphere",
            mesh_user_id="pattern-user",
            mesh_session_id="pattern-{input_session_id}",
            individual_meshes=False,
        )

        tasks = selector.get_conversion_tasks(run)
        assert len(tasks) == 3  # Should still match sphere-0, sphere-1, sphere-2

        for task in tasks:
            assert task["mesh_session_id"].startswith("pattern-sphere-")
            assert task["input_picks"].session_id in ["sphere-0", "sphere-1", "sphere-2"]
