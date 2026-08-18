"""Installed-package smoke tests for implementation modules and plugins."""

import importlib
import importlib.metadata
import pkgutil

import copick_utils
from click.testing import CliRunner


def test_all_implementation_modules_import():
    modules = [module.name for module in pkgutil.walk_packages(copick_utils.__path__, "copick_utils.")]

    assert modules
    for module in modules:
        importlib.import_module(module)


def test_all_copick_command_entry_points_load_and_render_help():
    entry_points = [
        entry_point
        for entry_point in importlib.metadata.entry_points()
        if entry_point.group.startswith("copick.") and entry_point.dist.name == "copick-utils"
    ]

    assert len(entry_points) == 32
    runner = CliRunner()
    for entry_point in entry_points:
        command = entry_point.load()
        result = runner.invoke(command, ["--help"])
        assert result.exit_code == 0, f"{entry_point.group}:{entry_point.name}\n{result.output}"
