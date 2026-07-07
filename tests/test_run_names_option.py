"""Tests that copick-utils CLI commands use the shared run-selection option.

All commands were de-duplicated onto ``add_run_names_option`` from
``copick.cli.util``; ``separate_components`` and ``fit_spline`` additionally
gained the ``-r`` short flag they previously lacked.
"""

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


# A representative slice across the conversion / processing / logical groups,
# including the two commands that previously lacked the -r short flag.
COMMANDS = [
    ("copick_utils.cli.separate_components", "separate_components"),
    ("copick_utils.cli.fit_spline", "fit_spline"),
    ("copick_utils.cli.picks2seg", "picks2seg"),
    ("copick_utils.cli.seg2picks", "seg2picks"),
    ("copick_utils.cli.segop", "segop"),
    ("copick_utils.cli.rescale", "rescale"),
    ("copick_utils.cli.combine_labels", "combine"),
]


@pytest.mark.parametrize(("dotted", "attr"), COMMANDS)
def test_command_exposes_standard_run_option(runner, dotted, attr):
    import importlib

    cmd = getattr(importlib.import_module(dotted), attr)
    out = runner.invoke(cmd, ["--help"]).output
    assert "--run-names" in out
    assert "-r" in out
