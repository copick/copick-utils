"""Tests for the Part 2 CLI flag normalization in copick-utils.

- picks2mesh / thickness_filter: `-t` is reserved for tomogram refs; their old
  `-t` short flag moved to `-mt` with a hidden `-t` deprecated alias.
- fit_spline: voxel spacing now comes from the input URI's `@voxel_spacing`; the
  `--voxel-spacing/-vs` flag is a hidden deprecated override.
"""

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def _hidden_alias(cmd, flag):
    param = next((p for p in cmd.params if flag in getattr(p, "opts", [])), None)
    return param, (param.hidden if param else None)


def test_picks2mesh_reserves_t_for_tomogram(runner):
    from copick_utils.cli.picks2mesh import picks2mesh

    out = runner.invoke(picks2mesh, ["--help"]).output
    assert "--mesh-type" in out and "-mt" in out
    assert "-t " not in out  # -t no longer shown (reserved for --tomogram)

    param, hidden = _hidden_alias(picks2mesh, "-t")
    assert param is not None and hidden is True
    assert param.name == "legacy_mesh_type"


def test_thickness_filter_reserves_t_for_tomogram(runner):
    from copick_utils.cli.thickness_filter import thickness_filter

    out = runner.invoke(thickness_filter, ["--help"]).output
    assert "--min-thickness" in out and "-mt" in out
    assert "-t " not in out

    param, hidden = _hidden_alias(thickness_filter, "-t")
    assert param is not None and hidden is True
    assert param.name == "legacy_min_thickness"


def test_fit_spline_derives_vs_from_uri(runner):
    from copick_utils.cli.fit_spline import fit_spline

    out = runner.invoke(fit_spline, ["--help"]).output
    assert "-i" in out and "--input" in out
    # voxel spacing is derived from the input URI; the flag is hidden + deprecated
    assert "--voxel-spacing" not in out
    assert "-vs" not in out

    param, hidden = _hidden_alias(fit_spline, "--voxel-spacing")
    assert param is not None and hidden is True
