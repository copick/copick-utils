"""Regression guard for the lazy-batch converter signature contract.

`lazy_conversion_worker` (converters/lazy_converter.py) calls each converter as
`converter_func(**task_params)` where task_params ALWAYS contains `object_name`, `user_id`,
`session_id`, `voxel_spacing`, the input object under a type-named key (`picks`/`mesh`/
`segmentation`), plus (for reference ops) `reference_mesh`/`reference_segmentation`/
`reference_tomogram_info`, plus the CLI's converter kwargs. A converter is only safe if it:
  - has `**kwargs` (to absorb voxel_spacing / reference keys / future additions),
  - names the output params `object_name`/`session_id`/`user_id` (not `output_*`/`pick_*`),
  - names the input object under the worker's type key, and
  - (reference ops) names `reference_mesh`/`reference_segmentation`.

This test pins that contract against the function each `*_lazy_batch` actually wraps, so the whole
class of "signature drift" bugs (seg2mesh, mesh2picks, picks2{mesh,surface,ellipsoid,plane,sphere},
picksin/picksout) cannot silently come back.
"""

import inspect

import pytest

from copick_utils.converters.caps_from_mesh import caps_from_mesh_lazy_batch
from copick_utils.converters.ellipsoid_from_picks import ellipsoid_from_picks_lazy_batch
from copick_utils.converters.mesh_from_picks import mesh_from_picks_lazy_batch
from copick_utils.converters.mesh_from_segmentation import mesh_from_segmentation_lazy_batch
from copick_utils.converters.picks_from_mesh import picks_from_mesh_lazy_batch
from copick_utils.converters.picks_from_segmentation import picks_from_segmentation_lazy_batch
from copick_utils.converters.plane_from_picks import plane_from_picks_lazy_batch
from copick_utils.converters.segmentation_from_mesh import segmentation_from_mesh_lazy_batch
from copick_utils.converters.segmentation_from_picks import segmentation_from_picks_lazy_batch
from copick_utils.converters.sphere_from_picks import sphere_from_picks_lazy_batch
from copick_utils.converters.surface_from_picks import surface_from_picks_lazy_batch
from copick_utils.logical.distance_operations import (
    limit_mesh_by_distance_lazy_batch,
    limit_picks_by_distance_lazy_batch,
    limit_segmentation_by_distance_lazy_batch,
)
from copick_utils.logical.point_operations import (
    picks_exclusion_by_mesh_lazy_batch,
    picks_inclusion_by_mesh_lazy_batch,
)

# (id, lazy_batch, input-key the worker uses, is-reference-op)
CASES = [
    ("seg2mesh", mesh_from_segmentation_lazy_batch, "segmentation", False),
    ("mesh2seg", segmentation_from_mesh_lazy_batch, "mesh", False),
    ("mesh2picks", picks_from_mesh_lazy_batch, "mesh", False),
    ("picks2seg", segmentation_from_picks_lazy_batch, "picks", False),
    ("seg2picks", picks_from_segmentation_lazy_batch, "segmentation", False),
    ("picks2mesh", mesh_from_picks_lazy_batch, "picks", False),
    ("picks2surface", surface_from_picks_lazy_batch, "picks", False),
    ("picks2ellipsoid", ellipsoid_from_picks_lazy_batch, "picks", False),
    ("picks2plane", plane_from_picks_lazy_batch, "picks", False),
    ("picks2sphere", sphere_from_picks_lazy_batch, "picks", False),
    ("mesh2caps", caps_from_mesh_lazy_batch, "mesh", False),
    ("picksin", picks_inclusion_by_mesh_lazy_batch, "picks", True),
    ("picksout", picks_exclusion_by_mesh_lazy_batch, "picks", True),
    ("clippicks", limit_picks_by_distance_lazy_batch, "picks", True),
    ("clipmesh", limit_mesh_by_distance_lazy_batch, "mesh", True),
    ("clipseg", limit_segmentation_by_distance_lazy_batch, "segmentation", True),
]


def _wrapped(lazy_batch):
    """Extract the converter_func a `*_lazy_batch` closure actually wraps (so we test the real one)."""
    for cell in lazy_batch.__closure__ or ():
        v = cell.cell_contents
        if callable(v) and getattr(v, "__name__", "") not in ("", "lazy_batch_converter"):
            return v
    raise AssertionError("could not locate wrapped converter_func")


@pytest.mark.parametrize("name,lazy_batch,input_key,is_ref", CASES, ids=[c[0] for c in CASES])
def test_converter_matches_worker_contract(name, lazy_batch, input_key, is_ref):
    fn = _wrapped(lazy_batch)
    sig = inspect.signature(fn)
    names = set(sig.parameters)

    # The output-naming params and the input object must be REAL parameters (a converter that only
    # absorbs them via **kwargs can't use them). Catches the output_*/pick_* renames (mesh2picks)
    # and the points-vs-picks input mismatch (picks2{mesh,surface,...}).
    for req in (input_key, "object_name", "session_id", "user_id"):
        assert req in names, f"{name}: {fn.__name__} must name parameter '{req}' (worker passes it by that key)"
    if is_ref:
        assert {
            "reference_mesh",
            "reference_segmentation",
        } <= names, f"{name}: reference op {fn.__name__} must name reference_mesh/reference_segmentation"

    # The converter must ACCEPT the exact kwargs the worker passes (either named or via **kwargs).
    # Catches the always-present `voxel_spacing` breaking converters that neither name it nor have
    # **kwargs (seg2mesh, seg2picks, the picks->shape fitters). bind_partial tolerates CLI-only
    # required args (filled from the command) but still rejects unexpected kwargs.
    kw = {
        input_key: object(),
        "run": object(),
        "object_name": "o",
        "user_id": "u",
        "session_id": "s",
        "voxel_spacing": 10.0,
    }
    if is_ref:
        kw.update(reference_mesh=None, reference_segmentation=None, reference_tomogram_info=None)
    sig.bind_partial(**kw)
