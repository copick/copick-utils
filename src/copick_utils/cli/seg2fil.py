"""CLI command for tracing filaments in segmentations."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option, add_run_names_option
from copick.util.log import get_logger
from copick.util.uri import expand_output_uri, parse_copick_uri

from copick_utils.cli.util import (
    add_filament_tracing_options,
    add_input_option,
    add_output_option,
    add_workers_option,
)
from copick_utils.util.config_models import create_simple_config


def length_in_angstrom(value, unit, input_uri):
    """A length option in angstroms, converted from voxels with the input URI's voxel spacing if needed."""
    if value is None or unit == "angstrom":
        return value
    vs_raw = parse_copick_uri(input_uri, "segmentation").get("voxel_spacing")
    if vs_raw is None or vs_raw == "*":
        raise click.BadParameter("Lengths in voxels require a voxel spacing in the input URI (e.g., @10.0).")
    return value * float(vs_raw)


def volume_in_angstrom3(value, unit, input_uri):
    """A volume option in cubic angstroms, converted from cubic voxels with the input URI's voxel spacing if needed."""
    if value is None or unit == "angstrom":
        return value
    vs_raw = parse_copick_uri(input_uri, "segmentation").get("voxel_spacing")
    if vs_raw is None or vs_raw == "*":
        raise click.BadParameter("Volumes in voxels require a voxel spacing in the input URI (e.g., @10.0).")
    return value * float(vs_raw) ** 3


@click.command(
    context_settings={"show_default": True},
    short_help="Trace filaments in segmentations.",
    no_args_is_help=True,
)
@add_config_option
@add_run_names_option
@optgroup.group("\nInput Options", help="Options related to the input segmentation.")
@add_input_option("segmentation")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_filament_tracing_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to the output filaments.")
@add_output_option("filaments", default_tool="seg2fil")
@add_output_option(
    "segmentation",
    flag="--instances",
    short_flag="-oi",
    param_name="instances_uri",
    required=False,
    description="Also write the instance segmentation of the traced filaments (each voxel holds the ID of its "
    "filament, the same IDs as the filaments). Smart defaults as for -o; ?instance=true is implied.",
)
@add_debug_option
def seg2fil(
    config,
    run_names,
    input_uri,
    min_length,
    min_aspect,
    min_radius,
    fill_lumen,
    min_volume,
    prune_length,
    junction_merge,
    max_bend,
    smoothing,
    extend_ends,
    label,
    length_unit,
    volume_unit,
    workers,
    output_uri,
    instances_uri,
    debug,
):
    """
    Trace filaments in segmentations.

    Traces the centreline of every filament (e.g. microtubules or actin) in a segmentation and
    stores it as a copick Filaments entry: the filament's fitted B-spline (an exact `bspline`
    curve that editors can reopen) and the centreline points copick regenerates from it.
    Filaments are numbered 1, 2, ... by length, longest first.

    Each connected component is skeletonized and split into branches between ends and junctions.
    Holes up to `--fill-lumen` are filled first, so a tube labelled by its wall alone traces as one
    filament. Side branches shorter than `--prune-length` are pruned, and at each junction the branches
    that continue most nearly straight (within `--max-bend`) are joined, so two crossing
    filaments stay two filaments. Free ends are extended to the edge of the segmentation, and
    filaments shorter than `--min-length`, shorter than `--min-aspect` label diameters (blobs),
    or thinner than `--min-radius` (slivers of segmentation noise) are rejected. Options left
    unset are derived from the segmentation's own thickness or the object's radius.

    The input can be a binary, multilabel (`--label`), panoptic or instance segmentation. An
    instance segmentation, for example one edited by hand or written by `--instances`, is traced
    one instance at a time and keeps its IDs. Sample picks from the filaments with
    `copick convert fil2picks`.

    URI Format:

        \b
        Segmentations: name:user_id/session_id@voxel_spacing
        Instance or panoptic input: append ?instance=true or ?panoptic=true
        Filaments: object_name:user_id/session_id

    Examples:

        \b
        # Trace microtubules and keep the instance segmentation of the traced filaments
        copick convert seg2fil -i "microtubule:easymode/job006@10.0" \\
            -o "microtubule:trace/1" --instances "microtubule:trace/1@10.0?instance=true"

        \b
        # Reject filaments shorter than 100 nm
        copick convert seg2fil -i "microtubule:easymode/job006@10.0" -o "microtubule:trace/1" --min-length 1000

        \b
        # Re-trace an edited instance segmentation, keeping its IDs
        copick convert seg2fil -i "microtubule:curated/1@10.0?instance=true" -o "microtubule:trace/curated"

    See Also:

        \b
        copick convert fil2picks: sample picks along the traced filaments
        copick convert fil2seg: paint tubes around filaments into an instance segmentation
        copick process fit-spline: segmentation to picks in one step (older, voxel units)
        copick process seg-stats: component and skeleton statistics to choose thresholds
    """
    from copick_utils.converters.filaments_from_segmentation import filaments_from_segmentation_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    min_length = length_in_angstrom(min_length, length_unit, input_uri)
    min_radius = length_in_angstrom(min_radius, length_unit, input_uri)
    fill_lumen = length_in_angstrom(fill_lumen, length_unit, input_uri)
    prune_length = length_in_angstrom(prune_length, length_unit, input_uri)
    junction_merge = length_in_angstrom(junction_merge, length_unit, input_uri)
    smoothing = length_in_angstrom(smoothing, length_unit, input_uri)
    min_volume = volume_in_angstrom3(min_volume, volume_unit, input_uri)

    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="segmentation",
            output_uri=output_uri,
            output_type="filaments",
            command_name="seg2fil",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    instance_params = {}
    if instances_uri:
        expanded = expand_output_uri(
            output_uri=instances_uri,
            input_uri=input_uri,
            input_type="segmentation",
            output_type="segmentation",
            command_name="seg2fil",
        )
        parsed = parse_copick_uri(expanded, "segmentation")
        if parsed.get("multilabel") or parsed.get("panoptic"):
            raise click.BadParameter("--instances writes an instance segmentation (?instance=true).")
        instance_params = {
            "instances_object_name": parsed["name"],
            "instances_user_id": parsed["user_id"],
            "instances_session_id": parsed["session_id"],
        }

    input_params = parse_copick_uri(input_uri, "segmentation")
    logger.info(f"Tracing filaments in segmentation '{input_params['name']}'")
    logger.info(f"Source segmentation pattern: {input_params['user_id']}/{input_params['session_id']}")

    results = filaments_from_segmentation_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        min_volume=min_volume,
        min_length=min_length,
        min_aspect=min_aspect,
        min_radius=min_radius,
        fill_lumen=fill_lumen,
        prune_length=prune_length,
        junction_merge=junction_merge,
        max_bend=max_bend,
        smoothing=smoothing,
        extend_ends=extend_ends,
        label=label,
        **instance_params,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    filaments = sum(result.get("filaments_written", 0) for result in results.values() if result)
    length = sum(result.get("total_length", 0.0) for result in results.values() if result)
    short = sum(result.get("rejected_length", 0) for result in results.values() if result)
    blobs = sum(result.get("rejected_aspect", 0) for result in results.values() if result)
    thin = sum(result.get("rejected_radius", 0) for result in results.values() if result)

    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(
        f"Total filaments: {filaments} ({length / 10:.0f} nm); rejected {short} short, {blobs} blob-like, {thin} thin",
    )

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
