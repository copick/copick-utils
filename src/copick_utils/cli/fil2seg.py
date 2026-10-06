"""CLI command for painting tubes around filaments into an instance segmentation."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option, add_run_names_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import add_input_option, add_output_option, add_workers_option
from copick_utils.util.config_models import create_simple_config


@click.command(
    context_settings={"show_default": True},
    short_help="Paint tubes around filaments into an instance segmentation.",
    no_args_is_help=True,
)
@add_config_option
@add_run_names_option
@optgroup.group("\nInput Options", help="Options related to the input filaments.")
@add_input_option("filaments")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--radius",
    type=float,
    default=None,
    help="Tube radius in angstroms for filaments without their own radius. Unset: the object's radius.",
)
@optgroup.option(
    "--tomo-type",
    "-tt",
    default="wbp",
    help="Type of tomogram whose shape the segmentation takes.",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to the output segmentation.")
@add_output_option("segmentation", default_tool="fil2seg")
@add_debug_option
def fil2seg(
    config,
    run_names,
    input_uri,
    radius,
    tomo_type,
    workers,
    output_uri,
    debug,
):
    """
    Paint tubes around filaments into an instance segmentation.

    Each filament becomes a tube around its centreline, labelled with the filament's ID, so the
    result is an instance segmentation with the same IDs as the filaments and their picks.
    A filament's tube radius is its own radius (`copick convert seg2fil` records the radius of
    the segmentation it traced), otherwise `--radius`, otherwise the object's radius. Where
    tubes overlap, each voxel goes to the nearest centreline. The segmentation has the shape of
    the run's `--tomo-type` tomogram at the voxel spacing given in the output URI.

    Useful for filaments drawn or edited by hand, or imported, that have no segmentation yet:
    for example as per-filament masks, or as input to `copick convert seg2fil`.

    URI Format:

        \b
        Filaments: object_name:user_id/session_id
        Segmentations: name:user_id/session_id@voxel_spacing?instance=true

    Examples:

        \b
        # Tubes of each filament's own radius, at 10 Å
        copick convert fil2seg -i "microtubule:manual/1" -o "microtubule:tubes/1@10.0?instance=true"

        \b
        # 125 Å tubes against a denoised tomogram
        copick convert fil2seg -i "microtubule:manual/1" -o "microtubule:tubes/1@10.0?instance=true" \\
            --radius 125 --tomo-type denoised

    See Also:

        \b
        copick convert seg2fil: trace filaments in a segmentation
        copick convert fil2picks: sample picks along filaments
        copick convert picks2seg: paint spheres around picks
    """
    from copick_utils.converters.segmentation_from_filaments import segmentation_from_filaments_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    output_params = parse_copick_uri(output_uri, "segmentation")
    if output_params.get("multilabel") or output_params.get("panoptic"):
        raise click.BadParameter("fil2seg writes an instance segmentation (?instance=true).")
    if not output_params.get("instance"):
        output_uri = output_uri + ("&" if "?" in output_uri else "?") + "instance=true"

    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="filaments",
            output_uri=output_uri,
            output_type="segmentation",
            command_name="fil2seg",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    input_params = parse_copick_uri(input_uri, "filaments")
    logger.info(f"Painting filaments '{input_params['object_name']}' into an instance segmentation")

    results = segmentation_from_filaments_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        radius=radius,
        tomo_type=tomo_type,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    filaments = sum(result.get("filaments_painted", 0) for result in results.values() if result)

    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total filaments painted: {filaments}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
