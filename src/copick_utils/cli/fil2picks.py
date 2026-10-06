"""CLI command for sampling picks along filaments."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option, add_run_names_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import (
    add_filament_sampling_options,
    add_input_option,
    add_output_option,
    add_workers_option,
)
from copick_utils.util.config_models import create_simple_config


@click.command(
    context_settings={"show_default": True},
    short_help="Sample picks along filaments.",
    no_args_is_help=True,
)
@add_config_option
@add_run_names_option
@optgroup.group("\nInput Options", help="Options related to the input filaments.")
@add_input_option("filaments")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_filament_sampling_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to the output picks.")
@add_output_option("picks", default_tool="fil2picks")
@add_debug_option
def fil2picks(
    config,
    run_names,
    input_uri,
    spacing,
    anchor,
    roll,
    seed,
    length_unit,
    workers,
    output_uri,
    debug,
):
    """
    Sample picks along filaments.

    Places picks along each filament at a fixed `--spacing`, measured along the filament's
    curve. Each pick's orientation has its +Z axis along the filament, in the filament's point
    order, and its rotation about that axis follows `--roll`. Every pick carries its filament's
    ID as its instance ID and the filament's score, and the picks are written filament by
    filament, in order along each one. Exported to RELION with
    `copick export picks --filament-columns on`, they carry RELION's filament conventions.

    A filament's stored curve is used when it still matches the filament: the B-spline that
    `copick convert seg2fil` fitted, or control points from an editor. A filament without one is
    sampled along a smooth curve derived from its points. Sampling is cheap and does not change
    the filaments, so different spacings can be tried without tracing again.

    URI Format:

        \b
        Filaments: object_name:user_id/session_id
        Picks: object_name:user_id/session_id

    Examples:

        \b
        # Picks every 82 Å (one tubulin dimer), in the filaments' own session
        copick convert fil2picks -i "microtubule:trace/1" -o "microtubule:trace/1" --spacing 82

        \b
        # Every 4 nm with a random rotation about the axis
        copick convert fil2picks -i "microtubule:trace/1" -o "microtubule:picks/4nm" --spacing 40 --roll random --seed 1

    See Also:

        \b
        copick convert seg2fil: trace filaments in a segmentation
        copick convert fil2seg: paint tubes around filaments into an instance segmentation
        copick export picks: export picks to RELION with --filament-columns
    """
    from copick_utils.converters.picks_from_filaments import picks_from_filaments_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="filaments",
            output_uri=output_uri,
            output_type="picks",
            command_name="fil2picks",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    input_params = parse_copick_uri(input_uri, "filaments")
    logger.info(f"Sampling filaments '{input_params['object_name']}' every {spacing} ({length_unit})")

    results = picks_from_filaments_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        spacing=spacing,
        anchor=anchor,
        roll=roll,
        seed=seed,
        length_unit=length_unit,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    filaments = sum(result.get("filaments_sampled", 0) for result in results.values() if result)
    points = sum(result.get("points_created", 0) for result in results.values() if result)

    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total: {points} picks from {filaments} filaments")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
