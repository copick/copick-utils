"""CLI command for rescaling segmentations to a different voxel spacing."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import add_input_option, add_output_option, add_workers_option
from copick_utils.util.config_models import create_simple_config


@click.command(
    context_settings={"show_default": True},
    short_help="Rescale segmentations to a different voxel spacing.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input segmentation.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_input_option("segmentation")
@optgroup.group("\nTool Options", help="Options related to rescaling.")
@optgroup.option(
    "--target-voxel-spacing",
    "-tvs",
    type=float,
    default=None,
    help="Target voxel spacing in angstroms. If omitted, derived from output URI @voxel_spacing.",
)
@optgroup.option(
    "--tomo-type",
    "-tt",
    type=str,
    default=None,
    help="Tomogram type to use as shape reference at target spacing (e.g. 'wbp'). "
    "When provided, output shape matches the tomogram exactly.",
)
@optgroup.option(
    "--order",
    type=click.Choice(["0", "1"]),
    default="0",
    help="Interpolation order: 0=nearest-neighbor (labels), 1=linear (float data).",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="rescale")
@add_debug_option
def rescale(
    config,
    run_names,
    input_uri,
    target_voxel_spacing,
    tomo_type,
    order,
    workers,
    output_uri,
    debug,
):
    """
    Rescale segmentations to a different voxel spacing.

    Resamples segmentation data using nearest-neighbor interpolation (default) to
    preserve label integrity. Supports both upscaling (finer spacing) and downscaling
    (coarser spacing). When --tomo-type is provided, the output shape is matched to
    an existing tomogram at the target spacing for exact alignment.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing

    \b
    Examples:
        # Upscale: 10 angstrom -> 5 angstrom
        copick process rescale -i "membrane:user1/auto@10.0" -o "membrane:rescale/0@5.0"

        # Downscale: 5 angstrom -> 20 angstrom
        copick process rescale -i "membrane:user1/manual@5.0" -o "membrane:rescale/0@20.0"

        # Match tomogram shape exactly
        copick process rescale -i "membrane:user1/auto@10.0" -o "membrane:rescale/0@5.0" --tomo-type wbp

        # Explicit target spacing (when output URI uses smart defaults)
        copick process rescale -i "membrane:user1/auto@10.0" -o "membrane_rescaled" --target-voxel-spacing 5.0

        # Rescale specific runs
        copick process rescale -i "organelle:pred/auto@10.0" -o "organelle:rescale/0@7.5" -r run1 -r run2

        # Linear interpolation (for non-label float data)
        copick process rescale -i "density:user1/auto@10.0" -o "density:rescale/0@5.0" --order 1
    """
    from copick_utils.process.rescale import rescale_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Resolve target voxel spacing
    output_params = parse_copick_uri(output_uri, "segmentation")
    output_vs = output_params.get("voxel_spacing")
    output_vs_float = float(output_vs) if output_vs is not None and output_vs != "*" else None

    if (
        target_voxel_spacing is not None
        and output_vs_float is not None
        and abs(target_voxel_spacing - output_vs_float) > 1e-6
    ):
        raise click.BadParameter(
            f"--target-voxel-spacing ({target_voxel_spacing}) conflicts with "
            f"output URI voxel spacing ({output_vs_float}). Use one or the other.",
        )

    tvs = target_voxel_spacing or output_vs_float
    if tvs is None:
        raise click.BadParameter(
            "Must specify target voxel spacing via --target-voxel-spacing or output URI @voxel_spacing "
            '(e.g. -o "name:user/session@5.0").',
        )

    # Create config from URIs
    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="segmentation",
            output_uri=output_uri,
            output_type="segmentation",
            command_name="rescale",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Log parameters
    input_params = parse_copick_uri(input_uri, "segmentation")
    logger.info(f"Rescaling segmentation '{input_params['name']}'")
    logger.info(
        f"Source: {input_params['user_id']}/{input_params['session_id']} @ {input_params.get('voxel_spacing', '*')} Å",
    )
    logger.info(f"Target voxel spacing: {tvs} Å")
    if tomo_type:
        logger.info(f"Shape reference: tomogram '{tomo_type}' @ {tvs} Å")
    logger.info(f"Interpolation order: {order} ({'nearest-neighbor' if order == '0' else 'linear'})")

    # Parallel discovery and processing
    results = rescale_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        target_voxel_spacing=tvs,
        tomo_type=tomo_type,
        order=int(order),
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_rescaled = sum(result.get("rescaled", 0) for result in results.values() if result)
    total_labels_ok = sum(result.get("labels_preserved", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total segmentations rescaled: {total_rescaled}")
    if total_rescaled > 0:
        logger.info(f"Labels preserved: {total_labels_ok}/{total_rescaled}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
