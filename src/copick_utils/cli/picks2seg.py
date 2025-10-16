import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import (
    add_input_option,
    add_output_option,
    add_picks_painting_options,
    add_workers_option,
)
from copick_utils.util.config_models import create_simple_config


@click.command(
    context_settings={"show_default": True},
    short_help="Convert picks to segmentation.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input picks.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_input_option("picks")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_picks_painting_options
@optgroup.option(
    "--tomo-type",
    "-tt",
    default="wbp",
    help="Type of tomogram to use as reference.",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="picks2seg")
@add_debug_option
def picks2seg(
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
    Convert picks to segmentation volumes by painting spheres.

    \b
    URI Format:
        Picks: object_name:user_id/session_id
        Segmentations: name:user_id/session_id@voxel_spacing

    \b
    Examples:
        # Convert single pick set to segmentation
        copick convert picks2seg -i "ribosome:user1/manual-001" -o "ribosome:picks2seg/painted-001@10.0"

        # Convert all manual picks using pattern matching
        copick convert picks2seg -i "ribosome:user1/manual-.*" -o "ribosome:picks2seg/painted-{input_session_id}@10.0"
    """
    from copick_utils.converters.segmentation_from_picks import segmentation_from_picks_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Create config directly from URIs with smart defaults
    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="picks",
            output_uri=output_uri,
            output_type="segmentation",
            command_name="picks2seg",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Extract parameters for logging
    input_params = parse_copick_uri(input_uri, "picks")
    output_params = parse_copick_uri(output_uri, "segmentation")

    voxel_spacing = output_params["voxel_spacing"]
    if isinstance(voxel_spacing, str):
        voxel_spacing = float(voxel_spacing)

    logger.info(f"Converting picks to segmentation for object '{input_params['object_name']}'")
    logger.info(f"Source picks pattern: {input_params['user_id']}/{input_params['session_id']}")
    logger.info(
        f"Target segmentation template: {output_params['name']} ({output_params['user_id']}/{output_params['session_id']})",
    )
    logger.info(f"Sphere radius: {radius}, voxel spacing: {voxel_spacing}")

    # Parallel discovery and processing - no sequential bottleneck!
    results = segmentation_from_picks_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        radius=radius,
        tomo_type=tomo_type,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_points = sum(result.get("points_converted", 0) for result in results.values() if result)
    total_voxels = sum(result.get("voxels_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("processed", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total conversion tasks completed: {total_processed}")
    logger.info(f"Total points converted: {total_points}")
    logger.info(f"Total voxels created: {total_voxels}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
