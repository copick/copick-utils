import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import (
    add_input_option,
    add_output_option,
    add_segmentation_processing_options,
    add_workers_option,
)
from copick_utils.util.config_models import create_simple_config


@click.command(
    context_settings={"show_default": True},
    short_help="Convert segmentation to picks.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input segmentations.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_input_option("segmentation")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_segmentation_processing_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output picks.")
@add_output_option("picks", default_tool="seg2picks")
@add_debug_option
def seg2picks(
    config,
    run_names,
    input_uri,
    segmentation_idx,
    maxima_filter_size,
    min_particle_size,
    max_particle_size,
    workers,
    output_uri,
    debug,
):
    """
    Convert segmentation volumes to picks by extracting centroids.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing
        Picks: object_name:user_id/session_id

    \b
    Examples:
        # Convert single segmentation to picks
        copick convert seg2picks -i "membrane:user1/manual-001@10.0" -o "membrane:seg2picks/centroid-001"

        # Convert all manual segmentations using pattern matching
        copick convert seg2picks -i "membrane:user1/manual-.*@10.0" -o "membrane:seg2picks/centroid-{input_session_id}"
    """
    from copick_utils.converters.picks_from_segmentation import picks_from_segmentation_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Create config directly from URIs with smart defaults
    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="segmentation",
            output_uri=output_uri,
            output_type="picks",
            command_name="seg2picks",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Extract parameters for logging
    input_params = parse_copick_uri(input_uri, "segmentation")
    output_params = parse_copick_uri(output_uri, "picks")

    logger.info(f"Converting segmentation to picks for '{input_params['name']}'")
    logger.info(
        f"Source segmentation pattern: {input_params['name']} ({input_params['user_id']}/{input_params['session_id']})",
    )
    logger.info(
        f"Target picks template: {output_params['object_name']} ({output_params['user_id']}/{output_params['session_id']})",
    )
    logger.info(f"Label {segmentation_idx}, particle size: {min_particle_size}-{max_particle_size}")

    # Parallel discovery and processing - no sequential bottleneck!
    results = picks_from_segmentation_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        segmentation_idx=segmentation_idx,
        maxima_filter_size=maxima_filter_size,
        min_particle_size=min_particle_size,
        max_particle_size=max_particle_size,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_points = sum(result.get("points_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("processed", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total conversion tasks completed: {total_processed}")
    logger.info(f"Total points created: {total_points}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
