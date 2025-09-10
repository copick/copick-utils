import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from copick_utils.cli.input_output_selection import ConversionSelector, validate_conversion_placeholders
from copick_utils.cli.util import (
    add_picks_output_options,
    add_segmentation_input_options,
    add_segmentation_processing_options,
    add_workers_option,
)


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
@add_segmentation_input_options
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_segmentation_processing_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output picks.")
@add_picks_output_options(default_tool="seg2picks")
@add_debug_option
def seg2picks(
    config,
    run_names,
    seg_name,
    seg_user_id,
    seg_session_id,
    voxel_spacing,
    multilabel,
    segmentation_idx,
    maxima_filter_size,
    min_particle_size,
    max_particle_size,
    workers,
    pick_object_name,
    pick_user_id,
    pick_session_id,
    debug,
):
    """
    Convert segmentation volumes to picks by extracting centroids.

    Supports flexible input/output selection modes:
    - One-to-one: exact session ID → exact session ID
    - Many-to-many: regex pattern → template with {input_session_id}

    Examples:
        # Convert single segmentation to picks
        seg2picks --seg-session-id "manual-001" --pick-session-id "centroid-001"

        # Convert all manual segmentations using pattern matching
        seg2picks --seg-session-id "manual-.*" --pick-session-id "centroid-{input_session_id}"
    """
    from copick_utils.converters.picks_from_segmentation import picks_from_segmentation_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Validate placeholder requirements
    try:
        validate_conversion_placeholders(seg_session_id, pick_session_id, individual_outputs=False)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Create conversion selector
    selector = ConversionSelector(
        input_type="segmentation",
        output_type="picks",
        input_object_name=pick_object_name,  # For picks, we use pick_object_name as both input and output object
        input_user_id=seg_user_id,
        input_session_id=seg_session_id,
        output_object_name=pick_object_name,
        output_user_id=pick_user_id,
        output_session_id=pick_session_id,
        segmentation_name=seg_name,
        voxel_spacing=voxel_spacing,
    )

    logger.info(f"Converting segmentation to picks for '{seg_name}'")
    logger.info(f"Selection mode: {selector.get_mode_description()}")
    logger.info(f"Source segmentation pattern: {seg_name} ({seg_user_id}/{seg_session_id})")
    logger.info(f"Target picks template: {pick_object_name} ({pick_user_id}/{pick_session_id})")
    logger.info(f"Label {segmentation_idx}, particle size: {min_particle_size}-{max_particle_size}")

    # Collect all conversion tasks across runs
    all_tasks = []
    runs_to_process = root.runs if run_names_list is None else [root.get_run(name) for name in run_names_list]

    for run in runs_to_process:
        tasks = selector.get_conversion_tasks(run)
        all_tasks.extend(tasks)

    if not all_tasks:
        logger.warning("No matching segmentations found for conversion")
        return

    logger.info(f"Found {len(all_tasks)} conversion tasks across {len(runs_to_process)} runs")

    results = picks_from_segmentation_batch(
        root=root,
        conversion_tasks=all_tasks,
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
