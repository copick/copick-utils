import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from copick_utils.cli.input_output_selection import validate_conversion_placeholders
from copick_utils.cli.util import (
    add_pick_input_options,
    add_picks_painting_options,
    add_segmentation_output_options,
    add_workers_option,
)
from copick_utils.converters.config_models import SelectorConfig, TaskConfig
from copick_utils.converters.segmentation_from_picks import segmentation_from_picks_lazy_batch


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
@add_pick_input_options
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_picks_painting_options
@optgroup.option(
    "--tomo-type",
    "-tt",
    default="wbp",
    help="Type of tomogram to use as reference.",
)
@optgroup.option(
    "--voxel-spacing",
    "-vs",
    type=float,
    required=True,
    help="Voxel spacing for the segmentation.",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_segmentation_output_options(default_tool="picks2seg", include_multilabel=False, include_tomo_type=False)
@add_debug_option
def picks2seg(
    config,
    run_names,
    pick_object_name,
    pick_user_id,
    pick_session_id,
    radius,
    tomo_type,
    voxel_spacing,
    workers,
    seg_name_output,
    seg_user_id_output,
    seg_session_id_output,
    debug,
):
    """
    Convert picks to segmentation volumes by painting spheres.

    \b
    Supports flexible input/output selection modes:
    - One-to-one: exact session ID → exact session ID
    - Many-to-many: regex pattern → template with {input_session_id}

    \b
    Examples:
        # Convert single pick set to segmentation
        copick convert picks2seg --pick-session-id "manual-001" --seg-session-id "painted-001"
        \b
        # Convert all manual picks using pattern matching
        copick convert picks2seg --pick-session-id "manual-.*" --seg-session-id "painted-{input_session_id}"
    """

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Validate placeholder requirements
    try:
        validate_conversion_placeholders(pick_session_id, seg_session_id_output, individual_outputs=False)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    logger.info(f"Converting picks to segmentation for object '{pick_object_name}'")
    logger.info(f"Source picks pattern: {pick_user_id}/{pick_session_id}")
    logger.info(f"Target segmentation template: {seg_name_output} ({seg_user_id_output}/{seg_session_id_output})")
    logger.info(f"Sphere radius: {radius}, voxel spacing: {voxel_spacing}")

    # Create type-safe Pydantic configuration
    selector_config = SelectorConfig(
        input_type="picks",
        output_type="segmentation",
        input_object_name=pick_object_name,
        input_user_id=pick_user_id,
        input_session_id=pick_session_id,
        output_object_name=seg_name_output,
        output_user_id=seg_user_id_output,
        output_session_id=seg_session_id_output,
        segmentation_name=seg_name_output,
        voxel_spacing=voxel_spacing,
    )

    config = TaskConfig(
        type="single_selector",
        selector=selector_config,
    )

    # Parallel discovery and processing - no sequential bottleneck!
    results = segmentation_from_picks_lazy_batch(
        root=root,
        config=config,
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
