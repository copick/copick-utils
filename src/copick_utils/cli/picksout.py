"""CLI commands for point filtering operations (inclusion/exclusion by volume)."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from copick_utils.cli.util import (
    add_mesh_input_options,
    add_pick_input_options,
    add_picks_output_options,
    add_segmentation_input_options,
    add_workers_option,
)
from copick_utils.converters.config_models import ReferenceConfig, SelectorConfig, TaskConfig
from copick_utils.logical.point_operations import picks_exclusion_by_mesh_lazy_batch


@click.command(
    context_settings={"show_default": True},
    short_help="Filter picks to exclude those inside a reference volume.",
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
@optgroup.group("\nReference Options", help="Options for reference volume (provide either mesh or segmentation).")
@add_mesh_input_options(prefix="ref")
@add_segmentation_input_options(
    prefix="ref",
    include_multilabel=False,
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output picks.")
@add_picks_output_options(default_tool="picksout")
@add_debug_option
def picksout(
    config,
    run_names,
    pick_object_name,
    pick_user_id,
    pick_session_id,
    ref_mesh_object_name,
    ref_mesh_user_id,
    ref_mesh_session_id,
    ref_seg_name,
    ref_seg_user_id,
    ref_seg_session_id,
    ref_voxel_spacing,
    workers,
    pick_object_name_output,
    pick_user_id_output,
    pick_session_id_output,
    debug,
):
    """
    Filter picks to exclude those inside a reference volume.

    \b
    The reference volume can be either a watertight mesh or a segmentation.
    Picks that fall inside the reference volume will be removed.

    \b
    Examples:
        # Exclude picks inside reference mesh
        copick logical picksout --pick-session-id "all-001" --ref-mesh-session-id "boundary-001" --pick-session-id "outside-001"
        \b
        # Exclude picks inside segmentation
        copick logical picksout --pick-session-id "all-001" --ref-seg-session-id "mask-001" --ref-voxel-spacing 10.0 --pick-session-id "outside-001"
    """

    logger = get_logger(__name__, debug=debug)

    # Validate that exactly one reference type is provided
    ref_mesh_provided = bool(ref_mesh_object_name or ref_mesh_user_id or ref_mesh_session_id)
    ref_seg_provided = bool(ref_seg_name or ref_seg_user_id or ref_seg_session_id)

    if not ref_mesh_provided and not ref_seg_provided:
        raise click.BadParameter("Must provide either reference mesh or reference segmentation")
    if ref_mesh_provided and ref_seg_provided:
        raise click.BadParameter("Cannot provide both reference mesh and reference segmentation")

    if ref_seg_provided and not ref_voxel_spacing:
        raise click.BadParameter("--ref-voxel-spacing is required when using reference segmentation")

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    logger.info(f"Excluding picks inside reference volume for object '{pick_object_name}'")
    logger.info(f"Source picks pattern: {pick_user_id}/{pick_session_id}")
    if ref_mesh_provided:
        logger.info(f"Reference mesh: {ref_mesh_object_name} ({ref_mesh_user_id}/{ref_mesh_session_id})")
    else:
        logger.info(f"Reference segmentation: {ref_seg_name} ({ref_seg_user_id}/{ref_seg_session_id})")
    logger.info(f"Target picks template: {pick_object_name_output} ({pick_user_id_output}/{pick_session_id_output})")

    # Create type-safe Pydantic configuration
    selector_config = SelectorConfig(
        input_type="picks",
        output_type="picks",
        input_object_name=pick_object_name,
        input_user_id=pick_user_id,
        input_session_id=pick_session_id,
        output_object_name=pick_object_name_output,
        output_user_id=pick_user_id_output,
        output_session_id=pick_session_id_output,
    )

    # Create reference configuration
    if ref_mesh_provided:
        reference_config = ReferenceConfig(
            reference_type="mesh",
            object_name=ref_mesh_object_name,
            user_id=ref_mesh_user_id,
            session_id=ref_mesh_session_id,
        )
    else:
        reference_config = ReferenceConfig(
            reference_type="segmentation",
            object_name=ref_seg_name,
            user_id=ref_seg_user_id,
            session_id=ref_seg_session_id,
            voxel_spacing=ref_voxel_spacing,
        )

    config = TaskConfig(
        type="single_selector_with_reference",
        selector=selector_config,
        reference=reference_config,
    )

    # Parallel discovery and processing - no sequential bottleneck!
    results = picks_exclusion_by_mesh_lazy_batch(
        root=root,
        config=config,
        run_names=run_names_list,
        workers=workers,
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
    logger.info(f"Total exclusion operations completed: {total_processed}")
    logger.info(f"Total points excluded (remaining): {total_points}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
