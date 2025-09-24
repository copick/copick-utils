"""CLI commands for distance-based limiting operations."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from copick_utils.cli.input_output_selection import ConversionSelector
from copick_utils.cli.util import (
    add_distance_options,
    add_mesh_input_options,
    add_pick_input_options,
    add_picks_output_options,
    add_segmentation_input_options,
    add_workers_option,
)


@click.command(
    context_settings={"show_default": True},
    short_help="Limit picks to those within distance of a reference surface.",
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
@optgroup.group("\nReference Options", help="Options for reference surface (provide either mesh or segmentation).")
@add_mesh_input_options(prefix="ref", required=False)
@add_segmentation_input_options(
    prefix="ref",
    include_multilabel=False,
    required=False,
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_distance_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output picks.")
@add_picks_output_options(default_tool="clippicks")
@add_debug_option
def clippicks(
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
    max_distance,
    sampling_density,
    workers,
    pick_object_name_output,
    pick_user_id_output,
    pick_session_id_output,
    debug,
):
    """
    Limit picks to those within a certain distance of a reference surface.

    \b
    The reference surface can be either a mesh or a segmentation.
    Only picks within the specified distance will be kept.

    \b
    Examples:
        # Limit picks to those near reference mesh surface
        copick clippicks --pick-session-id "all-001" --ref-mesh-session-id "boundary-001" --max-distance 50.0 --pick-session-id "limited-001"
        \b
        # Limit using segmentation as reference
        copick clippicks --pick-session-id "all-001" --ref-seg-session-id "mask-001" --ref-voxel-spacing 10.0 --max-distance 100.0 --pick-session-id "limited-001"
    """
    from copick_utils.logical.distance_operations import limit_picks_by_distance_batch

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

    # Create selector for input picks
    selector = ConversionSelector(
        input_type="picks",
        output_type="picks",
        input_object_name=pick_object_name,
        input_user_id=pick_user_id,
        input_session_id=pick_session_id,
        output_object_name=pick_object_name_output,
        output_user_id=pick_user_id_output,
        output_session_id=pick_session_id_output,
        individual_outputs=False,
    )

    logger.info(f"Limiting picks by distance for object '{pick_object_name}'")
    logger.info(f"Source picks pattern: {pick_user_id}/{pick_session_id}")
    if ref_mesh_provided:
        logger.info(f"Reference mesh: {ref_mesh_object_name} ({ref_mesh_user_id}/{ref_mesh_session_id})")
    else:
        logger.info(f"Reference segmentation: {ref_seg_name} ({ref_seg_user_id}/{ref_seg_session_id})")
    logger.info(f"Maximum distance: {max_distance} angstroms")
    logger.info(f"Target picks template: {pick_object_name_output} ({pick_user_id_output}/{pick_session_id_output})")

    # Collect all conversion tasks across runs
    all_tasks = []
    runs_to_process = root.runs if run_names_list is None else [root.get_run(name) for name in run_names_list]

    for run in runs_to_process:
        tasks = selector.get_conversion_tasks(run)

        # Add reference information to each task
        for task in tasks:
            if ref_mesh_provided:
                # Find reference mesh in the same run
                ref_meshes = run.get_meshes(
                    object_name=ref_mesh_object_name,
                    user_id=ref_mesh_user_id,
                    session_id=ref_mesh_session_id,
                )
                if ref_meshes:
                    task["reference_mesh"] = ref_meshes[0]
                    task["reference_segmentation"] = None
                else:
                    logger.warning(f"No reference mesh found in run {run.name}")
                    continue
            else:
                # Find reference segmentation in the same run
                ref_segs = run.get_segmentations(
                    name=ref_seg_name,
                    user_id=ref_seg_user_id,
                    session_id=ref_seg_session_id,
                    voxel_size=ref_voxel_spacing,
                )
                if ref_segs:
                    task["reference_mesh"] = None
                    task["reference_segmentation"] = ref_segs[0]
                else:
                    logger.warning(f"No reference segmentation found in run {run.name}")
                    continue

            task["max_distance"] = max_distance
            task["sampling_density"] = sampling_density
            all_tasks.append(task)

    if not all_tasks:
        logger.warning("No matching picks/reference pairs found for distance limiting")
        return

    logger.info(f"Found {len(all_tasks)} tasks across {len(runs_to_process)} runs")

    results = limit_picks_by_distance_batch(
        root=root,
        conversion_tasks=all_tasks,
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
    logger.info(f"Total distance limiting operations completed: {total_processed}")
    logger.info(f"Total points created: {total_points}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
