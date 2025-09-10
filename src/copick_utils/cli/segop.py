"""CLI commands for segmentation logical operations (boolean operations)."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from copick_utils.cli.input_output_selection import ConversionSelector
from copick_utils.cli.util import (
    add_boolean_operation_option,
    add_segmentation_boolean_output_options,
    add_segmentation_input_options,
    add_workers_option,
)


@click.command(
    context_settings={"show_default": True},
    short_help="Perform boolean operations between two segmentations.",
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
@add_segmentation_input_options(include_multilabel=False)
@add_segmentation_input_options(suffix="2", include_multilabel=False, include_voxel_spacing=False)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_boolean_operation_option
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_segmentation_boolean_output_options(default_tool="segop")
@add_debug_option
def segop(
    config,
    run_names,
    seg_name,
    seg_user_id,
    seg_session_id,
    voxel_spacing,
    seg_name2,
    seg_user_id2,
    seg_session_id2,
    operation,
    workers,
    seg_name_output,
    seg_user_id_output,
    seg_session_id_output,
    voxel_spacing_output,
    tomo_type,
    debug,
):
    """
    Perform boolean operations between two segmentations.

    \b
    Supports the following boolean operations:
    - union: Combine both segmentations (logical OR)
    - difference: First segmentation minus second segmentation
    - intersection: Common voxels of both segmentations (logical AND)
    - exclusion: Exclusive or (XOR) of both segmentations

    \b
    Note: Both input segmentations should be binary (non-multilabel) for meaningful results.

    \b
    Examples:
        # Union of two segmentation sets
        copick segop --operation union --seg-session-id "manual-001" --input2-session-id "auto-001" --seg-session-id "union-001" --voxel-spacing 10.0
        \b
        # Difference operation with pattern matching
        copick segop --operation difference --seg-session-id "manual-.*" --input2-session-id "mask-.*" --seg-session-id "diff-{input_session_id}" --voxel-spacing 10.0
    """
    from copick_utils.logical.segmentation_operations import (
        segmentation_difference_batch,
        segmentation_exclusion_batch,
        segmentation_intersection_batch,
        segmentation_union_batch,
    )

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Create selectors for both inputs
    selector1 = ConversionSelector(
        input_type="segmentation",
        output_type="segmentation",
        input_object_name=seg_name,
        input_user_id=seg_user_id,
        input_session_id=seg_session_id,
        output_object_name=seg_name_output,
        output_user_id=seg_user_id_output,
        output_session_id=seg_session_id_output,
        individual_outputs=False,  # Segmentations don't support individual outputs
        segmentation_name=seg_name,
        voxel_spacing=voxel_spacing,
    )

    selector2 = ConversionSelector(
        input_type="segmentation",
        output_type="segmentation",
        input_object_name=seg_name2,
        input_user_id=seg_user_id2,
        input_session_id=seg_session_id2,
        output_object_name=seg_name2,  # Not used for second input
        output_user_id=seg_user_id2,  # Not used for second input
        output_session_id=seg_session_id2,  # Not used for second input
        individual_outputs=False,
        segmentation_name=seg_name2,
        voxel_spacing=voxel_spacing,  # Use same voxel spacing for consistency
    )

    logger.info(f"Performing {operation} operation on segmentations for '{seg_name}'")
    logger.info(f"First segmentation pattern: {seg_user_id}/{seg_session_id}")
    logger.info(f"Second segmentation pattern: {seg_user_id2}/{seg_session_id2}")
    logger.info(f"Target segmentation template: {seg_name_output} ({seg_user_id_output}/{seg_session_id_output})")

    # Collect conversion tasks for both inputs across runs
    all_tasks_1 = []
    all_tasks_2 = []
    runs_to_process = root.runs if run_names_list is None else [root.get_run(name) for name in run_names_list]

    for run in runs_to_process:
        tasks1 = selector1.get_conversion_tasks(run)
        tasks2 = selector2.get_conversion_tasks(run)
        all_tasks_1.extend(tasks1)
        all_tasks_2.extend(tasks2)

    if not all_tasks_1:
        logger.warning("No matching first segmentations found for operation")
        return

    if not all_tasks_2:
        logger.warning("No matching second segmentations found for operation")
        return

    # Create paired tasks for boolean operations
    paired_tasks = []

    # Group tasks by run name
    tasks1_by_run = {}
    for task in all_tasks_1:
        run_name = task["input_object"].run.name
        if run_name not in tasks1_by_run:
            tasks1_by_run[run_name] = []
        tasks1_by_run[run_name].append(task)

    tasks2_by_run = {}
    for task in all_tasks_2:
        run_name = task["input_object"].run.name
        if run_name not in tasks2_by_run:
            tasks2_by_run[run_name] = []
        tasks2_by_run[run_name].append(task)

    # Pair up tasks from each run
    for run_name in tasks1_by_run:
        if run_name in tasks2_by_run:
            run_tasks_1 = tasks1_by_run[run_name]
            run_tasks_2 = tasks2_by_run[run_name]

            # For simplicity, pair in order
            for i, task1 in enumerate(run_tasks_1):
                if i < len(run_tasks_2):
                    task2 = run_tasks_2[i]

                    # Create combined task
                    paired_task = {
                        "input_segmentation": task1["input_object"],
                        "input2_segmentation": task2["input_object"],
                        "segmentation_object_name": task1["output_object_name"],
                        "segmentation_user_id": task1["output_user_id"],
                        "segmentation_session_id": task1["output_session_id"],
                        "voxel_spacing": voxel_spacing_output,
                        "tomo_type": tomo_type,
                        "is_multilabel": False,
                    }
                    paired_tasks.append(paired_task)

    if not paired_tasks:
        logger.warning("No paired segmentation tasks found for boolean operation")
        return

    logger.info(f"Found {len(paired_tasks)} paired tasks across {len(runs_to_process)} runs")

    # Select the appropriate batch function based on operation
    batch_functions = {
        "union": segmentation_union_batch,
        "difference": segmentation_difference_batch,
        "intersection": segmentation_intersection_batch,
        "exclusion": segmentation_exclusion_batch,
    }

    batch_function = batch_functions[operation]

    results = batch_function(
        root=root,
        conversion_tasks=paired_tasks,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_voxels = sum(result.get("voxels_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("processed", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total {operation} operations completed: {total_processed}")
    logger.info(f"Total voxels created: {total_voxels}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
