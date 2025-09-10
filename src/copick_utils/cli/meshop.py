"""CLI commands for mesh logical operations (boolean operations)."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from copick_utils.cli.input_output_selection import ConversionSelector, validate_conversion_placeholders
from copick_utils.cli.util import (
    add_boolean_operation_option,
    add_mesh_input_options,
    add_mesh_output_options,
    add_workers_option,
)


@click.command(
    context_settings={"show_default": True},
    short_help="Perform boolean operations between two meshes.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input meshes.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_mesh_input_options
@add_mesh_input_options(suffix="2")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_boolean_operation_option
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_mesh_output_options(default_tool="mesh-boolean")
@add_debug_option
def meshop(
    config,
    run_names,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    mesh_object_name2,
    mesh_user_id2,
    mesh_session_id2,
    operation,
    workers,
    mesh_object_name_output,
    mesh_user_id_output,
    mesh_session_id_output,
    individual_meshes,
    debug,
):
    """
    Perform boolean operations between two meshes.

    \b
    Supports the following boolean operations:
    - union: Combine both meshes
    - difference: First mesh minus second mesh
    - intersection: Common volume of both meshes
    - exclusion: Exclusive or (XOR) of both meshes

    \b
    Supports flexible input/output selection modes:
    - One-to-one: exact session IDs → exact session ID
    - Many-to-many: regex patterns → template with {input_session_id}

    \b
    Examples:
        # Union of two mesh sets
        copick meshop --operation union --mesh-session-id "manual-001" --input2-session-id "auto-001" --mesh-session-id "union-001"
        \b
        # Difference operation with pattern matching
        copick meshop --operation difference --mesh-session-id "manual-.*" --input2-session-id "mask-.*" --mesh-session-id "diff-{input_session_id}"
    """
    from copick_utils.logical.mesh_operations import (
        mesh_difference_batch,
        mesh_exclusion_batch,
        mesh_intersection_batch,
        mesh_union_batch,
    )

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Validate placeholder requirements
    try:
        validate_conversion_placeholders(mesh_session_id, mesh_session_id_output, individual_meshes)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Create selectors for both inputs
    selector1 = ConversionSelector(
        input_type="mesh",
        output_type="mesh",
        input_object_name=mesh_object_name,
        input_user_id=mesh_user_id,
        input_session_id=mesh_session_id,
        output_object_name=mesh_object_name_output or mesh_object_name,
        output_user_id=mesh_user_id_output,
        output_session_id=mesh_session_id_output,
        individual_outputs=individual_meshes,
    )

    selector2 = ConversionSelector(
        input_type="mesh",
        output_type="mesh",
        input_object_name=mesh_object_name2,
        input_user_id=mesh_user_id2,
        input_session_id=mesh_session_id2,
        output_object_name=mesh_object_name2,  # Not used for second input
        output_user_id=mesh_user_id2,  # Not used for second input
        output_session_id=mesh_session_id2,  # Not used for second input
        individual_outputs=False,  # Not used for second input
    )

    logger.info(f"Performing {operation} operation on meshes for object '{mesh_object_name}'")
    logger.info(f"First mesh pattern: {mesh_user_id}/{mesh_session_id}")
    logger.info(f"Second mesh pattern: {mesh_user_id2}/{mesh_session_id2}")
    logger.info(
        f"Target mesh template: {mesh_object_name_output or mesh_object_name} ({mesh_user_id_output}/{mesh_session_id_output})",
    )

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
        logger.warning("No matching first meshes found for operation")
        return

    if not all_tasks_2:
        logger.warning("No matching second meshes found for operation")
        return

    # Create paired tasks for boolean operations
    # For now, we'll match by run name and assume 1:1 correspondence
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

            # For simplicity, pair in order (could be made more sophisticated)
            for i, task1 in enumerate(run_tasks_1):
                if i < len(run_tasks_2):
                    task2 = run_tasks_2[i]

                    # Create combined task
                    paired_task = {
                        "input_mesh": task1["input_object"],
                        "input2_mesh": task2["input_object"],
                        "mesh_object_name": task1["output_object_name"],
                        "mesh_user_id": task1["output_user_id"],
                        "mesh_session_id": task1["output_session_id"],
                        "individual_meshes": task1["individual_outputs"],
                        "session_id_template": task1.get("session_id_template"),
                    }
                    paired_tasks.append(paired_task)

    if not paired_tasks:
        logger.warning("No paired mesh tasks found for boolean operation")
        return

    logger.info(f"Found {len(paired_tasks)} paired tasks across {len(runs_to_process)} runs")

    # Select the appropriate batch function based on operation
    batch_functions = {
        "union": mesh_union_batch,
        "difference": mesh_difference_batch,
        "intersection": mesh_intersection_batch,
        "exclusion": mesh_exclusion_batch,
    }

    batch_function = batch_functions[operation]

    results = batch_function(
        root=root,
        conversion_tasks=paired_tasks,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_vertices = sum(result.get("vertices_created", 0) for result in results.values() if result)
    total_faces = sum(result.get("faces_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("processed", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total {operation} operations completed: {total_processed}")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
