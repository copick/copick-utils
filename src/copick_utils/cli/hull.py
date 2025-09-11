"""CLI command for computing various hull operations on meshes."""
import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from copick_utils.cli.input_output_selection import ConversionSelector, validate_conversion_placeholders
from copick_utils.cli.util import (
    add_mesh_input_options,
    add_mesh_output_options,
    add_workers_option,
)


@click.command(
    context_settings={"show_default": True},
    short_help="Compute hull operations on meshes.",
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
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--hull-type",
    type=click.Choice(["convex"]),
    default="convex",
    help="Type of hull to compute.",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_mesh_output_options(default_tool="hull")
@add_debug_option
def hull(
    config,
    run_names,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    hull_type,
    workers,
    mesh_object_name_output,
    mesh_user_id_output,
    mesh_session_id_output,
    individual_meshes,
    debug,
):
    """
    Compute hull operations on meshes.

    \b
    Currently supports convex hull computation, where the convex hull is the
    smallest convex shape that contains all vertices of the original mesh.

    \b
    Examples:
        # Compute convex hull for all meshes of type "my-mesh"
        copick hull -mo my-mesh -mu user1 -ms session1 -moo hull -muo hull -mso hull-session

        \b
        # Process specific runs
        copick hull -r run1 -r run2 -mo my-mesh -mu user1 -ms session1 --hull-type convex
    """
    from copick_utils.process.hull import hull_from_mesh_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Validate placeholder requirements
    try:
        validate_conversion_placeholders(mesh_session_id, mesh_session_id_output, individual_meshes)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Create conversion selector
    selector = ConversionSelector(
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

    logger.info(f"Computing {hull_type} hull for meshes '{mesh_object_name}'")
    logger.info(f"Selection mode: {selector.get_mode_description()}")
    logger.info(f"Source mesh pattern: {mesh_user_id}/{mesh_session_id}")
    logger.info(
        f"Target mesh template: {mesh_object_name_output or mesh_object_name} ({mesh_user_id_output}/{mesh_session_id_output})",
    )

    # Collect all conversion tasks across runs
    all_tasks = []
    runs_to_process = root.runs if run_names_list is None else [root.get_run(name) for name in run_names_list]

    for run in runs_to_process:
        tasks = selector.get_conversion_tasks(run)
        all_tasks.extend(tasks)

    print(all_tasks)

    if not all_tasks:
        logger.warning("No matching meshes found for hull computation")
        return

    logger.info(f"Found {len(all_tasks)} conversion tasks across {len(runs_to_process)} runs")

    results = hull_from_mesh_batch(
        root=root,
        conversion_tasks=all_tasks,
        run_names=run_names_list,
        workers=workers,
        hull_type=hull_type,
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
    logger.info(f"Total {hull_type} hull operations completed: {total_processed}")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
