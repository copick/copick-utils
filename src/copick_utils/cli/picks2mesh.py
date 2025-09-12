import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from copick_utils.cli.input_output_selection import InputOutputSelector, validate_placeholders
from copick_utils.cli.util import (
    add_clustering_options,
    add_mesh_output_options,
    add_pick_input_options,
    add_workers_option,
)


@click.command(
    context_settings={"show_default": True},
    short_help="Convert picks to mesh using convex hull or alpha shapes.",
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
@optgroup.option(
    "--mesh-type",
    "-t",
    type=click.Choice(["convex_hull", "alpha_shape"]),
    default="convex_hull",
    help="Type of mesh to create.",
)
@optgroup.option(
    "--alpha",
    "-a",
    type=float,
    help="Alpha parameter for alpha shapes (required if mesh-type=alpha_shape).",
)
@add_clustering_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_mesh_output_options(default_tool="picks2mesh")
@add_debug_option
def picks2mesh(
    config,
    run_names,
    pick_object_name,
    pick_user_id,
    pick_session_id,
    mesh_type,
    alpha,
    use_clustering,
    clustering_method,
    clustering_eps,
    clustering_min_samples,
    clustering_n_clusters,
    workers,
    mesh_object_name_output,
    mesh_user_id_output,
    mesh_session_id_output,
    all_clusters,
    individual_meshes,
    debug,
):
    """
    Convert picks to meshes using convex hull or alpha shapes.

    \b
    Supports flexible input/output selection modes:
    - One-to-one: exact session ID → exact session ID
    - One-to-many: exact session ID → template with {instance_id}
    - Many-to-many: regex pattern → template with {input_session_id} and {instance_id}

    \b
    Examples:
        # Convert single pick set to single mesh
        copick convert picks2mesh --pick-session-id "manual-001" --mesh-session-id "mesh-001"

        # Create individual meshes from clusters
        copick convert picks2mesh --pick-session-id "manual-001" --mesh-session-id "mesh-{instance_id}" --individual-meshes

        # Convert all manual picks using pattern matching
        copick convert picks2mesh --pick-session-id "manual-.*" --mesh-session-id "mesh-{input_session_id}"
    """
    from copick_utils.converters.mesh_from_picks import mesh_from_picks_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    if mesh_type == "alpha_shape" and alpha is None:
        raise click.BadParameter("Alpha parameter is required for alpha shapes")

    # Validate placeholder requirements
    try:
        validate_placeholders(pick_session_id, mesh_session_id_output, individual_meshes)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Prepare clustering parameters
    clustering_params = {}
    if clustering_method == "dbscan":
        clustering_params = {"eps": clustering_eps, "min_samples": clustering_min_samples}
    elif clustering_method == "kmeans":
        clustering_params = {"n_clusters": clustering_n_clusters}

    # Create input/output selector
    selector = InputOutputSelector(
        pick_object_name=pick_object_name,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        mesh_object_name=mesh_object_name_output,
        mesh_user_id=mesh_user_id_output,
        mesh_session_id=mesh_session_id_output,
        individual_meshes=individual_meshes,
    )

    logger.info(f"Converting picks to {mesh_type} mesh for object '{pick_object_name}'")
    logger.info(f"Selection mode: {selector.get_mode_description()}")
    logger.info(f"Source picks pattern: {pick_user_id}/{pick_session_id}")
    logger.info(f"Target mesh template: {selector.mesh_object_name} ({mesh_user_id_output}/{mesh_session_id_output})")

    # Collect all conversion tasks across runs
    all_tasks = []
    runs_to_process = root.runs if run_names_list is None else [root.get_run(name) for name in run_names_list]

    for run in runs_to_process:
        tasks = selector.get_conversion_tasks(run)
        all_tasks.extend(tasks)

    if not all_tasks:
        logger.warning("No matching picks found for conversion")
        return

    logger.info(f"Found {len(all_tasks)} conversion tasks across {len(runs_to_process)} runs")

    results = mesh_from_picks_batch(
        root=root,
        conversion_tasks=all_tasks,
        run_names=run_names_list,
        workers=workers,
        mesh_type=mesh_type,
        alpha=alpha,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        all_clusters=all_clusters,
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
    logger.info(f"Total conversion tasks completed: {total_processed}")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
