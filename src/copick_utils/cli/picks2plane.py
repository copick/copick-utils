import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from copick_utils.cli.input_output_selection import validate_placeholders
from copick_utils.cli.util import (
    add_clustering_options,
    add_mesh_output_options,
    add_pick_input_options,
    add_workers_option,
)
from copick_utils.converters.config_models import SelectorConfig, TaskConfig
from copick_utils.converters.plane_from_picks import plane_from_picks_lazy_batch


@click.command(
    context_settings={"show_default": True},
    short_help="Convert picks to plane meshes.",
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
    "--padding",
    type=float,
    default=1.2,
    help="Padding factor for plane size (1.0=exact fit, >1.0=larger plane).",
)
@add_clustering_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_mesh_output_options(default_tool="picks2plane")
@add_debug_option
def picks2plane(
    config,
    run_names,
    pick_object_name,
    pick_user_id,
    pick_session_id,
    use_clustering,
    clustering_method,
    clustering_eps,
    clustering_min_samples,
    clustering_n_clusters,
    padding,
    all_clusters,
    workers,
    mesh_object_name_output,
    mesh_user_id_output,
    mesh_session_id_output,
    individual_meshes,
    debug,
):
    """
    Convert picks to plane meshes.

    \b
    Supports flexible input/output selection modes:
    - One-to-one: exact session ID → exact session ID
    - One-to-many: exact session ID → template with {instance_id}
    - Many-to-many: regex pattern → template with {input_session_id} and {instance_id}

    \b
    Examples:
        \b
        # Convert single pick set to single plane mesh
        copick convert picks2plane --pick-session-id "manual-001" --mesh-session-id "plane-001"
        \b
        # Create individual plane meshes from clusters
        copick convert picks2plane --pick-session-id "manual-001" --mesh-session-id "plane-{instance_id}" --individual-meshes
        \b
        # Convert all manual picks using pattern matching
        copick convert picks2plane --pick-session-id "manual-.*" --mesh-session-id "plane-{input_session_id}"
    """

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

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

    logger.info(f"Converting picks to plane mesh for object '{pick_object_name}'")
    logger.info(f"Source picks pattern: {pick_user_id}/{pick_session_id}")
    logger.info(f"Target mesh template: {mesh_object_name_output} ({mesh_user_id_output}/{mesh_session_id_output})")

    # Create type-safe Pydantic configuration
    selector_config = SelectorConfig(
        input_type="picks",
        output_type="mesh",
        input_object_name=pick_object_name,
        input_user_id=pick_user_id,
        input_session_id=pick_session_id,
        output_object_name=mesh_object_name_output,
        output_user_id=mesh_user_id_output,
        output_session_id=mesh_session_id_output,
        individual_outputs=individual_meshes,
    )

    config = TaskConfig(
        type="single_selector",
        selector=selector_config,
    )

    # Parallel discovery and processing - no sequential bottleneck!
    results = plane_from_picks_lazy_batch(
        root=root,
        config=config,
        run_names=run_names_list,
        workers=workers,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        padding=padding,
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
