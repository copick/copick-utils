import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import (
    add_clustering_options,
    add_input_option,
    add_output_option,
    add_workers_option,
)
from copick_utils.util.config_models import create_simple_config


@click.command(
    context_settings={"show_default": True},
    short_help="Convert picks to ellipsoid meshes.",
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
@add_input_option("picks")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--subdivisions",
    type=int,
    default=2,
    help="Number of ellipsoid subdivisions for mesh resolution.",
)
@optgroup.option(
    "--deduplicate-ellipsoids/--no-deduplicate-ellipsoids",
    is_flag=True,
    default=True,
    help="Merge overlapping ellipsoids to avoid duplicates.",
)
@optgroup.option(
    "--min-ellipsoid-distance",
    type=float,
    help="Minimum distance between ellipsoid centers for deduplication (default: 0.5 * average major axis).",
)
@add_clustering_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_output_option("mesh", default_tool="picks2ellipsoid")
@optgroup.option(
    "--individual-meshes/--no-individual-meshes",
    "-im",
    is_flag=True,
    default=False,
    help="Create individual meshes for each instance (enables {instance_id} placeholder).",
)
@add_debug_option
def picks2ellipsoid(
    config,
    run_names,
    input_uri,
    use_clustering,
    clustering_method,
    clustering_eps,
    clustering_min_samples,
    clustering_n_clusters,
    subdivisions,
    deduplicate_ellipsoids,
    min_ellipsoid_distance,
    all_clusters,
    workers,
    output_uri,
    individual_meshes,
    debug,
):
    """
    Convert picks to ellipsoid meshes.

    \b
    URI Format:
        Picks: object_name:user_id/session_id
        Meshes: object_name:user_id/session_id

    \b
    Examples:
        # Convert single pick set to single ellipsoid mesh
        copick convert picks2ellipsoid -i "ribosome:user1/manual-001" -o "ribosome:picks2ellipsoid/ellipsoid-001"

        # Create individual ellipsoid meshes
        copick convert picks2ellipsoid -i "ribosome:user1/manual-001" -o "ribosome:picks2ellipsoid/ellipsoid-{instance_id}" --individual-meshes

        # Convert all manual picks using pattern matching
        copick convert picks2ellipsoid -i "ribosome:user1/manual-.*" -o "ribosome:picks2ellipsoid/ellipsoid-{input_session_id}"
    """
    from copick_utils.converters.ellipsoid_from_picks import ellipsoid_from_picks_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Create config directly from URIs with smart defaults
    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="picks",
            output_uri=output_uri,
            output_type="mesh",
            individual_outputs=individual_meshes,
            command_name="picks2ellipsoid",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Extract parameters for logging
    input_params = parse_copick_uri(input_uri, "picks")
    output_params = parse_copick_uri(output_uri, "mesh")

    # Prepare clustering parameters
    clustering_params = {}
    if clustering_method == "dbscan":
        clustering_params = {"eps": clustering_eps, "min_samples": clustering_min_samples}
    elif clustering_method == "kmeans":
        clustering_params = {"n_clusters": clustering_n_clusters}

    logger.info(f"Converting picks to ellipsoid mesh for object '{input_params['object_name']}'")
    logger.info(f"Source picks pattern: {input_params['user_id']}/{input_params['session_id']}")
    logger.info(
        f"Target mesh template: {output_params['object_name']} ({output_params['user_id']}/{output_params['session_id']})",
    )

    # Parallel discovery and processing - no sequential bottleneck!
    results = ellipsoid_from_picks_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        subdivisions=subdivisions,
        all_clusters=all_clusters,
        deduplicate_ellipsoids_flag=deduplicate_ellipsoids,
        min_ellipsoid_distance=min_ellipsoid_distance,
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
