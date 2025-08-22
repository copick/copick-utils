"""CLI commands for converting picks to various mesh types."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

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
@add_mesh_output_options
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
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    create_multiple,
    individual_meshes,
    debug,
):
    """Convert picks to meshes using convex hull or alpha shapes."""
    from copick_utils.converters.mesh_from_picks import mesh_from_picks_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    if mesh_type == "alpha_shape" and alpha is None:
        raise click.BadParameter("Alpha parameter is required for alpha shapes")

    # Prepare clustering parameters
    clustering_params = {}
    if clustering_method == "dbscan":
        clustering_params = {"eps": clustering_eps, "min_samples": clustering_min_samples}
    elif clustering_method == "kmeans":
        clustering_params = {"n_clusters": clustering_n_clusters}

    logger.info(f"Converting picks to {mesh_type} mesh for object '{pick_object_name}'")
    logger.info(f"Source picks: {pick_user_id}/{pick_session_id}")
    logger.info(f"Target mesh: {mesh_object_name} ({mesh_user_id}/{mesh_session_id})")

    if run_names_list:
        logger.info(f"Processing {len(run_names_list)} specific runs")
    else:
        logger.info(f"Processing all {len(root.runs)} runs")

    results = mesh_from_picks_batch(
        root=root,
        pick_object_name=pick_object_name,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        mesh_object_name=mesh_object_name,
        mesh_session_id=mesh_session_id,
        mesh_user_id=mesh_user_id,
        mesh_type=mesh_type,
        alpha=alpha,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        create_multiple=create_multiple,
        individual_meshes=individual_meshes,
        session_id_template=mesh_session_id,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_vertices = sum(result.get("vertices_created", 0) for result in results.values() if result)
    total_faces = sum(result.get("faces_created", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")


@click.command(
    context_settings={"show_default": True},
    short_help="Convert picks to sphere meshes.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input picks.")
@optgroup.option(
    "--run-names", "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_pick_input_options
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--subdivisions",
    type=int,
    default=2,
    help="Number of sphere subdivisions for mesh resolution.",
)
@optgroup.option(
    "--deduplicate-spheres/--no-deduplicate-spheres",
    is_flag=True,
    default=True,
    help="Merge overlapping spheres to avoid duplicates.",
)
@optgroup.option(
    "--min-sphere-distance",
    type=float,
    help="Minimum distance between sphere centers for deduplication (default: 0.5 * average radius).",
)
@add_clustering_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_mesh_output_options
@add_debug_option
def picks2sphere(
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
    subdivisions,
    deduplicate_spheres,
    min_sphere_distance,
    create_multiple,
    workers,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    individual_meshes,
    debug,
):
    """Convert picks to sphere meshes."""
    from copick_utils.converters.sphere_from_picks import sphere_from_picks_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Prepare clustering parameters
    clustering_params = {}
    if clustering_method == "dbscan":
        clustering_params = {"eps": clustering_eps, "min_samples": clustering_min_samples}
    elif clustering_method == "kmeans":
        clustering_params = {"n_clusters": clustering_n_clusters}

    logger.info(f"Converting picks to sphere mesh for object '{pick_object_name}'")
    logger.info(f"Source picks: {pick_user_id}/{pick_session_id}")
    logger.info(f"Target mesh: {mesh_object_name} ({mesh_user_id}/{mesh_session_id})")

    if run_names_list:
        logger.info(f"Processing {len(run_names_list)} specific runs")
    else:
        logger.info(f"Processing all {len(root.runs)} runs")

    results = sphere_from_picks_batch(
        root=root,
        pick_object_name=pick_object_name,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        mesh_object_name=mesh_object_name,
        mesh_session_id=mesh_session_id,
        mesh_user_id=mesh_user_id,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        subdivisions=subdivisions,
        create_multiple=create_multiple,
        deduplicate_spheres=deduplicate_spheres,
        min_sphere_distance=min_sphere_distance,
        individual_meshes=individual_meshes,
        session_id_template=mesh_session_id,
        run_names=run_names_list,
        workers=workers,
    )
    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_vertices = sum(result.get("vertices_created", 0) for result in results.values() if result)
    total_faces = sum(result.get("faces_created", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")


@click.command(
    context_settings={"show_default": True},
    short_help="Convert picks to ellipsoid meshes.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input picks.")
@optgroup.option(
    "--run-names", "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_pick_input_options
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
@add_mesh_output_options
@add_debug_option
def picks2ellipsoid(
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
    subdivisions,
    deduplicate_ellipsoids,
    min_ellipsoid_distance,
    create_multiple,
    workers,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    individual_meshes,
    debug,
):
    """Convert picks to ellipsoid meshes."""
    from copick_utils.converters.ellipsoid_from_picks import ellipsoid_from_picks_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Prepare clustering parameters
    clustering_params = {}
    if clustering_method == "dbscan":
        clustering_params = {"eps": clustering_eps, "min_samples": clustering_min_samples}
    elif clustering_method == "kmeans":
        clustering_params = {"n_clusters": clustering_n_clusters}

    logger.info(f"Converting picks to ellipsoid mesh for object '{pick_object_name}'")
    logger.info(f"Source picks: {pick_user_id}/{pick_session_id}")
    logger.info(f"Target mesh: {mesh_object_name} ({mesh_user_id}/{mesh_session_id})")

    if run_names_list:
        logger.info(f"Processing {len(run_names_list)} specific runs")
    else:
        logger.info(f"Processing all {len(root.runs)} runs")

    results = ellipsoid_from_picks_batch(
        root=root,
        pick_object_name=pick_object_name,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        mesh_object_name=mesh_object_name,
        mesh_session_id=mesh_session_id,
        mesh_user_id=mesh_user_id,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        subdivisions=subdivisions,
        create_multiple=create_multiple,
        deduplicate_ellipsoids=deduplicate_ellipsoids,
        min_ellipsoid_distance=min_ellipsoid_distance,
        individual_meshes=individual_meshes,
        session_id_template=mesh_session_id,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_vertices = sum(result.get("vertices_created", 0) for result in results.values() if result)
    total_faces = sum(result.get("faces_created", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")


@click.command(
    context_settings={"show_default": True},
    short_help="Convert picks to plane meshes.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input picks.")
@optgroup.option(
    "--run-names", "-r",
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
@add_mesh_output_options
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
    create_multiple,
    workers,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    individual_meshes,
    debug,
):
    """Convert picks to plane meshes."""
    from copick_utils.converters.plane_from_picks import plane_from_picks_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Prepare clustering parameters
    clustering_params = {}
    if clustering_method == "dbscan":
        clustering_params = {"eps": clustering_eps, "min_samples": clustering_min_samples}
    elif clustering_method == "kmeans":
        clustering_params = {"n_clusters": clustering_n_clusters}

    logger.info(f"Converting picks to plane mesh for object '{pick_object_name}'")
    logger.info(f"Source picks: {pick_user_id}/{pick_session_id}")
    logger.info(f"Target mesh: {mesh_object_name} ({mesh_user_id}/{mesh_session_id})")

    if run_names_list:
        logger.info(f"Processing {len(run_names_list)} specific runs")
    else:
        logger.info(f"Processing all {len(root.runs)} runs")

    results = plane_from_picks_batch(
        root=root,
        pick_object_name=pick_object_name,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        mesh_object_name=mesh_object_name,
        mesh_session_id=mesh_session_id,
        mesh_user_id=mesh_user_id,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        padding=padding,
        create_multiple=create_multiple,
        individual_meshes=individual_meshes,
        session_id_template=mesh_session_id,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_vertices = sum(result.get("vertices_created", 0) for result in results.values() if result)
    total_faces = sum(result.get("faces_created", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")


@click.command(
    context_settings={"show_default": True},
    short_help="Convert picks to 2D surface meshes.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input picks.")
@optgroup.option(
    "--run-names", "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_pick_input_options
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--surface-method",
    type=click.Choice(["delaunay", "rbf", "grid"]),
    default="delaunay",
    help="Surface fitting method.",
)
@optgroup.option(
    "--grid-resolution",
    type=int,
    default=50,
    help="Resolution for grid-based surface methods.",
)
@add_clustering_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_mesh_output_options
@add_debug_option
def picks2surface(
    config,
    run_names,
    pick_object_name,
    pick_user_id,
    pick_session_id,
    surface_method,
    grid_resolution,
    use_clustering,
    clustering_method,
    clustering_eps,
    clustering_min_samples,
    clustering_n_clusters,
    create_multiple,
    workers,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    individual_meshes,
    debug,
):
    """Convert picks to 2D surface meshes."""
    from copick_utils.converters.surface_from_picks import surface_from_picks_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Prepare clustering parameters
    clustering_params = {}
    if clustering_method == "dbscan":
        clustering_params = {"eps": clustering_eps, "min_samples": clustering_min_samples}
    elif clustering_method == "kmeans":
        clustering_params = {"n_clusters": clustering_n_clusters}

    logger.info(f"Converting picks to {surface_method} surface mesh for object '{pick_object_name}'")
    logger.info(f"Source picks: {pick_user_id}/{pick_session_id}")
    logger.info(f"Target mesh: {mesh_object_name} ({mesh_user_id}/{mesh_session_id})")

    if run_names_list:
        logger.info(f"Processing {len(run_names_list)} specific runs")
    else:
        logger.info(f"Processing all {len(root.runs)} runs")

    results = surface_from_picks_batch(
        root=root,
        pick_object_name=pick_object_name,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        mesh_object_name=mesh_object_name,
        mesh_session_id=mesh_session_id,
        mesh_user_id=mesh_user_id,
        surface_method=surface_method,
        grid_resolution=grid_resolution,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        create_multiple=create_multiple,
        individual_meshes=individual_meshes,
        session_id_template=mesh_session_id,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_vertices = sum(result.get("vertices_created", 0) for result in results.values() if result)
    total_faces = sum(result.get("faces_created", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")
