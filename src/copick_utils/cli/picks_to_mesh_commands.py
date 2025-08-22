"""CLI commands for converting picks to various mesh types."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger


@click.command(
    context_settings={"show_default": True},
    short_help="Convert picks to mesh using convex hull or alpha shapes.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input picks.")
@optgroup.option(
    "--run-names",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@optgroup.option(
    "--pick-object-name",
    required=True,
    help="Name of the pick object to convert.",
)
@optgroup.option(
    "--pick-user-id",
    required=True,
    help="User ID of the picks to convert.",
)
@optgroup.option(
    "--pick-session-id",
    required=True,
    help="Session ID of the picks to convert.",
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--mesh-type",
    type=click.Choice(["convex_hull", "alpha_shape"]),
    default="convex_hull",
    help="Type of mesh to create.",
)
@optgroup.option(
    "--alpha",
    type=float,
    help="Alpha parameter for alpha shapes (required if mesh-type=alpha_shape).",
)
@optgroup.option(
    "--use-clustering/--no-use-clustering",
    is_flag=True,
    default=False,
    help="Cluster points before mesh creation.",
)
@optgroup.option(
    "--clustering-method",
    type=click.Choice(["dbscan", "kmeans"]),
    default="dbscan",
    help="Clustering method.",
)
@optgroup.option(
    "--clustering-eps",
    type=float,
    default=1.0,
    help="DBSCAN eps parameter - maximum distance between points in a cluster (in angstroms).",
)
@optgroup.option(
    "--clustering-min-samples",
    type=int,
    default=3,
    help="DBSCAN min_samples parameter.",
)
@optgroup.option(
    "--clustering-n-clusters",
    type=int,
    default=1,
    help="K-means n_clusters parameter.",
)
@optgroup.option(
    "--create-multiple/--no-create-multiple",
    is_flag=True,
    default=False,
    help="Create separate meshes for each cluster.",
)
@optgroup.option(
    "--individual-meshes/--no-individual-meshes",
    is_flag=True,
    default=False,
    help="Create individual mesh files for each mesh instead of combining them.",
)
@optgroup.option(
    "--session-id-template",
    help="Template for individual mesh session IDs. Use {base_session_id} and {mesh_id} as placeholders (default: '{base_session_id}-{mesh_id:03d}').",
)
@optgroup.option(
    "--workers",
    type=int,
    default=8,
    help="Number of worker processes.",
)
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@optgroup.option(
    "--mesh-object-name",
    required=True,
    help="Name of the mesh object to create.",
)
@optgroup.option(
    "--mesh-user-id",
    default="from-picks",
    help="User ID for created mesh.",
)
@optgroup.option(
    "--mesh-session-id",
    default="0",
    help="Session ID for created mesh.",
)
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
    create_multiple,
    individual_meshes,
    session_id_template,
    workers,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
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
        session_id_template=session_id_template,
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
    "--run-names",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@optgroup.option(
    "--pick-object-name",
    required=True,
    help="Name of the pick object to convert.",
)
@optgroup.option(
    "--pick-user-id",
    required=True,
    help="User ID of the picks to convert.",
)
@optgroup.option(
    "--pick-session-id",
    required=True,
    help="Session ID of the picks to convert.",
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--use-clustering/--no-use-clustering",
    is_flag=True,
    default=False,
    help="Cluster points before sphere fitting.",
)
@optgroup.option(
    "--clustering-method",
    type=click.Choice(["dbscan", "kmeans"]),
    default="dbscan",
    help="Clustering method.",
)
@optgroup.option(
    "--clustering-eps",
    type=float,
    default=1.0,
    help="DBSCAN eps parameter - maximum distance between points in a cluster (in angstroms).",
)
@optgroup.option(
    "--clustering-min-samples",
    type=int,
    default=3,
    help="DBSCAN min_samples parameter.",
)
@optgroup.option(
    "--clustering-n-clusters",
    type=int,
    default=1,
    help="K-means n_clusters parameter.",
)
@optgroup.option(
    "--subdivisions",
    type=int,
    default=2,
    help="Number of sphere subdivisions for mesh resolution.",
)
@optgroup.option(
    "--create-multiple/--no-create-multiple",
    is_flag=True,
    default=False,
    help="Create separate spheres for each cluster.",
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
@optgroup.option(
    "--individual-meshes/--no-individual-meshes",
    is_flag=True,
    default=False,
    help="Create individual mesh files for each sphere instead of combining them.",
)
@optgroup.option(
    "--session-id-template",
    help="Template for individual mesh session IDs. Use {base_session_id} and {sphere_id} as placeholders (default: '{base_session_id}-{sphere_id:03d}').",
)
@optgroup.option(
    "--workers",
    type=int,
    default=8,
    help="Number of worker processes.",
)
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@optgroup.option(
    "--mesh-object-name",
    required=True,
    help="Name of the mesh object to create.",
)
@optgroup.option(
    "--mesh-user-id",
    default="from-picks",
    help="User ID for created mesh.",
)
@optgroup.option(
    "--mesh-session-id",
    default="0",
    help="Session ID for created mesh.",
)
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
    create_multiple,
    deduplicate_spheres,
    min_sphere_distance,
    individual_meshes,
    session_id_template,
    workers,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
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
        session_id_template=session_id_template,
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
    "--run-names",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@optgroup.option(
    "--pick-object-name",
    required=True,
    help="Name of the pick object to convert.",
)
@optgroup.option(
    "--pick-user-id",
    required=True,
    help="User ID of the picks to convert.",
)
@optgroup.option(
    "--pick-session-id",
    required=True,
    help="Session ID of the picks to convert.",
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--use-clustering/--no-use-clustering",
    is_flag=True,
    default=False,
    help="Cluster points before ellipsoid fitting.",
)
@optgroup.option(
    "--clustering-method",
    type=click.Choice(["dbscan", "kmeans"]),
    default="dbscan",
    help="Clustering method.",
)
@optgroup.option(
    "--clustering-eps",
    type=float,
    default=1.0,
    help="DBSCAN eps parameter - maximum distance between points in a cluster (in angstroms).",
)
@optgroup.option(
    "--clustering-min-samples",
    type=int,
    default=3,
    help="DBSCAN min_samples parameter.",
)
@optgroup.option(
    "--clustering-n-clusters",
    type=int,
    default=1,
    help="K-means n_clusters parameter.",
)
@optgroup.option(
    "--subdivisions",
    type=int,
    default=2,
    help="Number of ellipsoid subdivisions for mesh resolution.",
)
@optgroup.option(
    "--create-multiple/--no-create-multiple",
    is_flag=True,
    default=False,
    help="Create separate ellipsoids for each cluster.",
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
@optgroup.option(
    "--individual-meshes/--no-individual-meshes",
    is_flag=True,
    default=False,
    help="Create individual mesh files for each ellipsoid instead of combining them.",
)
@optgroup.option(
    "--session-id-template",
    help="Template for individual mesh session IDs. Use {base_session_id} and {ellipsoid_id} as placeholders (default: '{base_session_id}-{ellipsoid_id:03d}').",
)
@optgroup.option(
    "--workers",
    type=int,
    default=8,
    help="Number of worker processes.",
)
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@optgroup.option(
    "--mesh-object-name",
    required=True,
    help="Name of the mesh object to create.",
)
@optgroup.option(
    "--mesh-user-id",
    default="from-picks",
    help="User ID for created mesh.",
)
@optgroup.option(
    "--mesh-session-id",
    default="0",
    help="Session ID for created mesh.",
)
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
    create_multiple,
    deduplicate_ellipsoids,
    min_ellipsoid_distance,
    individual_meshes,
    session_id_template,
    workers,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
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
        session_id_template=session_id_template,
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
    "--run-names",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@optgroup.option(
    "--pick-object-name",
    required=True,
    help="Name of the pick object to convert.",
)
@optgroup.option(
    "--pick-user-id",
    required=True,
    help="User ID of the picks to convert.",
)
@optgroup.option(
    "--pick-session-id",
    required=True,
    help="Session ID of the picks to convert.",
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--use-clustering/--no-use-clustering",
    is_flag=True,
    default=False,
    help="Cluster points before plane fitting.",
)
@optgroup.option(
    "--clustering-method",
    type=click.Choice(["dbscan", "kmeans"]),
    default="dbscan",
    help="Clustering method.",
)
@optgroup.option(
    "--clustering-eps",
    type=float,
    default=1.0,
    help="DBSCAN eps parameter - maximum distance between points in a cluster (in angstroms).",
)
@optgroup.option(
    "--clustering-min-samples",
    type=int,
    default=3,
    help="DBSCAN min_samples parameter.",
)
@optgroup.option(
    "--clustering-n-clusters",
    type=int,
    default=1,
    help="K-means n_clusters parameter.",
)
@optgroup.option(
    "--padding",
    type=float,
    default=1.2,
    help="Padding factor for plane size (1.0=exact fit, >1.0=larger plane).",
)
@optgroup.option(
    "--create-multiple/--no-create-multiple",
    is_flag=True,
    default=False,
    help="Create separate planes for each cluster.",
)
@optgroup.option(
    "--individual-meshes/--no-individual-meshes",
    is_flag=True,
    default=False,
    help="Create individual mesh files for each plane instead of combining them.",
)
@optgroup.option(
    "--session-id-template",
    help="Template for individual mesh session IDs. Use {base_session_id} and {plane_id} as placeholders (default: '{base_session_id}-{plane_id:03d}').",
)
@optgroup.option(
    "--workers",
    type=int,
    default=8,
    help="Number of worker processes.",
)
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@optgroup.option(
    "--mesh-object-name",
    required=True,
    help="Name of the mesh object to create.",
)
@optgroup.option(
    "--mesh-user-id",
    default="from-picks",
    help="User ID for created mesh.",
)
@optgroup.option(
    "--mesh-session-id",
    default="0",
    help="Session ID for created mesh.",
)
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
    individual_meshes,
    session_id_template,
    workers,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
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
        session_id_template=session_id_template,
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
    "--run-names",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@optgroup.option(
    "--pick-object-name",
    required=True,
    help="Name of the pick object to convert.",
)
@optgroup.option(
    "--pick-user-id",
    required=True,
    help="User ID of the picks to convert.",
)
@optgroup.option(
    "--pick-session-id",
    required=True,
    help="Session ID of the picks to convert.",
)
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
@optgroup.option(
    "--use-clustering/--no-use-clustering",
    is_flag=True,
    default=False,
    help="Cluster points before surface fitting.",
)
@optgroup.option(
    "--clustering-method",
    type=click.Choice(["dbscan", "kmeans"]),
    default="dbscan",
    help="Clustering method.",
)
@optgroup.option(
    "--clustering-eps",
    type=float,
    default=1.0,
    help="DBSCAN eps parameter - maximum distance between points in a cluster (in angstroms).",
)
@optgroup.option(
    "--clustering-min-samples",
    type=int,
    default=3,
    help="DBSCAN min_samples parameter.",
)
@optgroup.option(
    "--clustering-n-clusters",
    type=int,
    default=1,
    help="K-means n_clusters parameter.",
)
@optgroup.option(
    "--create-multiple/--no-create-multiple",
    is_flag=True,
    default=False,
    help="Create separate surfaces for each cluster.",
)
@optgroup.option(
    "--individual-meshes/--no-individual-meshes",
    is_flag=True,
    default=False,
    help="Create individual mesh files for each surface instead of combining them.",
)
@optgroup.option(
    "--session-id-template",
    help="Template for individual mesh session IDs. Use {base_session_id} and {surface_id} as placeholders (default: '{base_session_id}-{surface_id:03d}').",
)
@optgroup.option(
    "--workers",
    type=int,
    default=8,
    help="Number of worker processes.",
)
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@optgroup.option(
    "--mesh-object-name",
    required=True,
    help="Name of the mesh object to create.",
)
@optgroup.option(
    "--mesh-user-id",
    default="from-picks",
    help="User ID for created mesh.",
)
@optgroup.option(
    "--mesh-session-id",
    default="0",
    help="Session ID for created mesh.",
)
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
    individual_meshes,
    session_id_template,
    workers,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
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
        session_id_template=session_id_template,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_vertices = sum(result.get("vertices_created", 0) for result in results.values() if result)
    total_faces = sum(result.get("faces_created", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")