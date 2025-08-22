"""CLI commands for data conversion between different copick formats."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger


@click.command(
    context_settings={"show_default": True},
    short_help="Convert picks to segmentation.",
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
    help="Name of the object to process picks for.",
)
@optgroup.option(
    "--pick-user-id",
    required=True,
    help="User ID of the picks to convert.",
)
@optgroup.option(
    "--pick-session-id",
    required=True,
    help="Session ID pattern (regex) or exact session ID of the picks to convert.",
)
@optgroup.option(
    "--tomo-type",
    required=True,
    help="Type of tomogram to use as reference.",
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--radius",
    type=float,
    required=True,
    help="Radius of spheres in physical units (angstrom).",
)
@optgroup.option(
    "--voxel-spacing",
    type=float,
    required=True,
    help="Voxel spacing for the segmentation (angstrom/vox).",
)
@optgroup.option(
    "--workers",
    type=int,
    default=8,
    help="Number of worker processes.",
)
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@optgroup.option(
    "--seg-name",
    required=True,
    help="Name of the segmentation to create.",
)
@optgroup.option(
    "--seg-user-id",
    default="paintedPicks",
    help="User ID for the created segmentation.",
)
@optgroup.option(
    "--seg-session-id",
    default="0",
    help="Session ID for the created segmentation.",
)
@add_debug_option
def picks2seg(
    config,
    run_names,
    pick_object_name,
    pick_user_id,
    pick_session_id,
    tomo_type,
    radius,
    voxel_spacing,
    workers,
    seg_name,
    seg_user_id,
    seg_session_id,
    debug,
):
    """Convert picks to segmentation volumes."""
    from copick_utils.converters.segmentation_from_picks import segmentation_from_picks_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    logger.info(f"Converting picks to segmentation for object '{pick_object_name}'")
    logger.info(f"Source picks: {pick_user_id}/{pick_session_id}")
    logger.info(f"Target segmentation: {seg_name} ({seg_user_id}/{seg_session_id})")
    logger.info(f"Sphere radius: {radius} (voxel spacing: {voxel_spacing})")

    if run_names_list:
        logger.info(f"Processing {len(run_names_list)} specific runs")
    else:
        logger.info(f"Processing all {len(root.runs)} runs")

    results = segmentation_from_picks_batch(
        root=root,
        object_name=pick_object_name,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        radius=radius,
        painting_segmentation_name=seg_name,
        voxel_spacing=voxel_spacing,
        tomo_type=tomo_type,
        user_id=seg_user_id,
        session_id=seg_session_id,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_points = sum(result.get("points_converted", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total points converted: {total_points}")


@click.command(
    context_settings={"show_default": True},
    short_help="Convert segmentation to picks.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input segmentations.")
@optgroup.option(
    "--run-names",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@optgroup.option(
    "--object-name",
    required=True,
    help="Name of the object to process segmentations for.",
)
@optgroup.option(
    "--seg-user-id",
    required=True,
    help="User ID of the segmentations to convert.",
)
@optgroup.option(
    "--seg-session-id",
    required=True,
    help="Session ID of the segmentations to convert.",
)
@optgroup.option(
    "--segmentation-name",
    required=True,
    help="Name of the segmentation to process.",
)
@optgroup.option(
    "--segmentation-idx",
    type=int,
    required=True,
    help="Label value to extract from segmentation.",
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--maxima-filter-size",
    type=int,
    default=9,
    help="Size of maximum detection filter.",
)
@optgroup.option(
    "--min-particle-size",
    type=int,
    default=1000,
    help="Minimum particle size threshold.",
)
@optgroup.option(
    "--max-particle-size",
    type=int,
    default=50000,
    help="Maximum particle size threshold.",
)
@optgroup.option(
    "--voxel-spacing",
    type=float,
    required=True,
    help="Voxel spacing for scaling pick locations.",
)
@optgroup.option(
    "--workers",
    type=int,
    default=8,
    help="Number of worker processes.",
)
@optgroup.group("\nOutput Options", help="Options related to output picks.")
@optgroup.option(
    "--pick-user-id",
    required=True,
    help="User ID for the created picks.",
)
@optgroup.option(
    "--pick-session-id",
    required=True,
    help="Session ID for the created picks.",
)
@add_debug_option
def seg2picks(
    config,
    object_name,
    seg_user_id,
    seg_session_id,
    segmentation_name,
    segmentation_idx,
    maxima_filter_size,
    min_particle_size,
    max_particle_size,
    pick_user_id,
    pick_session_id,
    voxel_spacing,
    run_names,
    workers,
    debug,
):
    """Convert segmentation volumes to picks."""
    from copick_utils.converters.picks_from_segmentation import picks_from_segmentation_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    logger.info(f"Converting segmentation to picks for object '{object_name}'")
    logger.info(f"Source segmentation: {segmentation_name} ({seg_user_id}/{seg_session_id})")
    logger.info(f"Target picks: {pick_user_id}/{pick_session_id}")
    logger.info(f"Label {segmentation_idx}, particle size: {min_particle_size}-{max_particle_size}")

    if run_names_list:
        logger.info(f"Processing {len(run_names_list)} specific runs")
    else:
        logger.info(f"Processing all {len(root.runs)} runs")

    results = picks_from_segmentation_batch(
        root=root,
        object_name=object_name,
        seg_user_id=seg_user_id,
        seg_session_id=seg_session_id,
        segmentation_name=segmentation_name,
        segmentation_idx=segmentation_idx,
        maxima_filter_size=maxima_filter_size,
        min_particle_size=min_particle_size,
        max_particle_size=max_particle_size,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        voxel_spacing=voxel_spacing,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_points = sum(result.get("points_created", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total points created: {total_points}")


@click.command(
    context_settings={"show_default": True},
    short_help="Convert mesh to segmentation.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input meshes.")
@optgroup.option(
    "--run-names",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@optgroup.option(
    "--mesh-object-name",
    required=True,
    help="Name of the mesh object to convert.",
)
@optgroup.option(
    "--mesh-user-id",
    required=True,
    help="User ID of the mesh to convert.",
)
@optgroup.option(
    "--mesh-session-id",
    required=True,
    help="Session ID of the mesh to convert.",
)
@optgroup.option(
    "--tomo-type",
    default="wbp",
    help="Type of tomogram to use as reference.",
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--voxel-spacing",
    type=float,
    required=True,
    help="Voxel spacing for the segmentation.",
)
@optgroup.option(
    "--multilabel/--no-multilabel",
    is_flag=True,
    default=False,
    help="Create multilabel segmentation.",
)
@optgroup.option(
    "--workers",
    type=int,
    default=8,
    help="Number of worker processes.",
)
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@optgroup.option(
    "--seg-name",
    required=True,
    help="Name of the segmentation to create.",
)
@optgroup.option(
    "--seg-user-id",
    default="from-mesh",
    help="User ID for the created segmentation.",
)
@optgroup.option(
    "--seg-session-id",
    default="0",
    help="Session ID for the created segmentation.",
)
@add_debug_option
def mesh2seg(
    config,
    run_names,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    tomo_type,
    voxel_spacing,
    multilabel,
    workers,
    seg_name,
    seg_user_id,
    seg_session_id,
    debug,
):
    """Convert watertight meshes to segmentation volumes."""
    from copick_utils.converters.segmentation_from_mesh import segmentation_from_mesh_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    logger.info(f"Converting mesh to segmentation for object '{mesh_object_name}'")
    logger.info(f"Source mesh: {mesh_user_id}/{mesh_session_id}")
    logger.info(f"Target segmentation: {seg_name} ({seg_user_id}/{seg_session_id})")
    logger.info(f"Voxel spacing: {voxel_spacing}, multilabel: {multilabel}")

    if run_names_list:
        logger.info(f"Processing {len(run_names_list)} specific runs")
    else:
        logger.info(f"Processing all {len(root.runs)} runs")

    results = segmentation_from_mesh_batch(
        root=root,
        mesh_object_name=mesh_object_name,
        mesh_user_id=mesh_user_id,
        mesh_session_id=mesh_session_id,
        segmentation_name=seg_name,
        segmentation_user_id=seg_user_id,
        segmentation_session_id=seg_session_id,
        voxel_spacing=voxel_spacing,
        tomo_type=tomo_type,
        is_multilabel=multilabel,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_voxels = sum(result.get("voxels_created", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total voxels created: {total_voxels}")


@click.command(
    context_settings={"show_default": True},
    short_help="Convert segmentation to mesh.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input segmentations.")
@optgroup.option(
    "--run-names",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@optgroup.option(
    "--seg-name",
    required=True,
    help="Name of the segmentation to convert.",
)
@optgroup.option(
    "--seg-user-id",
    required=True,
    help="User ID of the segmentation to convert.",
)
@optgroup.option(
    "--seg-session-id",
    required=True,
    help="Session ID of the segmentation to convert.",
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--voxel-spacing",
    type=float,
    required=True,
    help="Voxel spacing of the segmentation.",
)
@optgroup.option(
    "--multilabel/--no-multilabel",
    is_flag=True,
    default=False,
    help="Source is multilabel segmentation.",
)
@optgroup.option(
    "--level",
    type=float,
    default=0.5,
    help="Isosurface level for marching cubes.",
)
@optgroup.option(
    "--step-size",
    type=int,
    default=1,
    help="Step size for marching cubes (higher = coarser mesh).",
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
    default="from-seg",
    help="User ID for the created mesh.",
)
@optgroup.option(
    "--mesh-session-id",
    default="0",
    help="Session ID for the created mesh.",
)
@add_debug_option
def seg2mesh(
    config,
    run_names,
    seg_name,
    seg_user_id,
    seg_session_id,
    voxel_spacing,
    multilabel,
    level,
    step_size,
    workers,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    debug,
):
    """Convert segmentation volumes to meshes using marching cubes."""
    from copick_utils.converters.mesh_from_segmentation import mesh_from_segmentation_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    logger.info(f"Converting segmentation to mesh for '{seg_name}'")
    logger.info(f"Source segmentation: {seg_user_id}/{seg_session_id}")
    logger.info(f"Target mesh: {mesh_object_name} ({mesh_user_id}/{mesh_session_id})")
    logger.info(f"Marching cubes level: {level}, step size: {step_size}")

    if run_names_list:
        logger.info(f"Processing {len(run_names_list)} specific runs")
    else:
        logger.info(f"Processing all {len(root.runs)} runs")

    results = mesh_from_segmentation_batch(
        root=root,
        segmentation_name=seg_name,
        segmentation_user_id=seg_user_id,
        segmentation_session_id=seg_session_id,
        mesh_object_name=mesh_object_name,
        mesh_user_id=mesh_user_id,
        mesh_session_id=mesh_session_id,
        voxel_spacing=voxel_spacing,
        is_multilabel=multilabel,
        level=level,
        step_size=step_size,
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
    short_help="Convert mesh to picks.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input meshes.")
@optgroup.option(
    "--run-names",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@optgroup.option(
    "--mesh-object-name",
    required=True,
    help="Name of the mesh object to sample from.",
)
@optgroup.option(
    "--mesh-user-id",
    required=True,
    help="User ID of the mesh to convert.",
)
@optgroup.option(
    "--mesh-session-id",
    required=True,
    help="Session ID of the mesh to convert.",
)
@optgroup.option(
    "--tomo-type",
    default="wbp",
    help="Type of tomogram to use as reference.",
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--sampling-type",
    type=click.Choice(["inside", "surface", "outside", "vertices"]),
    required=True,
    help="Type of sampling: inside (points inside mesh), surface (points on mesh surface), outside (points outside mesh), vertices (return mesh vertices).",
)
@optgroup.option(
    "--n-points",
    type=int,
    default=1000,
    help="Number of points to sample (ignored for 'vertices' type).",
)
@optgroup.option(
    "--voxel-spacing",
    type=float,
    required=True,
    help="Voxel spacing for coordinate scaling.",
)
@optgroup.option(
    "--min-dist",
    type=float,
    help="Minimum distance between points (default: 2 * voxel_spacing).",
)
@optgroup.option(
    "--edge-dist",
    type=float,
    default=32.0,
    help="Distance from volume edges in voxels.",
)
@optgroup.option(
    "--include-normals/--no-include-normals",
    is_flag=True,
    default=False,
    help="Include surface normals as orientations (surface sampling only).",
)
@optgroup.option(
    "--random-orientations/--no-random-orientations",
    is_flag=True,
    default=False,
    help="Generate random orientations for points.",
)
@optgroup.option(
    "--seed",
    type=int,
    help="Random seed for reproducible results.",
)
@optgroup.option(
    "--workers",
    type=int,
    default=8,
    help="Number of worker processes.",
)
@optgroup.group("\nOutput Options", help="Options related to output picks.")
@optgroup.option(
    "--pick-object-name",
    required=True,
    help="Name of the object for created picks.",
)
@optgroup.option(
    "--pick-user-id",
    default="from-mesh",
    help="User ID for created picks.",
)
@optgroup.option(
    "--pick-session-id",
    default="0",
    help="Session ID for created picks.",
)
@add_debug_option
def mesh2picks(
    config,
    run_names,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    tomo_type,
    sampling_type,
    n_points,
    voxel_spacing,
    min_dist,
    edge_dist,
    include_normals,
    random_orientations,
    seed,
    workers,
    pick_object_name,
    pick_user_id,
    pick_session_id,
    debug,
):
    """Convert meshes to picks using different sampling strategies."""
    from copick_utils.converters.picks_from_mesh import picks_from_mesh_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    logger.info(f"Converting mesh to picks for object '{mesh_object_name}'")
    logger.info(f"Source mesh: {mesh_user_id}/{mesh_session_id}")
    logger.info(f"Target picks: {pick_object_name} ({pick_user_id}/{pick_session_id})")
    logger.info(f"Sampling type: {sampling_type}, n_points: {n_points}")

    if run_names_list:
        logger.info(f"Processing {len(run_names_list)} specific runs")
    else:
        logger.info(f"Processing all {len(root.runs)} runs")

    results = picks_from_mesh_batch(
        root=root,
        mesh_object_name=mesh_object_name,
        mesh_user_id=mesh_user_id,
        mesh_session_id=mesh_session_id,
        sampling_type=sampling_type,
        n_points=n_points,
        pick_object_name=pick_object_name,
        pick_session_id=pick_session_id,
        pick_user_id=pick_user_id,
        voxel_spacing=voxel_spacing,
        tomo_type=tomo_type,
        min_dist=min_dist,
        edge_dist=edge_dist,
        include_normals=include_normals,
        random_orientations=random_orientations,
        seed=seed,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_points = sum(result.get("points_created", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total points created: {total_points}")


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
    help="Cluster points before spheroid fitting.",
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
