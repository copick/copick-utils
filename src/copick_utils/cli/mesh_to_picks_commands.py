"""CLI commands for converting meshes to picks."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger


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