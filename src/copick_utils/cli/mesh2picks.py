"""CLI commands for converting meshes to picks."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import add_input_option, add_output_option, add_tomogram_option, add_workers_option
from copick_utils.util.config_models import create_simple_config


@click.command(
    context_settings={"show_default": True},
    short_help="Convert mesh to picks.",
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
@add_input_option("mesh")
@add_tomogram_option(required=True)
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
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output picks.")
@add_output_option("picks", default_tool="mesh2picks")
@add_debug_option
def mesh2picks(
    config,
    run_names,
    input_uri,
    tomogram_uri,
    sampling_type,
    n_points,
    min_dist,
    edge_dist,
    include_normals,
    random_orientations,
    seed,
    workers,
    output_uri,
    debug,
):
    """
    Convert meshes to picks using different sampling strategies.

    \b
    URI Format:
        Meshes: object_name:user_id/session_id
        Picks: object_name:user_id/session_id
        Tomograms: tomo_type@voxel_spacing

    \b
    Examples:
        # Convert single mesh to picks with surface sampling
        copick convert mesh2picks -i "boundary:user1/boundary-001" --tomogram wbp@10.0 --sampling-type surface -o "boundary"

        # Convert all boundary meshes using pattern matching
        copick convert mesh2picks -i "boundary:user1/boundary-.*" -t wbp@10.0 --sampling-type inside -o "{input_session_id}"
    """
    from copick_utils.converters.picks_from_mesh import picks_from_mesh_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Parse tomogram URI to extract tomo_type and voxel_spacing
    try:
        tomogram_params = parse_copick_uri(tomogram_uri, "tomogram")
    except ValueError as e:
        raise click.BadParameter(f"Invalid tomogram URI: {e}") from e

    tomo_type = tomogram_params["tomo_type"]
    voxel_spacing = tomogram_params["voxel_spacing"]
    if isinstance(voxel_spacing, str):
        voxel_spacing = float(voxel_spacing)

    # Create config directly from URIs with smart defaults
    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="mesh",
            output_uri=output_uri,
            output_type="picks",
            command_name="mesh2picks",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Extract parameters for logging
    input_params = parse_copick_uri(input_uri, "mesh")
    output_params = parse_copick_uri(output_uri, "picks")

    logger.info(f"Converting mesh to picks for object '{input_params['object_name']}'")
    logger.info(f"Source mesh pattern: {input_params['user_id']}/{input_params['session_id']}")
    logger.info(
        f"Target picks template: {output_params['object_name']} ({output_params['user_id']}/{output_params['session_id']})",
    )
    logger.info(f"Tomogram: {tomo_type}@{voxel_spacing}")
    logger.info(f"Sampling type: {sampling_type}, n_points: {n_points}")

    # Parallel discovery and processing with consistent architecture!
    results = picks_from_mesh_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        sampling_type=sampling_type,
        n_points=n_points,
        voxel_spacing=voxel_spacing,
        tomo_type=tomo_type,
        min_dist=min_dist,
        edge_dist=edge_dist,
        include_normals=include_normals,
        random_orientations=random_orientations,
        seed=seed,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_points = sum(result.get("points_created", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total points created: {total_points}")
