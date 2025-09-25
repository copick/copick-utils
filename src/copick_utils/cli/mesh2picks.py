"""CLI commands for converting meshes to picks."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from copick_utils.cli.input_output_selection import validate_conversion_placeholders
from copick_utils.cli.util import add_mesh_input_options, add_picks_output_options, add_workers_option
from copick_utils.converters.config_models import SelectorConfig, TaskConfig
from copick_utils.converters.picks_from_mesh import picks_from_mesh_lazy_batch


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
@add_mesh_input_options
@optgroup.option(
    "--tomo-type",
    "-tt",
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
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output picks.")
@add_picks_output_options
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
    """
    Convert meshes to picks using different sampling strategies.

    \b
    Supports flexible input/output selection modes:
    - One-to-one: exact session ID → exact session ID
    - Many-to-many: regex pattern → template with {input_session_id}

    \b
    Examples:
        # Convert single mesh to picks
        copick convert mesh2picks --mesh-session-id "boundary-001" --pick-session-id "sampled-001" --sampling-type surface
        \b
        # Convert all boundary meshes using pattern matching
        copick convert mesh2picks --mesh-session-id "boundary-.*" --pick-session-id "sampled-{input_session_id}" --sampling-type inside
    """
    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Validate placeholder requirements
    try:
        validate_conversion_placeholders(mesh_session_id, pick_session_id, individual_outputs=False)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    logger.info(f"Converting mesh to picks for object '{mesh_object_name}'")
    logger.info(f"Source mesh pattern: {mesh_user_id}/{mesh_session_id}")
    logger.info(f"Target picks template: {pick_object_name} ({pick_user_id}/{pick_session_id})")
    logger.info(f"Sampling type: {sampling_type}, n_points: {n_points}")

    # Create type-safe Pydantic configuration
    selector_config = SelectorConfig(
        input_type="mesh",
        output_type="picks",
        input_object_name=mesh_object_name,
        input_user_id=mesh_user_id,
        input_session_id=mesh_session_id,
        output_object_name=pick_object_name,
        output_user_id=pick_user_id,
        output_session_id=pick_session_id,
    )

    config = TaskConfig(
        type="single_selector",
        selector=selector_config,
    )

    # Parallel discovery and processing with consistent architecture!
    results = picks_from_mesh_lazy_batch(
        root=root,
        config=config,
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
