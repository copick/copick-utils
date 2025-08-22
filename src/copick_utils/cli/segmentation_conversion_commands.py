"""CLI commands for segmentation conversion operations."""

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