"""CLI commands for segmentation processing operations."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import add_output_option


@click.command(
    context_settings={"show_default": True},
    short_help="Generate valid area box meshes for tomographic reconstructions.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input runs.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@optgroup.option(
    "--voxel-spacing",
    "-vs",
    type=float,
    required=True,
    help="Voxel spacing for the tomograms.",
)
@optgroup.option(
    "--tomo-type",
    "-tt",
    default="wbp",
    help="Type of tomogram to use as reference.",
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--angle",
    type=float,
    default=0.0,
    help="Rotation angle around Z-axis in degrees.",
)
@optgroup.option(
    "--workers",
    type=int,
    default=8,
    help="Number of worker processes.",
)
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_output_option("mesh", default_tool="validbox")
@add_debug_option
def validbox(
    config,
    run_names,
    voxel_spacing,
    tomo_type,
    angle,
    workers,
    output_uri,
    debug,
):
    """
    Generate valid area box meshes for tomographic reconstructions.

    \b
    URI Format:
        Meshes: object_name:user_id/session_id

    \b
    Creates box meshes representing the valid imaging area of tomographic
    reconstructions. The box dimensions are based on the tomogram voxel dimensions
    and can be optionally rotated around the Z-axis.

    \b
    Examples:
        # Generate validbox meshes for all runs
        copick process validbox --voxel-spacing 10.0 -o "validbox:auto/0"

        # Generate with rotation and specific tomogram type
        copick process validbox --voxel-spacing 10.0 --angle 45.0 --tomo-type "imod" -o "validbox:rotated/45deg"
    """
    from copick_utils.process.validbox import validbox_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Parse output URI
    try:
        output_params = parse_copick_uri(output_uri, "mesh")
    except ValueError as e:
        raise click.BadParameter(f"Invalid output URI: {e}") from e

    mesh_object_name_output = output_params["object_name"]
    mesh_user_id_output = output_params["user_id"]
    mesh_session_id_output = output_params["session_id"]

    logger.info(f"Generating validbox meshes for object '{mesh_object_name_output}'")
    logger.info(f"Voxel spacing: {voxel_spacing}, tomogram type: {tomo_type}")
    logger.info(f"Rotation angle: {angle} degrees")
    logger.info(f"Target mesh: {mesh_object_name_output} ({mesh_user_id_output}/{mesh_session_id_output})")

    if run_names_list:
        logger.info(f"Processing {len(run_names_list)} specific runs")
    else:
        logger.info(f"Processing all {len(root.runs)} runs")

    results = validbox_batch(
        root=root,
        voxel_spacing=voxel_spacing,
        mesh_object_name=mesh_object_name_output,
        mesh_user_id=mesh_user_id_output,
        mesh_session_id=mesh_session_id_output,
        tomo_type=tomo_type,
        angle=angle,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_vertices = sum(result.get("vertices_created", 0) for result in results.values() if result)
    total_faces = sum(result.get("faces_created", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
