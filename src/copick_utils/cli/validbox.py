"""CLI commands for segmentation processing operations."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import expand_output_uri, parse_copick_uri

from copick_utils.cli.util import add_output_option, add_tomogram_option


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
@add_tomogram_option(required=True)
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
    tomogram_uri,
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
        Tomograms: tomo_type@voxel_spacing

    \b
    Creates box meshes representing the valid imaging area of tomographic
    reconstructions. The box dimensions are based on the tomogram voxel dimensions
    and can be optionally rotated around the Z-axis.

    \b
    Examples:
        # Generate validbox meshes for all runs
        copick process validbox --tomogram wbp@10.0 -o "validbox"

        # Generate with rotation and specific tomogram type
        copick process validbox -t imod@10.0 --angle 45.0 -o "validbox/rotated"
    """
    from copick_utils.process.validbox import validbox_batch

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

    # Expand output URI with smart defaults (no input, so use synthetic input)
    try:
        output_uri = expand_output_uri(
            output_uri=output_uri,
            input_uri="validbox:*/*",  # Synthetic input for default object name
            input_type="mesh",
            output_type="mesh",
            command_name="validbox",
            individual_outputs=False,
        )
    except ValueError as e:
        raise click.BadParameter(f"Error expanding output URI: {e}") from e

    # Parse output URI (now fully expanded)
    try:
        output_params = parse_copick_uri(output_uri, "mesh")
    except ValueError as e:
        raise click.BadParameter(f"Invalid output URI: {e}") from e

    mesh_object_name_output = output_params["object_name"]
    mesh_user_id_output = output_params["user_id"]
    mesh_session_id_output = output_params["session_id"]

    logger.info(f"Generating validbox meshes for object '{mesh_object_name_output}'")
    logger.info(f"Tomogram: {tomo_type}@{voxel_spacing}")
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
