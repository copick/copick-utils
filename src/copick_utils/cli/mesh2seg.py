import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import (
    add_input_option,
    add_mesh_voxelization_options,
    add_output_option,
    add_workers_option,
)
from copick_utils.util.config_models import create_simple_config


@click.command(
    context_settings={"show_default": True},
    short_help="Convert mesh to segmentation.",
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
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_mesh_voxelization_options
@optgroup.option(
    "--tomo-type",
    "-tt",
    default="wbp",
    help="Type of tomogram to use as reference.",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="mesh2seg")
@add_debug_option
def mesh2seg(
    config,
    run_names,
    input_uri,
    mode,
    boundary_sampling_density,
    invert,
    tomo_type,
    workers,
    output_uri,
    debug,
):
    """
    Convert meshes to segmentation volumes with multiple voxelization modes.

    \b
    URI Format:
        Meshes: object_name:user_id/session_id
        Segmentations: name:user_id/session_id@voxel_spacing?multilabel=true

    \b
    Pattern Matching:
        - Exact: "membrane:user1/session1"
        - Glob: "membrane:user1/session*" or "membrane:*/manual-*"
        - Regex: "re:membrane:user\\d+/session\\d+"
        - Wildcard: "membrane" (expands to "membrane:*/*")

    \b
    Voxelization modes:
        - watertight: Fill entire interior volume using ray casting
        - boundary: Voxelize only the surface with controllable sampling density

    \b
    Additional options:
        - --invert: Fill outside instead of inside (watertight mode)
        - --boundary-sampling-density: Surface sampling density (boundary mode)

    \b
    Examples:
        # Convert mesh interior to segmentation (default)
        copick convert mesh2seg -i "membrane:user1/manual-001" -o "membrane:mesh2seg/from-mesh-001@10.0"

        # Convert mesh boundary only with high sampling density
        copick convert mesh2seg --mode boundary --boundary-sampling-density 2.0 \\
            -i "membrane:user1/manual-001" -o "membrane:mesh2seg/boundary-001@10.0"

        # Invert watertight mesh (fill outside)
        copick convert mesh2seg --invert -i "membrane:user1/manual-001" -o "membrane:mesh2seg/inverted-001@10.0"

        # Convert all manual meshes using pattern matching with multilabel output
        copick convert mesh2seg -i "membrane:user1/manual-.*" -o "membrane:mesh2seg/from-mesh-{input_session_id}@10.0?multilabel=true"
    """
    from copick_utils.converters.segmentation_from_mesh import segmentation_from_mesh_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Create config directly from URIs with smart defaults
    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="mesh",
            output_uri=output_uri,
            output_type="segmentation",
            command_name="mesh2seg",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Extract parameters for logging and processing
    input_params = parse_copick_uri(input_uri, "mesh")
    output_params = parse_copick_uri(output_uri, "segmentation")

    voxel_spacing_output = output_params["voxel_spacing"]
    if isinstance(voxel_spacing_output, str):
        voxel_spacing_output = float(voxel_spacing_output)
    multilabel_output = output_params.get("multilabel") or False

    logger.info(f"Converting mesh to segmentation for object '{input_params['object_name']}'")
    logger.info(f"Source mesh pattern: {input_params['user_id']}/{input_params['session_id']}")
    logger.info(
        f"Target segmentation template: {output_params['name']} ({output_params['user_id']}/{output_params['session_id']})",
    )
    logger.info(f"Mode: {mode}, voxel spacing: {voxel_spacing_output}, multilabel: {multilabel_output}")
    if mode == "boundary":
        logger.info(f"Boundary sampling density: {boundary_sampling_density}")
    if invert:
        logger.info("Volume inversion: enabled")

    # Parallel discovery and processing - no sequential bottleneck!
    results = segmentation_from_mesh_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        voxel_spacing=voxel_spacing_output,
        tomo_type=tomo_type,
        is_multilabel=multilabel_output,
        mode=mode,
        boundary_sampling_density=boundary_sampling_density,
        invert=invert,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_voxels = sum(result.get("voxels_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("processed", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total conversion tasks completed: {total_processed}")
    logger.info(f"Total voxels created: {total_voxels}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
