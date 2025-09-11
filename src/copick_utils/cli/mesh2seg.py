import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from copick_utils.cli.input_output_selection import ConversionSelector, validate_conversion_placeholders
from copick_utils.cli.util import (
    add_mesh_input_options,
    add_mesh_voxelization_options,
    add_segmentation_output_options,
    add_workers_option,
)


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
@add_mesh_input_options
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
@add_segmentation_output_options(default_tool="mesh2seg", include_tomo_type=False)
@add_debug_option
def mesh2seg(
    config,
    run_names,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    mode,
    boundary_sampling_density,
    invert,
    tomo_type,
    workers,
    seg_name,
    seg_user_id,
    seg_session_id,
    voxel_spacing,
    multilabel,
    debug,
):
    """
    Convert meshes to segmentation volumes with multiple voxelization modes.

    \b
    Voxelization modes:
    - watertight: Fill entire interior volume using ray casting
    - boundary: Voxelize only the surface with controllable sampling density

    \b
    Additional options:
    - --invert: Fill outside instead of inside (watertight mode)
    - --boundary-sampling-density: Surface sampling density (boundary mode)

    \b
    Supports flexible input/output selection modes:
    - One-to-one: exact session ID → exact session ID
    - Many-to-many: regex pattern → template with {input_session_id}

    \b
    Examples:
        # Convert mesh interior to segmentation (default)
        copick convert mesh2seg --mesh-session-id "manual-001" --seg-session-id "from-mesh-001"
        \b
        # Convert mesh boundary only with high sampling density
        copick convert mesh2seg --mode boundary --boundary-sampling-density 2.0 --mesh-session-id "manual-001" --seg-session-id "boundary-001"
        \b
        # Invert watertight mesh (fill outside)
        copick convert mesh2seg --invert --mesh-session-id "manual-001" --seg-session-id "inverted-001"
        \b
        # Convert all manual meshes using pattern matching
        copick convert mesh2seg --mesh-session-id "manual-.*" --seg-session-id "from-mesh-{input_session_id}"
    """
    from copick_utils.converters.segmentation_from_mesh import segmentation_from_mesh_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Validate placeholder requirements
    try:
        validate_conversion_placeholders(mesh_session_id, seg_session_id, individual_outputs=False)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Create conversion selector
    selector = ConversionSelector(
        input_type="mesh",
        output_type="segmentation",
        input_object_name=mesh_object_name,
        input_user_id=mesh_user_id,
        input_session_id=mesh_session_id,
        output_object_name=seg_name,
        output_user_id=seg_user_id,
        output_session_id=seg_session_id,
        voxel_spacing=voxel_spacing,
    )

    logger.info(f"Converting mesh to segmentation for object '{mesh_object_name}'")
    logger.info(f"Selection mode: {selector.get_mode_description()}")
    logger.info(f"Source mesh pattern: {mesh_user_id}/{mesh_session_id}")
    logger.info(f"Target segmentation template: {seg_name} ({seg_user_id}/{seg_session_id})")
    logger.info(f"Mode: {mode}, voxel spacing: {voxel_spacing}, multilabel: {multilabel}")
    if mode == "boundary":
        logger.info(f"Boundary sampling density: {boundary_sampling_density}")
    if invert:
        logger.info("Volume inversion: enabled")

    # Collect all conversion tasks across runs
    all_tasks = []
    runs_to_process = root.runs if run_names_list is None else [root.get_run(name) for name in run_names_list]

    for run in runs_to_process:
        tasks = selector.get_conversion_tasks(run)
        all_tasks.extend(tasks)

    if not all_tasks:
        logger.warning("No matching meshes found for conversion")
        return

    logger.info(f"Found {len(all_tasks)} conversion tasks across {len(runs_to_process)} runs")

    results = segmentation_from_mesh_batch(
        root=root,
        conversion_tasks=all_tasks,
        run_names=run_names_list,
        workers=workers,
        voxel_spacing=voxel_spacing,
        tomo_type=tomo_type,
        is_multilabel=multilabel,
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
