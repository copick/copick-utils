import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from copick_utils.cli.util import (
    add_distance_options,
    add_mesh_input_options,
    add_mesh_output_options,
    add_segmentation_input_options,
    add_workers_option,
)
from copick_utils.converters.config_models import ReferenceConfig, SelectorConfig, TaskConfig
from copick_utils.logical.distance_operations import limit_mesh_by_distance_lazy_batch, validate_conversion_placeholders


@click.command(
    context_settings={"show_default": True},
    short_help="Limit meshes to vertices within distance of a reference surface.",
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
@optgroup.group("\nReference Options", help="Options for reference surface (provide either mesh or segmentation).")
@add_mesh_input_options(prefix="ref", required=False)
@add_segmentation_input_options(
    prefix="ref",
    include_multilabel=False,
    required=False,
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_distance_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_mesh_output_options(default_tool="clipmesh")
@add_debug_option
def clipmesh(
    config,
    run_names,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    ref_mesh_object_name,
    ref_mesh_user_id,
    ref_mesh_session_id,
    ref_seg_name,
    ref_seg_user_id,
    ref_seg_session_id,
    ref_voxel_spacing,
    max_distance,
    mesh_voxel_spacing,
    workers,
    mesh_object_name_output,
    mesh_user_id_output,
    mesh_session_id_output,
    individual_meshes,
    debug,
):
    """
    Limit meshes to vertices within a certain distance of a reference surface.

    \b
    The reference surface can be either a mesh or a segmentation.
    Only mesh vertices within the specified distance will be kept.

    \b
    Examples:
        # Limit mesh to vertices near reference mesh surface
        copick clipmesh --mesh-session-id "full-001" --ref-mesh-session-id "boundary-001" --max-distance 50.0 --mesh-session-id "limited-001"
        \b
        # Limit using segmentation as reference
        copick clipmesh --mesh-session-id "full-001" --ref-seg-session-id "mask-001" --ref-voxel-spacing 10.0 --max-distance 100.0 --mesh-session-id "limited-001"
    """

    logger = get_logger(__name__, debug=debug)

    # Validate that exactly one reference type is provided
    ref_mesh_provided = bool(ref_mesh_object_name or ref_mesh_user_id or ref_mesh_session_id)
    ref_seg_provided = bool(ref_seg_name or ref_seg_user_id or ref_seg_session_id)

    if not ref_mesh_provided and not ref_seg_provided:
        raise click.BadParameter("Must provide either reference mesh or reference segmentation")
    if ref_mesh_provided and ref_seg_provided:
        raise click.BadParameter("Cannot provide both reference mesh and reference segmentation")

    if ref_seg_provided and not ref_voxel_spacing:
        raise click.BadParameter("--ref-voxel-spacing is required when using reference segmentation")

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Validate placeholder requirements
    try:
        validate_conversion_placeholders(mesh_session_id, mesh_session_id_output, individual_meshes)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    logger.info(f"Limiting meshes by distance for object '{mesh_object_name}'")
    logger.info(f"Source mesh pattern: {mesh_user_id}/{mesh_session_id}")
    if ref_mesh_provided:
        logger.info(f"Reference mesh: {ref_mesh_object_name} ({ref_mesh_user_id}/{ref_mesh_session_id})")
    else:
        logger.info(f"Reference segmentation: {ref_seg_name} ({ref_seg_user_id}/{ref_seg_session_id})")
    logger.info(f"Maximum distance: {max_distance} angstroms")
    logger.info(
        f"Target mesh template: {mesh_object_name_output or mesh_object_name} ({mesh_user_id_output}/{mesh_session_id_output})",
    )

    # Create type-safe Pydantic configuration
    selector_config = SelectorConfig(
        input_type="mesh",
        output_type="mesh",
        input_object_name=mesh_object_name,
        input_user_id=mesh_user_id,
        input_session_id=mesh_session_id,
        output_object_name=mesh_object_name_output,
        output_user_id=mesh_user_id_output,
        output_session_id=mesh_session_id_output,
    )

    # Create reference configuration
    if ref_mesh_provided:
        reference_config = ReferenceConfig(
            reference_type="mesh",
            object_name=ref_mesh_object_name,
            user_id=ref_mesh_user_id,
            session_id=ref_mesh_session_id,
            additional_params={
                "max_distance": max_distance,
                "mesh_voxel_spacing": mesh_voxel_spacing,
            },
        )
    else:
        reference_config = ReferenceConfig(
            reference_type="segmentation",
            object_name=ref_seg_name,
            user_id=ref_seg_user_id,
            session_id=ref_seg_session_id,
            voxel_spacing=ref_voxel_spacing,
            additional_params={
                "max_distance": max_distance,
                "mesh_voxel_spacing": mesh_voxel_spacing,
            },
        )

    config = TaskConfig(
        type="single_selector_with_reference",
        selector=selector_config,
        reference=reference_config,
    )

    # Parallel discovery and processing - no sequential bottleneck!
    results = limit_mesh_by_distance_lazy_batch(
        root=root,
        config=config,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_vertices = sum(result.get("vertices_created", 0) for result in results.values() if result)
    total_faces = sum(result.get("faces_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("processed", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total distance limiting operations completed: {total_processed}")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
