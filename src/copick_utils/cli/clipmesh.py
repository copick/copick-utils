import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import (
    add_distance_options,
    add_input_option,
    add_output_option,
    add_reference_mesh_option,
    add_reference_seg_option,
    add_workers_option,
)
from copick_utils.util.config_models import create_reference_config


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
@add_input_option("mesh")
@optgroup.group("\nReference Options", help="Options for reference surface (provide either mesh or segmentation).")
@add_reference_mesh_option(required=False)
@add_reference_seg_option(required=False)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_distance_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_output_option("mesh", default_tool="clipmesh")
@optgroup.option(
    "--individual-meshes/--no-individual-meshes",
    "-im",
    is_flag=True,
    default=False,
    help="Create individual meshes for each instance (enables {instance_id} placeholder).",
)
@add_debug_option
def clipmesh(
    config,
    run_names,
    input_uri,
    ref_mesh_uri,
    ref_seg_uri,
    max_distance,
    mesh_voxel_spacing,
    workers,
    output_uri,
    individual_meshes,
    debug,
):
    """
    Limit meshes to vertices within a certain distance of a reference surface.

    \b
    URI Format:
        Meshes: object_name:user_id/session_id
        Segmentations: name:user_id/session_id@voxel_spacing

    \b
    The reference surface can be either a mesh or a segmentation.
    Only mesh vertices within the specified distance will be kept.

    \b
    Examples:
        # Limit mesh to vertices near reference mesh surface
        copick logical clipmesh -i "membrane:user1/full-001" -rm "boundary:user1/boundary-001" -o "membrane:clipmesh/limited-001" --max-distance 50.0

        # Limit using segmentation as reference
        copick logical clipmesh -i "membrane:user1/full-001" -rs "mask:user1/mask-001@10.0" -o "membrane:clipmesh/limited-001" --max-distance 100.0
    """
    from copick_utils.logical.distance_operations import limit_mesh_by_distance_lazy_batch

    logger = get_logger(__name__, debug=debug)

    # Validate that exactly one reference type is provided
    if not ref_mesh_uri and not ref_seg_uri:
        raise click.BadParameter("Must provide either --ref-mesh or --ref-seg")
    if ref_mesh_uri and ref_seg_uri:
        raise click.BadParameter("Cannot provide both --ref-mesh and --ref-seg")

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Determine reference type and URI
    reference_uri = ref_mesh_uri or ref_seg_uri
    reference_type = "mesh" if ref_mesh_uri else "segmentation"

    # Create config directly from URIs with additional params for distance operations
    try:
        task_config = create_reference_config(
            input_uri=input_uri,
            input_type="mesh",
            output_uri=output_uri,
            output_type="mesh",
            reference_uri=reference_uri,
            reference_type=reference_type,
            additional_params={"max_distance": max_distance, "mesh_voxel_spacing": mesh_voxel_spacing},
            command_name="clipmesh",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Extract parameters for logging
    input_params = parse_copick_uri(input_uri, "mesh")
    output_params = parse_copick_uri(output_uri, "mesh")
    ref_params = parse_copick_uri(reference_uri, reference_type)

    logger.info(f"Limiting meshes by distance for object '{input_params['object_name']}'")
    logger.info(f"Source mesh pattern: {input_params['user_id']}/{input_params['session_id']}")
    if reference_type == "mesh":
        logger.info(f"Reference mesh: {ref_params['object_name']} ({ref_params['user_id']}/{ref_params['session_id']})")
    else:
        logger.info(
            f"Reference segmentation: {ref_params['name']} ({ref_params['user_id']}/{ref_params['session_id']})",
        )
    logger.info(f"Maximum distance: {max_distance} angstroms")
    logger.info(
        f"Target mesh template: {output_params['object_name']} ({output_params['user_id']}/{output_params['session_id']})",
    )

    # Parallel discovery and processing - no sequential bottleneck!
    results = limit_mesh_by_distance_lazy_batch(
        root=root,
        config=task_config,
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
