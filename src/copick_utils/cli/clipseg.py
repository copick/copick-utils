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
    short_help="Limit segmentations to voxels within distance of a reference surface.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input segmentations.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_input_option("segmentation")
@optgroup.group("\nReference Options", help="Options for reference surface (provide either mesh or segmentation).")
@add_reference_mesh_option(required=False)
@add_reference_seg_option(required=False)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_distance_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="clipseg")
@add_debug_option
def clipseg(
    config,
    run_names,
    input_uri,
    ref_mesh_uri,
    ref_seg_uri,
    max_distance,
    mesh_voxel_spacing,
    workers,
    output_uri,
    debug,
):
    """
    Limit segmentations to voxels within a certain distance of a reference surface.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing
        Meshes: object_name:user_id/session_id

    \b
    The reference surface can be either a mesh or another segmentation.
    Only segmentation voxels within the specified distance will be kept.

    \b
    Examples:
        # Limit segmentation to voxels near reference mesh
        copick logical clipseg -i "membrane:user1/full-001@10.0" -rm "boundary:user1/boundary-001" -o "membrane:clipseg/limited-001@10.0" --max-distance 50.0

        # Limit using another segmentation as reference
        copick logical clipseg -i "membrane:user1/full-001@10.0" -rs "mask:user1/mask-001@10.0" -o "membrane:clipseg/limited-001@10.0" --max-distance 100.0
    """
    from copick_utils.logical.distance_operations import limit_segmentation_by_distance_lazy_batch

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

    # Extract voxel_spacing and multilabel from output for additional_params
    output_params_temp = parse_copick_uri(output_uri, "segmentation")
    voxel_spacing_output = output_params_temp["voxel_spacing"]
    if isinstance(voxel_spacing_output, str):
        voxel_spacing_output = float(voxel_spacing_output)
    multilabel_output = output_params_temp.get("multilabel") or False

    # Create config directly from URIs with additional params and smart defaults
    try:
        task_config = create_reference_config(
            input_uri=input_uri,
            input_type="segmentation",
            output_uri=output_uri,
            output_type="segmentation",
            reference_uri=reference_uri,
            reference_type=reference_type,
            additional_params={
                "max_distance": max_distance,
                "mesh_voxel_spacing": mesh_voxel_spacing,
                "voxel_spacing": voxel_spacing_output,
                "is_multilabel": multilabel_output,
            },
            command_name="clipseg",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Extract parameters for logging
    input_params = parse_copick_uri(input_uri, "segmentation")
    output_params = parse_copick_uri(output_uri, "segmentation")
    ref_params = parse_copick_uri(reference_uri, reference_type)

    logger.info(f"Limiting segmentations by distance for '{input_params['name']}'")
    logger.info(f"Source segmentation pattern: {input_params['user_id']}/{input_params['session_id']}")
    if reference_type == "mesh":
        logger.info(f"Reference mesh: {ref_params['object_name']} ({ref_params['user_id']}/{ref_params['session_id']})")
    else:
        logger.info(
            f"Reference segmentation: {ref_params['name']} ({ref_params['user_id']}/{ref_params['session_id']})",
        )
    logger.info(f"Maximum distance: {max_distance} angstroms")
    logger.info(
        f"Target segmentation template: {output_params['name']} ({output_params['user_id']}/{output_params['session_id']})",
    )

    # Parallel discovery and processing - no sequential bottleneck!
    results = limit_segmentation_by_distance_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
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
    logger.info(f"Total distance limiting operations completed: {total_processed}")
    logger.info(f"Total voxels created: {total_voxels}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
