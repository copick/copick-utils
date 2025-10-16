"""CLI commands for point filtering operations (inclusion/exclusion by volume)."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import (
    add_input_option,
    add_output_option,
    add_reference_mesh_option,
    add_reference_seg_option,
    add_workers_option,
)
from copick_utils.util.config_models import create_reference_config


@click.command(
    context_settings={"show_default": True},
    short_help="Filter picks to exclude those inside a reference volume.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input picks.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_input_option("picks")
@optgroup.group("\nReference Options", help="Options for reference volume (provide either mesh or segmentation).")
@add_reference_mesh_option(required=False)
@add_reference_seg_option(required=False)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output picks.")
@add_output_option("picks", default_tool="picksout")
@add_debug_option
def picksout(
    config,
    run_names,
    input_uri,
    ref_mesh_uri,
    ref_seg_uri,
    workers,
    output_uri,
    debug,
):
    """
    Filter picks to exclude those inside a reference volume.

    \b
    URI Format:
        Picks: object_name:user_id/session_id
        Meshes: object_name:user_id/session_id
        Segmentations: name:user_id/session_id@voxel_spacing

    \b
    The reference volume can be either a watertight mesh or a segmentation.
    Picks that fall inside the reference volume will be removed.

    \b
    Examples:
        # Exclude picks inside reference mesh
        copick logical picksout -i "ribosome:user1/all-001" -rm "boundary:user1/boundary-001" -o "ribosome:picksout/outside-001"

        # Exclude picks inside segmentation
        copick logical picksout -i "ribosome:user1/all-001" -rs "mask:user1/mask-001@10.0" -o "ribosome:picksout/outside-001"
    """
    from copick_utils.logical.point_operations import picks_exclusion_by_mesh_lazy_batch

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

    # Create config directly from URIs with smart defaults
    try:
        task_config = create_reference_config(
            input_uri=input_uri,
            input_type="picks",
            output_uri=output_uri,
            output_type="picks",
            reference_uri=reference_uri,
            reference_type=reference_type,
            command_name="picksout",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Extract parameters for logging
    input_params = parse_copick_uri(input_uri, "picks")
    output_params = parse_copick_uri(output_uri, "picks")
    ref_params = parse_copick_uri(reference_uri, reference_type)

    logger.info(f"Excluding picks inside reference volume for object '{input_params['object_name']}'")
    logger.info(f"Source picks pattern: {input_params['user_id']}/{input_params['session_id']}")
    if reference_type == "mesh":
        logger.info(f"Reference mesh: {ref_params['object_name']} ({ref_params['user_id']}/{ref_params['session_id']})")
    else:
        logger.info(
            f"Reference segmentation: {ref_params['name']} ({ref_params['user_id']}/{ref_params['session_id']})",
        )
    logger.info(
        f"Target picks template: {output_params['object_name']} ({output_params['user_id']}/{output_params['session_id']})",
    )

    # Parallel discovery and processing - no sequential bottleneck!
    results = picks_exclusion_by_mesh_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_points = sum(result.get("points_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("processed", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total exclusion operations completed: {total_processed}")
    logger.info(f"Total points excluded (remaining): {total_points}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
