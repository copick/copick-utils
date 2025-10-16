"""CLI command for filtering connected components by size."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import add_input_option, add_output_option, add_workers_option


@click.command(
    context_settings={"show_default": True},
    short_help="Filter connected components in segmentations by size.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input segmentation.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_input_option("segmentation")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--connectivity",
    "-cn",
    type=click.Choice(["face", "face-edge", "all"]),
    default="all",
    help="Connectivity for connected components (face=6-connected, face-edge=18-connected, all=26-connected).",
)
@optgroup.option(
    "--min-size",
    type=float,
    default=None,
    help="Minimum component volume in cubic angstroms (Å³) to keep (optional).",
)
@optgroup.option(
    "--max-size",
    type=float,
    default=None,
    help="Maximum component volume in cubic angstroms (Å³) to keep (optional).",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="filter-components")
@add_debug_option
def filter_components(
    config,
    run_names,
    input_uri,
    connectivity,
    min_size,
    max_size,
    workers,
    output_uri,
    debug,
):
    """
    Filter connected components in segmentations by size.

    This command identifies connected components in a segmentation and removes those
    that fall outside the specified size range (in cubic angstroms). Useful for
    removing noise, small artifacts, or overly large components.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing

    \b
    Examples:
        # Remove small noise components (keep only larger than 50000 Å³)
        copick process filter-components -i "membrane:user1/auto-001@10.0" -o "membrane_clean" --min-size 50000

        # Keep only medium-sized components (between 10000 and 1000000 Å³)
        copick process filter-components -i "particles:user1/.*@10.0" -o "particles_filtered" --min-size 10000 --max-size 1000000

        # Remove large components (keep only smaller than 500000 Å³)
        copick process filter-components -i "noise:user1/pred@10.0" -o "small_features" --max-size 500000
    """

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Parse input URI
    try:
        input_params = parse_copick_uri(input_uri, "segmentation")
    except ValueError as e:
        raise click.BadParameter(f"Invalid input URI: {e}") from e

    segmentation_name = input_params["name"]
    segmentation_user_id = input_params["user_id"]
    segmentation_session_id = input_params["session_id"]
    voxel_spacing = input_params.get("voxel_spacing")

    if voxel_spacing is None:
        raise click.BadParameter("Input URI must include voxel spacing (e.g., @10.0)")

    # Parse output URI - if no voxel spacing specified, inherit from input
    if "@" not in output_uri:
        output_uri = f"{output_uri}@{voxel_spacing}"

    try:
        output_params = parse_copick_uri(output_uri, "segmentation")
    except ValueError as e:
        raise click.BadParameter(f"Invalid output URI: {e}") from e

    output_name = output_params["name"]
    output_user_id = output_params["user_id"]
    output_session_id = output_params["session_id"]

    logger.info(f"Filtering components for segmentation '{segmentation_name}'")
    logger.info(f"Input segmentation: {segmentation_user_id}/{segmentation_session_id} @ {voxel_spacing}Å")
    logger.info(f"Output segmentation: {output_name} ({output_user_id}/{output_session_id})")
    logger.info(f"Connectivity: {connectivity}")
    if min_size is not None:
        logger.info(f"Minimum size: {min_size} Å³")
    if max_size is not None:
        logger.info(f"Maximum size: {max_size} Å³")

    # Import batch function
    from copick_utils.process.filter_components import filter_components_batch

    # Process runs
    results = filter_components_batch(
        root=root,
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        segmentation_session_id=segmentation_session_id,
        voxel_spacing=voxel_spacing,
        connectivity=connectivity,
        min_size=min_size,
        max_size=max_size,
        output_user_id=output_user_id,
        output_session_id=output_session_id,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_kept = sum(result.get("components_kept", 0) for result in results.values() if result)
    total_removed = sum(result.get("components_removed", 0) for result in results.values() if result)
    total_voxels = sum(result.get("voxels_kept", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total components kept: {total_kept}")
    logger.info(f"Total components removed: {total_removed}")
    logger.info(f"Total voxels in filtered segmentations: {total_voxels}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
