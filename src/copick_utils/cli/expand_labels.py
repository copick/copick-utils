"""CLI command for expanding labels in segmentations."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import add_input_option, add_output_option, add_workers_option


@click.command(
    context_settings={"show_default": True},
    short_help="Expand labels in segmentations to fill holes and gaps.",
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
    "--distance",
    "-d",
    type=float,
    required=True,
    help="Distance in angstroms (Å) by which to expand labels.",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="expand-labels")
@add_debug_option
def expand_labels(
    config,
    run_names,
    input_uri,
    distance,
    workers,
    output_uri,
    debug,
):
    """
    Expand labels in segmentations to fill holes and gaps.

    Uses scikit-image's expand_labels to grow label regions outward by a specified
    distance without overlapping into neighboring regions. Useful for filling small
    holes in segmentations or closing gaps between label boundaries.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing

    \b
    Examples:
        # Expand labels by 20 angstroms
        copick process expand-labels -i "membrane:user1/auto-001@10.0" -o "membrane_filled" --distance 20.0

        # Expand labels across specific runs
        copick process expand-labels -i "organelle:user1/pred@10.0" -o "organelle_expanded" --distance 15.0 -r run1 -r run2

        # Expand with custom output URI
        copick process expand-labels -i "membrane:user1/manual@10.0" -o "membrane:expand-labels/0@10.0" --distance 30.0
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

    output_user_id = output_params["user_id"]
    output_session_id = output_params["session_id"]

    logger.info(f"Expanding labels for segmentation '{segmentation_name}'")
    logger.info(f"Input segmentation: {segmentation_user_id}/{segmentation_session_id} @ {voxel_spacing}Å")
    logger.info(f"Output segmentation: {output_params['name']} ({output_user_id}/{output_session_id})")
    logger.info(f"Expansion distance: {distance} Å ({distance / voxel_spacing:.1f} voxels)")

    # Import batch function
    from copick_utils.process.expand_labels import expand_labels_batch

    # Process runs
    results = expand_labels_batch(
        root=root,
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        segmentation_session_id=segmentation_session_id,
        voxel_spacing=voxel_spacing,
        distance=distance,
        output_user_id=output_user_id,
        output_session_id=output_session_id,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_added = sum(result.get("voxels_added", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total voxels added across all runs: {total_added}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
