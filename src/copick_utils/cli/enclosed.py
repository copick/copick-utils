"""CLI command for enclosed segmentation operations (finding and absorbing enclosed components)."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import add_dual_input_options, add_output_option, add_workers_option
from copick_utils.util.config_models import create_dual_selector_config


@click.command(
    context_settings={"show_default": True},
    short_help="Remove enclosed components from a segmentation.",
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
@add_dual_input_options("segmentation")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--voxel-spacing",
    "-vs",
    type=float,
    required=True,
    help="Voxel spacing for input and output segmentations.",
)
@optgroup.option(
    "--margin",
    "-m",
    type=int,
    default=1,
    help="Number of voxels to dilate when checking if components are enclosed.",
)
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
    help="Minimum component volume in cubic angstroms (Å³) to consider (optional).",
)
@optgroup.option(
    "--max-size",
    type=float,
    default=None,
    help="Maximum component volume in cubic angstroms (Å³) to consider (optional).",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="enclosed")
@add_debug_option
def enclosed(
    config,
    run_names,
    input1_uri,
    input2_uri,
    voxel_spacing,
    margin,
    connectivity,
    min_size,
    max_size,
    workers,
    output_uri,
    debug,
):
    """
    Remove enclosed components from a segmentation.

    This command identifies connected components in the first segmentation (inner) that are
    completely surrounded by the second segmentation (outer), and removes them from the inner
    segmentation. Useful for cleaning up noise, artifacts, or unwanted fragments.

    \b
    URI Format:
        Segmentations: name:user_id/session_id (voxel spacing specified via --voxel-spacing)

    \b
    Algorithm:
        1. Label connected components in the inner segmentation (input1)
        2. Dilate each component by the specified margin
        3. Check if the dilated component is fully contained within the outer segmentation (input2)
        4. If enclosed (and within size limits), remove the component from the inner segmentation
        5. Output cleaned version of the inner segmentation

    \b
    Examples:
        # Remove small vesicle fragments that are enclosed by membrane
        copick logical enclosed -vs 10.0 -i1 "vesicle:user1/auto-001" -i2 "membrane:user1/manual-001" -o "vesicle_clean"

        # Remove noise fragments with size filtering (volumes in Å³)
        copick logical enclosed -vs 10.0 -i1 "fragments:user1/.*" -i2 "cell:user1/.*" -o "cleaned" --min-size 1000 --max-size 100000 --margin 2
    """

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Append voxel spacing to URIs (only if not already present)
    input1_uri_full = f"{input1_uri}@{voxel_spacing}" if "@" not in input1_uri else input1_uri
    input2_uri_full = f"{input2_uri}@{voxel_spacing}" if "@" not in input2_uri else input2_uri
    output_uri_full = f"{output_uri}@{voxel_spacing}" if "@" not in output_uri else output_uri

    # Create config directly from URIs with smart defaults
    try:
        task_config = create_dual_selector_config(
            input1_uri=input1_uri_full,
            input2_uri=input2_uri_full,
            input_type="segmentation",
            output_uri=output_uri_full,
            output_type="segmentation",
            command_name="enclosed",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Extract parameters for logging
    input1_params = parse_copick_uri(input1_uri, "segmentation")
    input2_params = parse_copick_uri(input2_uri, "segmentation")
    output_params = parse_copick_uri(output_uri_full, "segmentation")

    logger.info(
        f"Removing enclosed components from '{input1_params['name']}' using '{input2_params['name']}' as reference",
    )
    logger.info(f"Segmentation to clean: {input1_params['user_id']}/{input1_params['session_id']}")
    logger.info(f"Reference segmentation: {input2_params['user_id']}/{input2_params['session_id']}")
    logger.info(
        f"Target segmentation template: {output_params['name']} ({output_params['user_id']}/{output_params['session_id']})",
    )
    logger.info(f"Parameters: margin={margin}, connectivity={connectivity}, min_size={min_size}, max_size={max_size}")

    # Map connectivity string to numeric value
    connectivity_map = {
        "face": 1,
        "face-edge": 2,
        "all": 3,
    }
    connectivity_value = connectivity_map[connectivity]

    # Import the lazy batch converter
    from copick_utils.logical.enclosed_operations import segmentation_enclosed_lazy_batch

    # Parallel discovery and processing
    results = segmentation_enclosed_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        voxel_spacing=voxel_spacing,
        margin=margin,
        connectivity=connectivity_value,
        min_size=min_size,
        max_size=max_size,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_voxels_kept = sum(result.get("voxels_kept", 0) for result in results.values() if result)
    total_processed = sum(result.get("processed", 0) for result in results.values() if result)
    total_components_removed = sum(result.get("components_removed", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total enclosed operations completed: {total_processed}")
    logger.info(f"Total components removed: {total_components_removed}")
    logger.info(f"Total voxels remaining in cleaned segmentations: {total_voxels_kept}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
