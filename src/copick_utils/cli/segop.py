"""CLI commands for segmentation logical operations (boolean operations)."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import (
    add_boolean_operation_option,
    add_dual_input_options,
    add_output_option,
    add_workers_option,
)
from copick_utils.converters.config_models import create_dual_selector_config


@click.command(
    context_settings={"show_default": True},
    short_help="Perform boolean operations between two segmentations.",
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
@add_boolean_operation_option
@optgroup.option(
    "--tomo-type",
    "-tt",
    default="wbp",
    help="Type of tomogram to use as reference.",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="segop")
@add_debug_option
def segop(
    config,
    run_names,
    input1_uri,
    input2_uri,
    operation,
    tomo_type,
    workers,
    output_uri,
    debug,
):
    """
    Perform boolean operations between two segmentations.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing

    \b
    Supports the following boolean operations:
        - union: Combine both segmentations (logical OR)
        - difference: First segmentation minus second segmentation
        - intersection: Common voxels of both segmentations (logical AND)
        - exclusion: Exclusive or (XOR) of both segmentations

    \b
    Note: Both input segmentations should be binary (non-multilabel) for meaningful results.
    Both inputs must have the same voxel spacing.

    \b
    Examples:
        # Union of two segmentation sets
        copick logical segop --operation union -i1 "membrane:user1/manual-001@10.0" -i2 "vesicle:user1/auto-001@10.0" -o "combined:segop/union-001@10.0"

        # Difference operation with pattern matching
        copick logical segop --operation difference -i1 "membrane:user1/manual-.*@10.0" -i2 "mask:user1/mask-.*@10.0" -o "membrane:segop/diff-{input_session_id}@10.0"
    """

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Create config directly from URIs
    try:
        task_config = create_dual_selector_config(
            input1_uri=input1_uri,
            input2_uri=input2_uri,
            input_type="segmentation",
            output_uri=output_uri,
            output_type="segmentation",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Extract parameters for logging
    input1_params = parse_copick_uri(input1_uri, "segmentation")
    input2_params = parse_copick_uri(input2_uri, "segmentation")
    output_params = parse_copick_uri(output_uri, "segmentation")

    voxel_spacing_output = output_params["voxel_spacing"]
    if isinstance(voxel_spacing_output, str):
        voxel_spacing_output = float(voxel_spacing_output)

    logger.info(f"Performing {operation} operation on segmentations for '{input1_params['name']}'")
    logger.info(f"First segmentation pattern: {input1_params['user_id']}/{input1_params['session_id']}")
    logger.info(f"Second segmentation pattern: {input2_params['user_id']}/{input2_params['session_id']}")
    logger.info(
        f"Target segmentation template: {output_params['name']} ({output_params['user_id']}/{output_params['session_id']})",
    )

    # Import the appropriate lazy batch converter based on operation
    from copick_utils.logical.segmentation_operations import (
        segmentation_difference_lazy_batch,
        segmentation_exclusion_lazy_batch,
        segmentation_intersection_lazy_batch,
        segmentation_union_lazy_batch,
    )

    # Select the appropriate lazy batch converter based on operation
    lazy_batch_functions = {
        "union": segmentation_union_lazy_batch,
        "difference": segmentation_difference_lazy_batch,
        "intersection": segmentation_intersection_lazy_batch,
        "exclusion": segmentation_exclusion_lazy_batch,
    }

    lazy_batch_function = lazy_batch_functions[operation]

    # Parallel discovery and processing - no sequential bottleneck!
    results = lazy_batch_function(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        voxel_spacing=voxel_spacing_output,
        tomo_type=tomo_type,
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
    logger.info(f"Total {operation} operations completed: {total_processed}")
    logger.info(f"Total voxels created: {total_voxels}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
