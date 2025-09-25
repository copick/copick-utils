"""CLI commands for segmentation logical operations (boolean operations)."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from copick_utils.cli.util import (
    add_boolean_operation_option,
    add_segmentation_boolean_output_options,
    add_segmentation_input_options,
    add_workers_option,
)
from copick_utils.converters.config_models import SelectorConfig, TaskConfig


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
@add_segmentation_input_options(include_multilabel=False)
@add_segmentation_input_options(suffix="2", include_multilabel=False, include_voxel_spacing=False)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_boolean_operation_option
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_segmentation_boolean_output_options(default_tool="segop")
@add_debug_option
def segop(
    config,
    run_names,
    seg_name,
    seg_user_id,
    seg_session_id,
    voxel_spacing,
    seg_name2,
    seg_user_id2,
    seg_session_id2,
    operation,
    workers,
    seg_name_output,
    seg_user_id_output,
    seg_session_id_output,
    voxel_spacing_output,
    tomo_type,
    debug,
):
    """
    Perform boolean operations between two segmentations.

    \b
    Supports the following boolean operations:
    - union: Combine both segmentations (logical OR)
    - difference: First segmentation minus second segmentation
    - intersection: Common voxels of both segmentations (logical AND)
    - exclusion: Exclusive or (XOR) of both segmentations

    \b
    Note: Both input segmentations should be binary (non-multilabel) for meaningful results.

    \b
    Examples:
        # Union of two segmentation sets
        copick logical segop --operation union --seg-session-id "manual-001" --input2-session-id "auto-001" --seg-session-id "union-001" --voxel-spacing 10.0
        \b
        # Difference operation with pattern matching
        copick logical segop --operation difference --seg-session-id "manual-.*" --input2-session-id "mask-.*" --seg-session-id "diff-{input_session_id}" --voxel-spacing 10.0
    """

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    logger.info(f"Performing {operation} operation on segmentations for '{seg_name}'")
    logger.info(f"First segmentation pattern: {seg_user_id}/{seg_session_id}")
    logger.info(f"Second segmentation pattern: {seg_user_id2}/{seg_session_id2}")
    logger.info(f"Target segmentation template: {seg_name_output} ({seg_user_id_output}/{seg_session_id_output})")

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

    # Create type-safe Pydantic selector configurations
    selector1_config = SelectorConfig(
        input_type="segmentation",
        output_type="segmentation",
        input_object_name=seg_name,
        input_user_id=seg_user_id,
        input_session_id=seg_session_id,
        output_object_name=seg_name_output,
        output_user_id=seg_user_id_output,
        output_session_id=seg_session_id_output,
        segmentation_name=seg_name,
        voxel_spacing=voxel_spacing,
    )

    selector2_config = SelectorConfig(
        input_type="segmentation",
        output_type="segmentation",
        input_object_name=seg_name2,
        input_user_id=seg_user_id2,
        input_session_id=seg_session_id2,
        output_object_name=seg_name2,  # Not used for second input
        output_user_id=seg_user_id2,  # Not used for second input
        output_session_id=seg_session_id2,  # Not used for second input
        segmentation_name=seg_name2,
        voxel_spacing=voxel_spacing,  # Use same voxel spacing for consistency
    )

    config = TaskConfig(
        type="dual_selector",
        selectors=[selector1_config, selector2_config],
        pairing_method="index_order",
        additional_params={
            "voxel_spacing": voxel_spacing_output,
            "tomo_type": tomo_type,
            "is_multilabel": False,
        },
    )

    # Parallel discovery and processing - no sequential bottleneck!
    results = lazy_batch_function(
        root=root,
        config=config,
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
    logger.info(f"Total {operation} operations completed: {total_processed}")
    logger.info(f"Total voxels created: {total_voxels}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
