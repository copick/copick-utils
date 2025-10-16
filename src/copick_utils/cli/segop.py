"""CLI commands for segmentation logical operations (boolean operations)."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import (
    add_boolean_operation_option,
    add_multi_input_options,
    add_output_option,
    add_workers_option,
)
from copick_utils.util.config_models import (
    create_dual_selector_config,
    create_multi_selector_config,
)


@click.command(
    context_settings={"show_default": True},
    short_help="Perform boolean operations between segmentations.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to input segmentations.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_multi_input_options("segmentation")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_boolean_operation_option
@optgroup.option(
    "--voxel-spacing",
    "-vs",
    type=float,
    required=True,
    help="Voxel spacing for input and output segmentations.",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="segop")
@add_debug_option
def segop(
    config,
    run_names,
    input_uris,
    operation,
    voxel_spacing,
    workers,
    output_uri,
    debug,
):
    """
    Perform boolean operations between segmentations.

    \b
    URI Format:
        Segmentations: name:user_id/session_id (voxel spacing via --voxel-spacing)

    \b
    Pattern Support:
        - Glob (default): Use * and ? wildcards (e.g., "membrane:user*/session-*")
        - Regex: Prefix with 're:' (e.g., "re:membrane:user\\d+/session-\\d+")

    \b
    Operations:
        - union: Combine segmentations (logical OR) - accepts N≥1 inputs
        - difference: First minus second - requires exactly 2 inputs
        - intersection: Common voxels (logical AND) - requires exactly 2 inputs
        - exclusion: Exclusive or (XOR) - requires exactly 2 inputs

    \b
    Note: All segmentations are converted to binary for boolean operations.
    Voxel spacing applies globally to all inputs and output.

    \b
    Single-Input Pattern Expansion (union only):
        When providing a single -i flag with a pattern, the union operation will
        expand the pattern within each run and merge all matching segmentations.
        This is useful for combining multiple versions/annotations within each run.

    \b
    Examples:
        # Single-input union: merge all matching segmentations within each run
        copick logical segop --operation union -vs 10.0 \\
            -i "membrane:user*/manual-*" \\
            -o "merged"

        # N-way union with multiple -i flags (merge across different objects)
        copick logical segop --operation union -vs 10.0 \\
            -i "membrane:user1/manual-*" \\
            -i "vesicle:user2/auto-*" \\
            -i "ribosome:user3/pred-*" \\
            -o "merged"

        # N-way union with regex patterns
        copick logical segop --operation union -vs 10.0 \\
            -i "re:membrane:user1/manual-\\d+" \\
            -i "re:vesicle:user2/auto-\\d+" \\
            -o "merged"

        # 2-way difference (exactly 2 inputs required)
        copick logical segop --operation difference -vs 10.0 \\
            -i "membrane:user1/manual-001" \\
            -i "mask:user1/mask-001" \\
            -o "membrane:segop/masked"
    """
    logger = get_logger(__name__, debug=debug)

    # VALIDATION: Check input count vs operation
    num_inputs = len(input_uris)

    if operation in ["difference", "intersection", "exclusion"]:
        if num_inputs != 2:
            raise click.BadParameter(
                f"'{operation}' operation requires exactly 2 inputs, got {num_inputs}. Provide exactly 2 -i flags.",
            )
    elif operation == "union" and num_inputs < 1:
        raise click.BadParameter(
            f"'{operation}' operation requires at least 1 input, got {num_inputs}. Provide 1 or more -i flags.",
        )

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Append voxel spacing to all URIs
    input_uris_full = [f"{uri}@{voxel_spacing}" if "@" not in uri else uri for uri in input_uris]
    output_uri_full = f"{output_uri}@{voxel_spacing}" if "@" not in output_uri else output_uri

    # Create appropriate config based on input count
    try:
        if num_inputs == 1:
            # Single input with pattern expansion (only for union)
            from copick_utils.util.config_models import create_single_selector_config

            task_config = create_single_selector_config(
                input_uri=input_uris_full[0],
                input_type="segmentation",
                output_uri=output_uri_full,
                output_type="segmentation",
                command_name="segop",
                operation=operation,
            )
        elif num_inputs == 2:
            # Use existing dual selector for 2-input operations
            task_config = create_dual_selector_config(
                input1_uri=input_uris_full[0],
                input2_uri=input_uris_full[1],
                input_type="segmentation",
                output_uri=output_uri_full,
                output_type="segmentation",
                command_name="segop",
            )
        else:
            # Use new multi selector for N-way operations (N≥3)
            task_config = create_multi_selector_config(
                input_uris=input_uris_full,
                input_type="segmentation",
                output_uri=output_uri_full,
                output_type="segmentation",
                command_name="segop",
            )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Logging
    if num_inputs == 1:
        logger.info(f"Performing {operation} operation with pattern-based input expansion")
        params = parse_copick_uri(input_uris[0], "segmentation")
        logger.info(f"  Pattern: {params['name']} ({params['user_id']}/{params['session_id']})")
        logger.info("  Note: Pattern will be expanded to multiple segmentations per run")
    else:
        logger.info(f"Performing {operation} operation on {num_inputs} segmentations")
        for i, uri in enumerate(input_uris, start=1):
            params = parse_copick_uri(uri, "segmentation")
            logger.info(f"  Input {i}: {params['name']} ({params['user_id']}/{params['session_id']})")

    output_params = parse_copick_uri(output_uri_full, "segmentation")
    logger.info(f"Target: {output_params['name']} ({output_params['user_id']}/{output_params['session_id']})")

    # Select appropriate lazy batch converter
    if num_inputs == 1:
        # Single input with pattern expansion (only union supports this)
        from copick_utils.logical.segmentation_operations import segmentation_multi_union_lazy_batch

        lazy_batch_functions = {
            "union": segmentation_multi_union_lazy_batch,
        }
    elif num_inputs == 2:
        # Use existing dual converters
        from copick_utils.logical.segmentation_operations import (
            segmentation_difference_lazy_batch,
            segmentation_exclusion_lazy_batch,
            segmentation_intersection_lazy_batch,
            segmentation_union_lazy_batch,
        )

        lazy_batch_functions = {
            "union": segmentation_union_lazy_batch,
            "difference": segmentation_difference_lazy_batch,
            "intersection": segmentation_intersection_lazy_batch,
            "exclusion": segmentation_exclusion_lazy_batch,
        }
    else:
        # Use new N-way converters (only union supports N>2)
        from copick_utils.logical.segmentation_operations import segmentation_multi_union_lazy_batch

        lazy_batch_functions = {
            "union": segmentation_multi_union_lazy_batch,
        }

    lazy_batch_function = lazy_batch_functions[operation]

    # Execute parallel discovery and processing
    results = lazy_batch_function(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        voxel_spacing=voxel_spacing,
    )

    # Aggregate results
    successful = sum(1 for r in results.values() if r and r.get("processed", 0) > 0)
    total_voxels = sum(r.get("voxels_created", 0) for r in results.values() if r)
    total_processed = sum(r.get("processed", 0) for r in results.values() if r)

    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total {operation} operations completed: {total_processed}")
    logger.info(f"Total voxels created: {total_voxels}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
