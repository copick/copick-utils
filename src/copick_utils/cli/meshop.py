"""CLI commands for mesh logical operations (boolean operations)."""

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
    short_help="Perform boolean operations between two meshes.",
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
@add_dual_input_options("mesh")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_boolean_operation_option
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_output_option("mesh", default_tool="meshop")
@optgroup.option(
    "--individual-meshes/--no-individual-meshes",
    "-im",
    is_flag=True,
    default=False,
    help="Create individual meshes for each instance (enables {instance_id} placeholder).",
)
@add_debug_option
def meshop(
    config,
    run_names,
    input1_uri,
    input2_uri,
    operation,
    workers,
    output_uri,
    individual_meshes,
    debug,
):
    """
    Perform boolean operations between two meshes.

    \b
    URI Format:
        Meshes: object_name:user_id/session_id

    \b
    Supports the following operations:
        - union: Combine both meshes using boolean union
        - difference: First mesh minus second mesh
        - intersection: Common volume of both meshes
        - exclusion: Exclusive or (XOR) of both meshes
        - concatenate: Simple concatenation without boolean operations

    \b
    Examples:
        # Union of two mesh sets
        copick logical meshop --operation union -i1 "membrane:user1/manual-001" -i2 "vesicle:user1/auto-001" -o "combined:meshop/union-001"

        # Difference operation with pattern matching
        copick logical meshop --operation difference -i1 "membrane:user1/manual-.*" -i2 "mask:user1/mask-.*" -o "membrane:meshop/diff-{input_session_id}"
    """

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Create config directly from URIs
    try:
        task_config = create_dual_selector_config(
            input1_uri=input1_uri,
            input2_uri=input2_uri,
            input_type="mesh",
            output_uri=output_uri,
            output_type="mesh",
            individual_outputs=individual_meshes,
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Extract parameters for logging
    input1_params = parse_copick_uri(input1_uri, "mesh")
    input2_params = parse_copick_uri(input2_uri, "mesh")
    output_params = parse_copick_uri(output_uri, "mesh")

    logger.info(f"Performing {operation} operation on meshes for object '{input1_params['object_name']}'")
    logger.info(f"First mesh pattern: {input1_params['user_id']}/{input1_params['session_id']}")
    logger.info(f"Second mesh pattern: {input2_params['user_id']}/{input2_params['session_id']}")
    logger.info(
        f"Target mesh template: {output_params['object_name']} ({output_params['user_id']}/{output_params['session_id']})",
    )

    # Import the appropriate lazy batch converter based on operation
    from copick_utils.logical.mesh_operations import (
        mesh_concatenate_lazy_batch,
        mesh_difference_lazy_batch,
        mesh_exclusion_lazy_batch,
        mesh_intersection_lazy_batch,
        mesh_union_lazy_batch,
    )

    # Select the appropriate lazy batch converter
    lazy_batch_functions = {
        "union": mesh_union_lazy_batch,
        "difference": mesh_difference_lazy_batch,
        "intersection": mesh_intersection_lazy_batch,
        "exclusion": mesh_exclusion_lazy_batch,
        "concatenate": mesh_concatenate_lazy_batch,
    }

    lazy_batch_function = lazy_batch_functions[operation]

    # Parallel discovery and processing - no sequential bottleneck!
    results = lazy_batch_function(
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
    logger.info(f"Total {operation} operations completed: {total_processed}")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
