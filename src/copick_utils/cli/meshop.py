"""CLI commands for mesh logical operations (boolean operations)."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from copick_utils.cli.input_output_selection import validate_conversion_placeholders
from copick_utils.cli.util import (
    add_boolean_operation_option,
    add_mesh_input_options,
    add_mesh_output_options,
    add_workers_option,
)
from copick_utils.converters.config_models import SelectorConfig, TaskConfig


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
@add_mesh_input_options
@add_mesh_input_options(suffix="2")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_boolean_operation_option
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_mesh_output_options(default_tool="meshop")
@add_debug_option
def meshop(
    config,
    run_names,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    mesh_object_name2,
    mesh_user_id2,
    mesh_session_id2,
    operation,
    workers,
    mesh_object_name_output,
    mesh_user_id_output,
    mesh_session_id_output,
    individual_meshes,
    debug,
):
    """
    Perform boolean operations between two meshes.

    \b
    Supports the following operations:
    - union: Combine both meshes using boolean union
    - difference: First mesh minus second mesh
    - intersection: Common volume of both meshes
    - exclusion: Exclusive or (XOR) of both meshes
    - concatenate: Simple concatenation without boolean operations

    \b
    Supports flexible input/output selection modes:
    - One-to-one: exact session IDs → exact session ID
    - Many-to-many: regex patterns → template with {input_session_id}

    \b
    Examples:
        # Union of two mesh sets
        copick logical meshop --operation union --mesh-session-id "manual-001" --input2-session-id "auto-001" --mesh-session-id "union-001"
        \b
        # Difference operation with pattern matching
        copick logical meshop --operation difference --mesh-session-id "manual-.*" --input2-session-id "mask-.*" --mesh-session-id "diff-{input_session_id}"
    """
    from copick_utils.logical.mesh_operations import (
        mesh_concatenate,
        mesh_difference,
        mesh_exclusion,
        mesh_intersection,
        mesh_union,
    )

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Validate placeholder requirements
    try:
        validate_conversion_placeholders(mesh_session_id, mesh_session_id_output, individual_meshes)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    logger.info(f"Performing {operation} operation on meshes for object '{mesh_object_name}'")
    logger.info(f"First mesh pattern: {mesh_user_id}/{mesh_session_id}")
    logger.info(f"Second mesh pattern: {mesh_user_id2}/{mesh_session_id2}")
    logger.info(
        f"Target mesh template: {mesh_object_name_output or mesh_object_name} ({mesh_user_id_output}/{mesh_session_id_output})",
    )

    # Select the appropriate converter function based on operation
    converter_functions = {
        "union": mesh_union,
        "difference": mesh_difference,
        "intersection": mesh_intersection,
        "exclusion": mesh_exclusion,
        "concatenate": mesh_concatenate,
    }

    converter_functions[operation]

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

    # Create type-safe Pydantic selector configurations
    selector1_config = SelectorConfig(
        input_type="mesh",
        output_type="mesh",
        input_object_name=mesh_object_name,
        input_user_id=mesh_user_id,
        input_session_id=mesh_session_id,
        output_object_name=mesh_object_name_output,
        output_user_id=mesh_user_id_output,
        output_session_id=mesh_session_id_output,
        individual_outputs=individual_meshes,
    )

    selector2_config = SelectorConfig(
        input_type="mesh",
        output_type="mesh",
        input_object_name=mesh_object_name2,
        input_user_id=mesh_user_id2,
        input_session_id=mesh_session_id2,
        output_object_name=mesh_object_name2,  # Not used for second input
        output_user_id=mesh_user_id2,  # Not used for second input
        output_session_id=mesh_session_id2,  # Not used for second input
        individual_outputs=False,  # Not used for second input
    )

    config = TaskConfig(
        type="dual_selector",
        selectors=[selector1_config, selector2_config],
        pairing_method="index_order",
    )

    # Parallel discovery and processing - no sequential bottleneck!
    results = lazy_batch_function(
        root=root,
        config=config,
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
