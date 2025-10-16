"""CLI commands for mesh logical operations (boolean operations)."""

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
    short_help="Perform boolean operations between meshes.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to input meshes.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_multi_input_options("mesh")
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
    input_uris,
    operation,
    workers,
    output_uri,
    individual_meshes,
    debug,
):
    """
    Perform boolean operations between meshes.

    \b
    URI Format:
        Meshes: object_name:user_id/session_id

    \b
    Pattern Support:
        - Glob (default): Use * and ? wildcards (e.g., "membrane:user*/session-*")
        - Regex: Prefix with 're:' (e.g., "re:membrane:user\\d+/session-\\d+")

    \b
    Operations:
        - union: Combine meshes using boolean union - accepts N≥1 inputs
        - difference: First minus second - requires exactly 2 inputs
        - intersection: Common volume - requires exactly 2 inputs
        - exclusion: Exclusive or (XOR) - requires exactly 2 inputs
        - concatenate: Simple concatenation without boolean ops - accepts N≥1 inputs

    \b
    Single-Input Pattern Expansion (union & concatenate):
        When providing a single -i flag with a pattern, union/concatenate operations
        will expand the pattern within each run and merge all matching meshes.
        This is useful for combining multiple versions/annotations within each run.

    \b
    Examples:
        # Single-input union: merge all matching meshes within each run
        copick logical meshop --operation union \\
            -i "membrane:user*/manual-*" \\
            -o "merged"

        # Single-input concatenation: concatenate all matching meshes per run
        copick logical meshop --operation concatenate \\
            -i "part*:user1/session-*" \\
            -o "combined"

        # N-way union with multiple -i flags (merge across different objects)
        copick logical meshop --operation union \\
            -i "membrane:user1/manual-*" \\
            -i "vesicle:user2/auto-*" \\
            -i "ribosome:user3/pred-*" \\
            -o "merged"

        # N-way union with regex patterns
        copick logical meshop --operation union \\
            -i "re:membrane:user1/manual-\\d+" \\
            -i "re:vesicle:user2/auto-\\d+" \\
            -o "merged"

        # 2-way difference (exactly 2 inputs required)
        copick logical meshop --operation difference \\
            -i "membrane:user1/manual-001" \\
            -i "mask:user1/mask-001" \\
            -o "membrane:meshop/masked"

        # N-way concatenation with multiple -i flags
        copick logical meshop --operation concatenate \\
            -i "part1:user1/session" \\
            -i "part2:user1/session" \\
            -i "part3:user1/session" \\
            -o "combined"
    """
    logger = get_logger(__name__, debug=debug)

    # VALIDATION: Check input count vs operation
    num_inputs = len(input_uris)

    if operation in ["difference", "intersection", "exclusion"]:
        if num_inputs != 2:
            raise click.BadParameter(
                f"'{operation}' operation requires exactly 2 inputs, got {num_inputs}. Provide exactly 2 -i flags.",
            )
    elif operation in ["union", "concatenate"] and num_inputs < 1:
        raise click.BadParameter(
            f"'{operation}' operation requires at least 1 input, got {num_inputs}. Provide 1 or more -i flags.",
        )

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Create appropriate config
    try:
        if num_inputs == 1:
            # Single input with pattern expansion (for union and concatenate)
            from copick_utils.util.config_models import create_single_selector_config

            task_config = create_single_selector_config(
                input_uri=input_uris[0],
                input_type="mesh",
                output_uri=output_uri,
                output_type="mesh",
                command_name="meshop",
                operation=operation,
            )
        elif num_inputs == 2:
            task_config = create_dual_selector_config(
                input1_uri=input_uris[0],
                input2_uri=input_uris[1],
                input_type="mesh",
                output_uri=output_uri,
                output_type="mesh",
                individual_outputs=individual_meshes,
                command_name="meshop",
            )
        else:
            task_config = create_multi_selector_config(
                input_uris=input_uris,
                input_type="mesh",
                output_uri=output_uri,
                output_type="mesh",
                individual_outputs=individual_meshes,
                command_name="meshop",
            )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Logging
    if num_inputs == 1:
        logger.info(f"Performing {operation} operation with pattern-based input expansion")
        params = parse_copick_uri(input_uris[0], "mesh")
        logger.info(f"  Pattern: {params['object_name']} ({params['user_id']}/{params['session_id']})")
        logger.info("  Note: Pattern will be expanded to multiple meshes per run")
    else:
        logger.info(f"Performing {operation} operation on {num_inputs} meshes")
        for i, uri in enumerate(input_uris, start=1):
            params = parse_copick_uri(uri, "mesh")
            logger.info(f"  Input {i}: {params['object_name']} ({params['user_id']}/{params['session_id']})")

    output_params = parse_copick_uri(output_uri, "mesh")
    logger.info(f"Target: {output_params['object_name']} ({output_params['user_id']}/{output_params['session_id']})")

    # Select appropriate lazy batch converter
    if num_inputs == 1:
        # Single input with pattern expansion (union and concatenate)
        from copick_utils.logical.mesh_operations import (
            mesh_multi_concatenate_lazy_batch,
            mesh_multi_union_lazy_batch,
        )

        lazy_batch_functions = {
            "union": mesh_multi_union_lazy_batch,
            "concatenate": mesh_multi_concatenate_lazy_batch,
        }
    elif num_inputs == 2:
        from copick_utils.logical.mesh_operations import (
            mesh_concatenate_lazy_batch,
            mesh_difference_lazy_batch,
            mesh_exclusion_lazy_batch,
            mesh_intersection_lazy_batch,
            mesh_union_lazy_batch,
        )

        lazy_batch_functions = {
            "union": mesh_union_lazy_batch,
            "difference": mesh_difference_lazy_batch,
            "intersection": mesh_intersection_lazy_batch,
            "exclusion": mesh_exclusion_lazy_batch,
            "concatenate": mesh_concatenate_lazy_batch,
        }
    else:
        from copick_utils.logical.mesh_operations import (
            mesh_multi_concatenate_lazy_batch,
            mesh_multi_union_lazy_batch,
        )

        lazy_batch_functions = {
            "union": mesh_multi_union_lazy_batch,
            "concatenate": mesh_multi_concatenate_lazy_batch,
        }

    lazy_batch_function = lazy_batch_functions[operation]

    # Execute
    results = lazy_batch_function(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
    )

    # Aggregate results
    successful = sum(1 for r in results.values() if r and r.get("processed", 0) > 0)
    total_vertices = sum(r.get("vertices_created", 0) for r in results.values() if r)
    total_faces = sum(r.get("faces_created", 0) for r in results.values() if r)
    total_processed = sum(r.get("processed", 0) for r in results.values() if r)

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
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
