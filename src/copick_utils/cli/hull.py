"""CLI command for computing various hull operations on meshes."""
import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import (
    add_input_option,
    add_output_option,
    add_workers_option,
)
from copick_utils.util.config_models import create_simple_config


@click.command(
    context_settings={"show_default": True},
    short_help="Compute hull operations on meshes.",
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
@add_input_option("mesh")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--hull-type",
    type=click.Choice(["convex"]),
    default="convex",
    help="Type of hull to compute.",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_output_option("mesh", default_tool="hull")
@optgroup.option(
    "--individual-meshes/--no-individual-meshes",
    "-im",
    is_flag=True,
    default=False,
    help="Create individual meshes for each instance (enables {instance_id} placeholder).",
)
@add_debug_option
def hull(
    config,
    run_names,
    input_uri,
    hull_type,
    workers,
    output_uri,
    individual_meshes,
    debug,
):
    """
    Compute hull operations on meshes.

    \b
    URI Format:
        Meshes: object_name:user_id/session_id

    \b
    Currently supports convex hull computation, where the convex hull is the
    smallest convex shape that contains all vertices of the original mesh.

    \b
    Examples:
        # Compute convex hull for meshes
        copick process hull -i "membrane:user1/session1" -o "membrane:hull/hull-session"

        # Process specific runs
        copick process hull -r run1 -r run2 -i "membrane:user1/session1" -o "membrane:hull/convex-001" --hull-type convex
    """
    from copick_utils.process.hull import hull_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Create config directly from URIs with smart defaults
    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="mesh",
            output_uri=output_uri,
            output_type="mesh",
            individual_outputs=individual_meshes,
            command_name="hull",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Extract parameters for logging
    input_params = parse_copick_uri(input_uri, "mesh")
    output_params = parse_copick_uri(output_uri, "mesh")

    logger.info(f"Computing {hull_type} hull for meshes '{input_params['object_name']}'")
    logger.info(f"Source mesh pattern: {input_params['user_id']}/{input_params['session_id']}")
    logger.info(
        f"Target mesh template: {output_params['object_name']} ({output_params['user_id']}/{output_params['session_id']})",
    )

    # Parallel discovery and processing - no sequential bottleneck!
    results = hull_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        hull_type=hull_type,
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
    logger.info(f"Total {hull_type} hull operations completed: {total_processed}")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
