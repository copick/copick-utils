import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import (
    add_cap_extraction_options,
    add_input_option,
    add_output_option,
    add_workers_option,
)
from copick_utils.util.config_models import create_simple_config


@click.command(
    context_settings={"show_default": True},
    short_help="Extract the top/bottom surfaces (caps) of a slab box mesh.",
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
@add_cap_extraction_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_output_option("mesh", default_tool="mesh2caps")
@add_debug_option
def mesh2caps(
    config,
    run_names,
    input_uri,
    axis,
    angle_threshold,
    surface,
    auto_axis,
    workers,
    output_uri,
    debug,
):
    """
    Extract the top/bottom surfaces ("caps") of a closed slab box mesh as an open mesh.

    A boundary slab (e.g. the ``valid-sample`` mesh) is a closed box: a top surface, a parallel
    bottom surface, and 4 side walls. This command keeps only the near-horizontal cap faces and
    drops the near-vertical side walls, classifying faces geometrically by their normal orientation
    relative to the slab axis (so it works even on a re-triangulated boolean result).

    The resulting open mesh feeds ``copick logical clippicks`` to select particles within a distance
    of the top/bottom of the specimen WITHOUT the side walls contaminating the distance field.

    URI Format:

        \b
        Meshes: object_name:user_id/session_id

    Examples:

        \b
        # Extract both caps of the valid-sample slab
        copick convert mesh2caps -i "valid-sample:meshop/0" -o "valid-sample-caps:mesh2caps/0"

        \b
        # Extract only the top cap, with a tighter cap angle
        copick convert mesh2caps --surface top --angle-threshold 30 \\
            -i "valid-sample:meshop/0" -o "valid-sample-caps:mesh2caps/top-0"

        \b
        # Strongly tilted slab: infer the slab normal automatically
        copick convert mesh2caps --auto-axis -i "valid-sample:meshop/0" -o "valid-sample-caps:mesh2caps/0"

    See Also:

        \b
        copick logical clippicks: select picks by distance to the extracted caps
        copick logical meshop: build the slab box the caps are extracted from
    """
    from copick_utils.converters.caps_from_mesh import caps_from_mesh_lazy_batch

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
            command_name="mesh2caps",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Extract parameters for logging
    input_params = parse_copick_uri(input_uri, "mesh")
    output_params = parse_copick_uri(output_uri, "mesh")

    logger.info(f"Extracting slab caps for object '{input_params['object_name']}'")
    logger.info(f"Source mesh pattern: {input_params['user_id']}/{input_params['session_id']}")
    logger.info(
        f"Target mesh template: {output_params['object_name']} ({output_params['user_id']}/{output_params['session_id']})",
    )
    logger.info(
        f"Surface: {surface}, axis: {axis}, angle threshold: {angle_threshold} deg, auto-axis: {auto_axis}",
    )

    # Parallel discovery and processing - no sequential bottleneck!
    results = caps_from_mesh_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        axis=axis,
        angle_threshold=angle_threshold,
        which=surface,
        auto_axis=auto_axis,
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
    logger.info(f"Total conversion tasks completed: {total_processed}")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
