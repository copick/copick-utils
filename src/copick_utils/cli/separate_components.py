import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import add_input_option, add_output_option, add_workers_option
from copick_utils.util.config_models import create_simple_config


@click.command(
    context_settings={"show_default": True},
    short_help="Separate connected components in segmentations.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input segmentation.")
@optgroup.option(
    "--run-names",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_input_option("segmentation")
@optgroup.group("\nTool Options", help="Options related to this tool.")
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
    help="Minimum component volume in cubic angstroms (Å³) to keep (optional).",
)
@optgroup.option(
    "--multilabel/--binary",
    is_flag=True,
    default=True,
    help="Process as multilabel segmentation (analyze each label separately).",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="components")
@add_debug_option
def separate_components(
    config,
    run_names,
    input_uri,
    connectivity,
    min_size,
    multilabel,
    workers,
    output_uri,
    debug,
):
    """Separate connected components in segmentations into individual segmentations.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing

    \b
    For multilabel segmentations, connected components analysis is performed on each
    label separately. Output segmentations use {instance_id} placeholder for auto-numbering
    (e.g., "inst-0", "inst-1", etc.).

    \b
    Examples:
        # Separate components with smart defaults (auto user_id and session template)
        copick process separate_components -i "membrane:user1/manual-001@10.0" -o "{instance_id}"

        # Custom session prefix
        copick process separate_components -i "membrane:user1/manual-001@10.0" -o "membrane:components/inst-{instance_id}"

        # Full URI specification
        copick process separate_components -i "membrane:user1/manual-001@10.0" -o "membrane:components/comp-{instance_id}@10.0"
    """
    from copick_utils.process.connected_components import separate_components_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Create config from URIs with smart defaults (individual_outputs for {instance_id})
    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="segmentation",
            output_uri=output_uri,
            output_type="segmentation",
            individual_outputs=True,
            command_name="components",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Log parameters
    input_params = parse_copick_uri(input_uri, "segmentation")
    logger.info(f"Separating connected components for segmentation '{input_params['name']}'")
    logger.info(f"Connectivity: {connectivity}")
    if min_size is not None:
        logger.info(f"Minimum size: {min_size} Å³")
    logger.info(f"Processing as {'multilabel' if multilabel else 'binary'} segmentation")

    # Parallel discovery and processing
    results = separate_components_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        connectivity=connectivity,
        min_size=min_size,
        multilabel=multilabel,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_components = sum(result.get("components_created", 0) for result in results.values() if result)

    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total components created: {total_components}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
