import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import expand_output_uri, parse_copick_uri

from copick_utils.cli.util import add_input_option, add_output_option


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
@optgroup.option(
    "--workers",
    type=int,
    default=8,
    help="Number of worker processes.",
)
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
    from copick_utils.process.connected_components import separate_components_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Expand output URI with smart defaults (individual_outputs=True for {instance_id})
    try:
        output_uri = expand_output_uri(
            output_uri=output_uri,
            input_uri=input_uri,
            input_type="segmentation",
            output_type="segmentation",
            command_name="components",
            individual_outputs=True,
        )
    except ValueError as e:
        raise click.BadParameter(f"Error expanding output URI: {e}") from e

    # Parse input URI
    try:
        input_params = parse_copick_uri(input_uri, "segmentation")
    except ValueError as e:
        raise click.BadParameter(f"Invalid input URI: {e}") from e

    segmentation_name = input_params["name"]
    segmentation_user_id = input_params["user_id"]
    segmentation_session_id = input_params["session_id"]

    # Parse output URI (now fully expanded)
    try:
        output_params = parse_copick_uri(output_uri, "segmentation")
    except ValueError as e:
        raise click.BadParameter(f"Invalid output URI: {e}") from e

    output_user_id = output_params["user_id"]
    output_session_id_template = output_params["session_id"]

    # Validate that output_session_id_template contains {instance_id}
    if "{instance_id}" not in output_session_id_template:
        raise click.BadParameter("Output URI must contain {instance_id} placeholder for separate_components command")

    logger.info(f"Separating connected components for segmentation '{segmentation_name}'")
    logger.info(f"Source segmentation: {segmentation_user_id}/{segmentation_session_id}")
    logger.info(f"Output template: {output_params['name']} ({output_user_id}/{output_session_id_template})")
    logger.info(f"Connectivity: {connectivity}")
    if min_size is not None:
        logger.info(f"Minimum size: {min_size} Å³")
    logger.info(f"Processing as {'multilabel' if multilabel else 'binary'} segmentation")

    results = separate_components_batch(
        root=root,
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        segmentation_session_id=segmentation_session_id,
        connectivity=connectivity,
        min_size=min_size,
        session_id_template=output_session_id_template,
        output_user_id=output_user_id,
        multilabel=multilabel,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_components = sum(result.get("components_created", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total components created: {total_components}")
