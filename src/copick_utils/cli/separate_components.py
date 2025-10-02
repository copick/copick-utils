import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import add_input_option


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
    type=click.Choice(["6", "18", "26"]),
    default="26",
    help="Connectivity for connected components analysis (6, 18, or 26).",
)
@optgroup.option(
    "--min-size",
    type=int,
    default=0,
    help="Minimum size of components to keep (in voxels).",
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
@optgroup.option(
    "--session-id-prefix",
    default="inst-",
    help="Prefix for output segmentation session IDs.",
)
@optgroup.option(
    "--output-user-id",
    default="components",
    help="User ID for output segmentations.",
)
@add_debug_option
def separate_components(
    config,
    run_names,
    input_uri,
    connectivity,
    min_size,
    multilabel,
    workers,
    session_id_prefix,
    output_user_id,
    debug,
):
    """Separate connected components in segmentations into individual segmentations.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing

    \b
    For multilabel segmentations, connected components analysis is performed on each
    label separately. Output segmentations are created with session IDs using the
    specified prefix and incremental numbering (e.g., "inst-0", "inst-1", etc.).

    \b
    Examples:
        # Separate components from single segmentation
        copick process separate_components -i "membrane:user1/manual-001@10.0" --session-id-prefix "inst-" --output-user-id "components"

        # Process binary segmentation
        copick process separate_components -i "membrane:user1/manual-001@10.0" --binary --session-id-prefix "comp-"
    """
    from copick_utils.process.connected_components import separate_components_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None
    connectivity_int = int(connectivity)

    # Parse input URI
    try:
        input_params = parse_copick_uri(input_uri, "segmentation")
    except ValueError as e:
        raise click.BadParameter(f"Invalid input URI: {e}") from e

    segmentation_name = input_params["name"]
    segmentation_user_id = input_params["user_id"]
    segmentation_session_id = input_params["session_id"]

    logger.info(f"Separating connected components for segmentation '{segmentation_name}'")
    logger.info(f"Source segmentation: {segmentation_user_id}/{segmentation_session_id}")
    logger.info(f"Output prefix: {session_id_prefix}, user ID: {output_user_id}")
    logger.info(f"Connectivity: {connectivity_int}, min size: {min_size} voxels")
    logger.info(f"Processing as {'multilabel' if multilabel else 'binary'} segmentation")

    results = separate_components_batch(
        root=root,
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        segmentation_session_id=segmentation_session_id,
        connectivity=connectivity_int,
        min_size=min_size,
        session_id_prefix=session_id_prefix,
        output_user_id=output_user_id,
        multilabel=multilabel,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_components = sum(result.get("components_created", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total components created: {total_components}")
