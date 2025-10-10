import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import expand_output_uri, parse_copick_uri

from copick_utils.cli.util import add_input_option, add_output_option


@click.command(
    context_settings={"show_default": True},
    short_help="3D skeletonization of segmentations.",
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
    "--method",
    type=click.Choice(["skimage", "distance_transform"]),
    default="skimage",
    help="Skeletonization method.",
)
@optgroup.option(
    "--remove-noise/--keep-noise",
    is_flag=True,
    default=True,
    help="Remove small objects before skeletonization.",
)
@optgroup.option(
    "--min-object-size",
    type=int,
    default=50,
    help="Minimum size of objects to keep during preprocessing.",
)
@optgroup.option(
    "--remove-short-branches/--keep-short-branches",
    is_flag=True,
    default=True,
    help="Remove short branches from skeleton.",
)
@optgroup.option(
    "--min-branch-length",
    type=int,
    default=5,
    help="Minimum length of branches to keep.",
)
@optgroup.option(
    "--workers",
    type=int,
    default=8,
    help="Number of worker processes.",
)
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="skel")
@add_debug_option
def skeletonize(
    config,
    run_names,
    input_uri,
    method,
    remove_noise,
    min_object_size,
    remove_short_branches,
    min_branch_length,
    workers,
    output_uri,
    debug,
):
    """3D skeletonization of segmentations using pattern matching.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing

    \b
    This command can process multiple segmentations by matching session IDs against
    a pattern. This is useful for processing the output of connected components
    separation (e.g., pattern "inst-.*" to match "inst-0", "inst-1", etc.).

    \b
    Examples:
        # Skeletonize exact match
        copick process skeletonize -i "membrane:user1/inst-0@10.0" -o "membrane:skel/skel-0@10.0"

        # Skeletonize all instances using pattern
        copick process skeletonize -i "membrane:user1/inst-.*@10.0" -o "membrane:skel/skel-{input_session_id}@10.0"
    """
    from copick_utils.process.skeletonize import skeletonize_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Expand output URI with smart defaults
    try:
        output_uri = expand_output_uri(
            output_uri=output_uri,
            input_uri=input_uri,
            input_type="segmentation",
            output_type="segmentation",
            command_name="skeletonize",
            individual_outputs=False,
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
    session_id_pattern = input_params["session_id"]

    # Parse output URI (now fully expanded)
    try:
        output_params = parse_copick_uri(output_uri, "segmentation")
    except ValueError as e:
        raise click.BadParameter(f"Invalid output URI: {e}") from e

    output_user_id = output_params["user_id"]
    output_session_id_template = output_params["session_id"]

    logger.info(f"Skeletonizing segmentations '{segmentation_name}'")
    logger.info(f"Source segmentations: {segmentation_user_id} matching pattern '{session_id_pattern}'")
    logger.info(f"Method: {method}, output user ID: {output_user_id}")
    logger.info(f"Preprocessing: remove_noise={remove_noise} (min_size={min_object_size})")
    logger.info(f"Post-processing: remove_short_branches={remove_short_branches} (min_length={min_branch_length})")
    logger.info(f"Output session ID template: '{output_session_id_template}'")

    results = skeletonize_batch(
        root=root,
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        session_id_pattern=session_id_pattern,
        method=method,
        remove_noise=remove_noise,
        min_object_size=min_object_size,
        remove_short_branches=remove_short_branches,
        min_branch_length=min_branch_length,
        output_session_id_template=output_session_id_template,
        output_user_id=output_user_id,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_skeletons = sum(result.get("skeletons_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("segmentations_processed", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total segmentations processed: {total_processed}")
    logger.info(f"Total skeletons created: {total_skeletons}")
