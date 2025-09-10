import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger


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
@optgroup.option(
    "--segmentation-name",
    required=True,
    help="Name of the segmentations to process.",
)
@optgroup.option(
    "--segmentation-user-id",
    required=True,
    help="User ID of the segmentations to process.",
)
@optgroup.option(
    "--session-id-pattern",
    required=True,
    help="Session ID pattern (regex) or exact session ID to match segmentations.",
)
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
@optgroup.option(
    "--output-session-id-template",
    help="Template for output session IDs. Use {input_session_id} as placeholder. Default: same as input.",
)
@optgroup.option(
    "--output-user-id",
    default="skel",
    help="User ID for output segmentations.",
)
@add_debug_option
def skeletonize(
    config,
    run_names,
    segmentation_name,
    segmentation_user_id,
    session_id_pattern,
    method,
    remove_noise,
    min_object_size,
    remove_short_branches,
    min_branch_length,
    workers,
    output_session_id_template,
    output_user_id,
    debug,
):
    """3D skeletonization of segmentations using regex pattern matching.

    This command can process multiple segmentations by matching session IDs against
    a regex pattern. This is useful for processing the output of connected components
    separation (e.g., pattern "inst-.*" to match "inst-0", "inst-1", etc.).

    Examples:
    - Exact match: --session-id-pattern "inst-0"
    - Regex pattern: --session-id-pattern "inst-.*"
    - Regex pattern: --session-id-pattern "inst-[0-9]+"
    """
    from copick_utils.process.skeletonize import skeletonize_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    logger.info(f"Skeletonizing segmentations '{segmentation_name}'")
    logger.info(f"Source segmentations: {segmentation_user_id} matching pattern '{session_id_pattern}'")
    logger.info(f"Method: {method}, output user ID: {output_user_id}")
    logger.info(f"Preprocessing: remove_noise={remove_noise} (min_size={min_object_size})")
    logger.info(f"Post-processing: remove_short_branches={remove_short_branches} (min_length={min_branch_length})")

    if output_session_id_template:
        logger.info(f"Output session ID template: '{output_session_id_template}'")
    else:
        logger.info("Output session IDs: same as input")

    if run_names_list:
        logger.info(f"Processing {len(run_names_list)} specific runs")
    else:
        logger.info(f"Processing all {len(root.runs)} runs")

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
