import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option, add_run_names_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import add_input_option, add_output_option, add_workers_option
from copick_utils.util.config_models import create_simple_config


@click.command(
    context_settings={"show_default": True},
    short_help="Skeletonize segmentations in 3D using pattern matching.",
    no_args_is_help=True,
)
@add_config_option
@add_run_names_option
@optgroup.group("\nInput Options", help="Options related to the input segmentation.")
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
@add_workers_option
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
    """Skeletonize segmentations in 3D using pattern matching.

    Reduces each input segmentation to a 1-voxel-wide medial skeleton, exposing the
    centerlines of the objects it contains. Two backends are available: `skimage`
    (scikit-image 3D thinning) and `distance_transform` (local maxima of the
    Euclidean distance transform).

    The input session ID is treated as a regex, so a single invocation can skeletonize
    many segmentations at once. This pairs naturally with the output of connected-component
    separation (e.g. pattern `inst-.*` to match `inst-0`, `inst-1`, etc.). Optional cleanup
    removes small objects before thinning and prunes short spur branches afterwards.

    URI Format:

    \b
    Segmentations: name:user_id/session_id@voxel_spacing

    Examples:

    \b
    # Skeletonize a single segmentation (exact session match)
    copick process skeletonize -i "membrane:user1/inst-0@10.0" -o "membrane:skel/skel-0@10.0"

    \b
    # Skeletonize every instance matched by a session-ID pattern
    copick process skeletonize -i "membrane:user1/inst-.*@10.0" \\
        -o "membrane:skel/skel-{input_session_id}@10.0"

    \b
    # Use the distance-transform backend and keep short branches
    copick process skeletonize --method distance_transform --keep-short-branches \\
        -i "membrane:user1/inst-.*@10.0" -o "membrane:skel/skel-{input_session_id}@10.0"

    See Also:

    \b
    copick process separate-components: split a segmentation into the inst-* instances skeletonized here
    copick process filter-components: drop small connected components before skeletonizing
    """
    from copick_utils.process.skeletonize import skeletonize_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Create config from URIs with smart defaults
    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="segmentation",
            output_uri=output_uri,
            output_type="segmentation",
            command_name="skeletonize",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Log parameters
    input_params = parse_copick_uri(input_uri, "segmentation")
    logger.info(f"Skeletonizing segmentations '{input_params['name']}'")
    logger.info(f"Source segmentations: {input_params['user_id']} matching pattern '{input_params['session_id']}'")
    logger.info(f"Method: {method}")
    logger.info(f"Preprocessing: remove_noise={remove_noise} (min_size={min_object_size})")
    logger.info(f"Post-processing: remove_short_branches={remove_short_branches} (min_length={min_branch_length})")

    # Parallel discovery and processing
    results = skeletonize_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        method=method,
        remove_noise=remove_noise,
        min_object_size=min_object_size,
        remove_short_branches=remove_short_branches,
        min_branch_length=min_branch_length,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_skeletons = sum(result.get("skeletons_created", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total skeletons created: {total_skeletons}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
