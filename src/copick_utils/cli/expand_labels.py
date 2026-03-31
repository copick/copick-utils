"""CLI command for expanding labels in segmentations."""

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
    short_help="Expand labels in segmentations to fill holes and gaps.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input segmentation.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_input_option("segmentation")
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--distance",
    "-d",
    type=float,
    required=True,
    help="Distance in angstroms (Å) by which to expand labels.",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="expand-labels")
@add_debug_option
def expand_labels(
    config,
    run_names,
    input_uri,
    distance,
    workers,
    output_uri,
    debug,
):
    """
    Expand labels in segmentations to fill holes and gaps.

    Uses scikit-image's expand_labels to grow label regions outward by a specified
    distance without overlapping into neighboring regions. Useful for filling small
    holes in segmentations or closing gaps between label boundaries.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing

    \b
    Examples:
        # Expand labels by 20 angstroms
        copick process expand-labels -i "membrane:user1/auto-001@10.0" -o "membrane_filled" --distance 20.0

        # Expand labels across specific runs
        copick process expand-labels -i "organelle:user1/pred@10.0" -o "organelle_expanded" --distance 15.0 -r run1 -r run2

        # Expand with custom output URI
        copick process expand-labels -i "membrane:user1/manual@10.0" -o "membrane:expand-labels/0@10.0" --distance 30.0
    """
    from copick_utils.process.expand_labels import expand_labels_lazy_batch

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
            command_name="expand-labels",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Log parameters
    input_params = parse_copick_uri(input_uri, "segmentation")
    logger.info(f"Expanding labels for segmentation '{input_params['name']}'")
    logger.info(f"Source segmentation pattern: {input_params['user_id']}/{input_params['session_id']}")
    logger.info(f"Expansion distance: {distance} Å")

    # Parallel discovery and processing
    results = expand_labels_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        distance=distance,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_added = sum(result.get("voxels_added", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total voxels added across all runs: {total_added}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
