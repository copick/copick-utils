"""CLI command for combining single-label segmentations into a multilabel segmentation."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import add_input_option, add_output_option, add_workers_option
from copick_utils.util.config_models import create_single_selector_config


@click.command(
    context_settings={"show_default": True},
    short_help="Combine single-label segmentations into a multilabel segmentation.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input segmentations.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_input_option("segmentation")
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to the output segmentation.")
@add_output_option("segmentation", default_tool="combine")
@add_debug_option
def combine(
    config,
    run_names,
    input_uri,
    workers,
    output_uri,
    debug,
):
    """
    Combine single-label segmentations into one multilabel segmentation.

    This is the inverse of `copick process split`. Takes multiple binary/single-label
    segmentations (matched by a pattern) and merges them into a single multilabel
    volume. Each input segmentation's name is looked up in the copick config to
    determine its integer label value.

    \b
    Overlap Resolution:
        When multiple inputs overlap, the lowest label value wins. This is
        deterministic and reproducible. Overlaps are logged as warnings.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing
        Use glob/regex patterns to match multiple segmentations per run.

    \b
    Examples:
        # Combine all segmentations from a split operation
        copick process combine -i "*:split/*@20" -o "multilabel:combine/0"

        # Combine specific user's segmentations using regex
        copick process combine -i "re:.*:napari/manual@20" -o "combined:combine/0"

        # Combine for specific runs
        copick process combine -r run1 -r run2 -i "*:user1/session@10.0" -o "labels:combine/0"
    """
    from copick_utils.process.combine_labels import combine_labels_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Create config for single-pattern → N-way operation
    try:
        task_config = create_single_selector_config(
            input_uri=input_uri,
            input_type="segmentation",
            output_uri=output_uri,
            output_type="segmentation",
            command_name="combine",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Log parameters
    input_params = parse_copick_uri(input_uri, "segmentation")
    logger.info(
        f"Combining segmentations matching '{input_params['name']}:{input_params['user_id']}/{input_params['session_id']}'",
    )

    # Parallel discovery and processing
    results = combine_labels_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_labels = sum(result.get("labels_combined", 0) for result in results.values() if result)
    total_overlaps = sum(result.get("overlapping_voxels", 0) for result in results.values() if result)

    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total labels combined: {total_labels}")
    if total_overlaps > 0:
        logger.warning(f"Total overlapping voxels: {total_overlaps}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
