"""CLI command for splitting multilabel segmentations into individual single-class segmentations."""

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
    short_help="Split multilabel segmentations into single-class segmentations.",
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
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="split")
@add_debug_option
def split(
    config,
    run_names,
    input_uri,
    workers,
    output_uri,
    debug,
):
    """
    Split multilabel segmentations into individual single-class binary segmentations.

    This command takes a multilabel segmentation and creates separate binary segmentations
    for each label value. Each output segmentation is named after the corresponding
    PickableObject (as defined in the copick config) and uses the same session ID as
    the input.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing
        Voxel spacing is optional — omit to match any voxel spacing.

    \b
    Label-to-Object Mapping:
        The tool looks up each label value in the pickable_objects configuration
        and uses the object name for the output segmentation:
        - Label 1 (ribosome) → ribosome:split/session-001@10.0
        - Label 2 (membrane) → membrane:split/session-001@10.0
        - Label 3 (proteasome) → proteasome:split/session-001@10.0

    \b
    Examples:
        # Split multilabel segmentation (outputs named by pickable objects)
        copick process split -i "predictions:model/run-001@10.0" -o ":split/0"

        # Split without specifying voxel spacing (matches any)
        copick process split -i "proofread:napari/manual" -o ":split-proofread/0"

        # Process specific runs only
        copick process split -i "labels:*/*@10.0" -o ":per-class/0" -r TS_001 -r TS_002
    """
    from copick_utils.process.split_labels import split_labels_lazy_batch

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
            command_name="split",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Log parameters
    input_params = parse_copick_uri(input_uri, "segmentation")
    logger.info(f"Splitting multilabel segmentation '{input_params['name']}'")

    # Parallel discovery and processing
    results = split_labels_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
    )

    # Aggregate results
    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_labels = sum(result.get("labels_split", 0) for result in results.values() if result)

    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total labels split: {total_labels}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
