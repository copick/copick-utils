"""CLI command for splitting multilabel segmentations into individual single-class segmentations."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import add_input_option, add_workers_option


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
@optgroup.group("\nTool Options", help="Options related to this tool.")
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@optgroup.option(
    "--output-user-id",
    type=str,
    default="split",
    help="User ID for output segmentations.",
)
@add_debug_option
def split(
    config,
    run_names,
    input_uri,
    workers,
    output_user_id,
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
        copick process split -i "predictions:model/run-001@10.0"

        # Split with custom output user ID
        copick process split -i "classes:annotator/manual@10.0" --output-user-id "per-class"

        # Process specific runs only
        copick process split -i "labels:*/*@10.0" --run-names TS_001 --run-names TS_002
    """

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Parse input URI
    try:
        input_params = parse_copick_uri(input_uri, "segmentation")
    except ValueError as e:
        raise click.BadParameter(f"Invalid input URI: {e}") from e

    segmentation_name = input_params["name"]
    segmentation_user_id = input_params["user_id"]
    segmentation_session_id = input_params["session_id"]
    voxel_spacing = input_params.get("voxel_spacing")

    if voxel_spacing is None or voxel_spacing == "*":
        raise click.BadParameter("Input URI must include a specific voxel spacing (e.g., @10.0)")

    # Check for patterns in critical fields
    if "*" in segmentation_name or "*" in segmentation_user_id or "*" in segmentation_session_id:
        raise click.BadParameter(
            "Input URI cannot contain wildcards for splitting. "
            "Please specify exact segmentation name, user_id, and session_id.",
        )

    logger.info(f"Splitting multilabel segmentation '{segmentation_name}'")
    logger.debug(f"Input: {segmentation_user_id}/{segmentation_session_id} @ {voxel_spacing}Å")
    logger.debug(f"Output user ID: {output_user_id}")
    logger.debug(f"Workers: {workers}")

    # Import batch function
    from copick_utils.process.split_labels import split_labels_batch

    # Process runs
    results = split_labels_batch(
        root=root,
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        segmentation_session_id=segmentation_session_id,
        voxel_spacing=float(voxel_spacing),
        output_user_id=output_user_id,
        run_names=run_names_list,
        workers=workers,
    )

    # Aggregate results
    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_labels = sum(result.get("labels_split", 0) for result in results.values() if result)

    # Collect all unique object names created
    all_object_names = set()
    for result in results.values():
        if result and result.get("object_names"):
            all_object_names.update(result["object_names"])

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total labels split: {total_labels}")
    logger.info(f"Object names created: {', '.join(sorted(all_object_names))}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
