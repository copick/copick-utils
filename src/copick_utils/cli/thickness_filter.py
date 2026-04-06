"""CLI command for filtering thin regions from segmentations by thickness."""

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
    short_help="Remove thin regions from segmentations.",
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
    "--min-thickness",
    "-t",
    type=float,
    required=True,
    help="Minimum thickness (diameter) of structures to keep. Unit set by --thickness-unit.",
)
@optgroup.option(
    "--thickness-unit",
    type=click.Choice(["angstrom", "voxel"]),
    default="angstrom",
    help="Unit for --min-thickness: 'angstrom' for Å, 'voxel' for voxel diameters.",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="thickness-filter")
@add_debug_option
def thickness_filter(
    config,
    run_names,
    input_uri,
    min_thickness,
    thickness_unit,
    workers,
    output_uri,
    debug,
):
    """
    Remove thin regions from segmentations based on inscribed sphere diameter.

    This command identifies thin sheet-like artifacts in a segmentation and removes
    them. A region is kept only if it contains at least one voxel whose largest
    inscribed sphere has a diameter >= min-thickness. The algorithm computes the
    Euclidean distance transform, thresholds to find thick cores, dilates them back,
    and intersects with the original mask.

    For multilabel segmentations, each label is filtered independently.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing

    \b
    Examples:
        # Remove membrane regions thinner than 50 Å
        copick process thickness-filter -i "membrane:user1/auto-001@10.0" -o "membrane_thick" --min-thickness 50.0

        # Filter by voxel diameters instead of angstroms
        copick process thickness-filter -i "membrane:user1/auto-001@10.0" -o "membrane_thick" --min-thickness 5 --thickness-unit voxel

        # Process specific runs with regex pattern
        copick process thickness-filter -i "membrane:user1/.*@10.0" -o "membrane_thick" --min-thickness 40.0 -r run001 -r run002
    """
    from copick_utils.process.thickness_filter import thickness_filter_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Convert voxel units to angstroms if needed (linear, not cubic)
    if thickness_unit == "voxel":
        input_params = parse_copick_uri(input_uri, "segmentation")
        vs_raw = input_params.get("voxel_spacing")
        if vs_raw is None or vs_raw == "*":
            raise click.BadParameter("--thickness-unit voxel requires voxel spacing in the input URI (e.g., @10.0)")
        min_thickness = min_thickness * float(vs_raw)

    # Create config from URIs with smart defaults
    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="segmentation",
            output_uri=output_uri,
            output_type="segmentation",
            command_name="thickness-filter",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Log parameters
    input_params = parse_copick_uri(input_uri, "segmentation")
    logger.info(f"Filtering thin regions from segmentation '{input_params['name']}'")
    logger.info(f"Source segmentation pattern: {input_params['user_id']}/{input_params['session_id']}")
    logger.info(f"Minimum thickness: {min_thickness:.1f} Å")

    # Parallel discovery and processing
    results = thickness_filter_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        min_thickness=min_thickness,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_before = sum(result.get("voxels_before", 0) for result in results.values() if result)
    total_after = sum(result.get("voxels_after", 0) for result in results.values() if result)
    total_removed = sum(result.get("voxels_removed", 0) for result in results.values() if result)

    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total voxels before: {total_before}")
    logger.info(f"Total voxels after: {total_after}")
    logger.info(f"Total voxels removed: {total_removed}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
