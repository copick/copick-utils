"""CLI command for filtering connected components by size."""

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
    short_help="Filter connected components in segmentations by size.",
    no_args_is_help=True,
)
@add_config_option
@add_run_names_option
@optgroup.group("\nInput Options", help="Options related to the input segmentation.")
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
    help="Minimum component volume to keep (optional). Unit set by --size-unit.",
)
@optgroup.option(
    "--max-size",
    type=float,
    default=None,
    help="Maximum component volume to keep (optional). Unit set by --size-unit.",
)
@optgroup.option(
    "--size-unit",
    type=click.Choice(["angstrom", "voxel"]),
    default="angstrom",
    help="Unit for --min-size and --max-size: 'angstrom' for Å³, 'voxel' for cubic voxels.",
)
@optgroup.option(
    "--keep-largest",
    type=int,
    default=None,
    help="Keep only the N largest connected components by voxel count (e.g. 1 = keep only the "
    "single largest). Applied in addition to any --min-size/--max-size filter.",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="filter-components")
@add_debug_option
def filter_components(
    config,
    run_names,
    input_uri,
    connectivity,
    min_size,
    max_size,
    size_unit,
    keep_largest,
    workers,
    output_uri,
    debug,
):
    """
    Filter connected components in segmentations by size.

    This command identifies connected components in a segmentation and removes those that
    fall outside the specified size range. Sizes can be given in cubic angstroms (default)
    or cubic voxels via --size-unit, and component adjacency is controlled by --connectivity
    (face = 6-connected, face-edge = 18-connected, all = 26-connected).

    Pass --keep-largest N to retain only the N largest components by voxel count, applied on
    top of any --min-size/--max-size limits. Run `copick process seg-stats` first to inspect
    component sizes and choose sensible thresholds.

    URI Format:

        \b
        Segmentations: name:user_id/session_id@voxel_spacing

    Examples:

        \b
        # Remove small noise components (keep only larger than 50000 Å³)
        copick process filter-components -i "membrane:user1/auto-001@10.0" -o "membrane_clean" --min-size 50000

        \b
        # Filter by cubic voxels instead of angstroms
        copick process filter-components -i "membrane:user1/auto-001@10.0" -o "membrane_clean" \\
            --min-size 50 --size-unit voxel

        \b
        # Keep only medium-sized components (between 10000 and 1000000 Å³)
        copick process filter-components -i "particles:user1/.*@10.0" -o "particles_filtered" \\
            --min-size 10000 --max-size 1000000

        \b
        # Keep only the single largest connected component
        copick process filter-components -i "membrane:user1/auto-001@10.0" -o "membrane_main" --keep-largest 1

    See Also:

        \b
        copick process seg-stats: report connected-component size statistics to choose thresholds
        copick process separate-components: relabel each connected component as a distinct class
        copick logical enclosed: remove components fully enclosed by another segmentation
    """
    from copick_utils.process.filter_components import filter_components_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Convert voxel sizes to angstrom³ if needed
    if size_unit == "voxel":
        # Parse voxel spacing from URI for conversion
        input_params = parse_copick_uri(input_uri, "segmentation")
        vs_raw = input_params.get("voxel_spacing")
        if vs_raw is None or vs_raw == "*":
            raise click.BadParameter("--size-unit voxel requires voxel spacing in the input URI (e.g., @10.0)")
        voxel_volume = float(vs_raw) ** 3
        if min_size is not None:
            min_size = min_size * voxel_volume
        if max_size is not None:
            max_size = max_size * voxel_volume

    # Create config from URIs with smart defaults
    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="segmentation",
            output_uri=output_uri,
            output_type="segmentation",
            command_name="filter-components",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Log parameters
    input_params = parse_copick_uri(input_uri, "segmentation")
    logger.info(f"Filtering components for segmentation '{input_params['name']}'")
    logger.info(f"Source segmentation pattern: {input_params['user_id']}/{input_params['session_id']}")
    logger.info(f"Connectivity: {connectivity}")
    if min_size is not None:
        logger.info(f"Minimum size: {min_size} Å³")
    if max_size is not None:
        logger.info(f"Maximum size: {max_size} Å³")
    if keep_largest is not None:
        if keep_largest < 1:
            raise click.BadParameter("--keep-largest must be >= 1")
        logger.info(f"Keeping only the {keep_largest} largest component(s)")

    # Parallel discovery and processing
    results = filter_components_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        connectivity=connectivity,
        min_size=min_size,
        max_size=max_size,
        keep_largest=keep_largest,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_kept = sum(result.get("components_kept", 0) for result in results.values() if result)
    total_removed = sum(result.get("components_removed", 0) for result in results.values() if result)
    total_voxels = sum(result.get("voxels_kept", 0) for result in results.values() if result)

    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total components kept: {total_kept}")
    logger.info(f"Total components removed: {total_removed}")
    logger.info(f"Total voxels in filtered segmentations: {total_voxels}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
