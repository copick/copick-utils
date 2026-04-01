"""CLI command for analyzing connected component sizes in segmentations."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from copick_utils.cli.util import add_input_option, add_workers_option


@click.command(
    context_settings={"show_default": True},
    short_help="Analyze connected component sizes per label in segmentations.",
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
    "--connectivity",
    "-cn",
    type=click.Choice(["face", "face-edge", "all"]),
    default="all",
    help="Connectivity for connected components (face=6-connected, face-edge=18-connected, all=26-connected).",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to the output.")
@optgroup.option(
    "--output-format",
    "-f",
    type=click.Choice(["csv", "plot"]),
    default="csv",
    help="Output format: 'csv' for tabular data, 'plot' for a histogram.",
)
@optgroup.option(
    "--output-path",
    "-op",
    type=click.Path(),
    required=True,
    help="Output file path. For plots, format is inferred from extension (.html, .png, .pdf, .svg).",
)
@add_debug_option
def seg_stats(
    config,
    run_names,
    input_uri,
    connectivity,
    workers,
    output_format,
    output_path,
    debug,
):
    """
    Analyze connected component sizes per label in segmentations.

    This command identifies connected components in a segmentation and reports
    the volume of each component (in voxels and cubic angstroms), grouped by label.
    Output can be a CSV file or a histogram plot.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing
        Voxel spacing is optional — omit to match all voxel spacings.

    \b
    Examples:
        # Export component stats as CSV
        copick process seg-stats -i "membrane:user1/auto-001@10.0" -f csv -op ./membrane_stats.csv

        # Analyze without specifying voxel spacing (matches all)
        copick process seg-stats -i "proofread:napari/manual" -f csv -op ./stats.csv

        # Create a histogram plot as HTML (interactive)
        copick process seg-stats -i "membrane:user1/auto-001@10.0" -f plot -op ./membrane_hist.html

        # Create a histogram plot as PDF
        copick process seg-stats -i "organelle:user1/pred@10.0" -f plot -op ./organelle_hist.pdf

        # Analyze specific runs and export as PNG
        copick process seg-stats -i "membrane:user1/auto-001@10.0" -f plot -op ./stats.png -r run1 -r run2
    """
    from copick_utils.process.seg_stats import export_stats_csv, export_stats_plot, seg_stats_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    logger.info(f"Analyzing components for segmentation URI '{input_uri}'")
    logger.info(f"Connectivity: {connectivity}")
    logger.info(f"Output format: {output_format} -> {output_path}")

    # Process runs — URI resolution handled by resolve_copick_objects inside the worker
    results = seg_stats_batch(
        root=root,
        input_uri=input_uri,
        connectivity=connectivity,
        run_names=run_names_list,
        workers=workers,
    )

    # Summarize
    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_components = sum(len(result.get("components", [])) for result in results.values() if result)

    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Analyzed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total components found: {total_components}")

    # Export results
    if output_format == "csv":
        export_stats_csv(results, output_path)
    elif output_format == "plot":
        export_stats_plot(results, output_path, root=root)

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
