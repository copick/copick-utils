import click
import copick
from click_option_group import optgroup
from copick.cli.util import (
    add_config_option,
    add_debug_option,
    add_run_names_option,
    resolve_deprecated_option,
)
from copick.util.log import get_logger
from copick.util.uri import expand_output_uri, parse_copick_uri

from copick_utils.cli.util import add_input_option, add_output_option, add_workers_option
from copick_utils.util.config_models import create_simple_config


@click.command(
    context_settings={"show_default": True},
    short_help="Fit 3D splines to skeletons and generate oriented picks.",
    no_args_is_help=True,
)
@add_config_option
@add_run_names_option
@optgroup.group("\nInput Options", help="Options related to the input segmentation.")
@add_input_option("segmentation")
# TODO:remove once deprecation takes effect -- legacy --voxel-spacing/-vs override (vs now comes from the input URI)
@optgroup.option(
    "--voxel-spacing",
    "-vs",
    type=float,
    required=False,
    default=None,
    hidden=True,
    help="Deprecated: the @voxel_spacing in -i/--input is used instead.",
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--spacing-distance",
    type=float,
    required=True,
    help="Distance between consecutive sampled points along each spline, in voxels.",
)
@optgroup.option(
    "--smoothing-factor",
    type=float,
    help="Smoothing parameter for spline fitting (scipy's s, in voxels squared; auto if not provided).",
)
@optgroup.option(
    "--degree",
    type=int,
    default=3,
    help="Degree of the spline (1-5).",
)
# TODO:remove once deprecation takes effect -- --connectivity-radius has no effect (26-connected skeleton graph)
@optgroup.option(
    "--connectivity-radius",
    type=float,
    default=None,
    hidden=True,
    help="Deprecated and ignored: skeleton voxels are joined to their 26 neighbours.",
)
@optgroup.option(
    "--compute-transforms/--no-compute-transforms",
    is_flag=True,
    default=True,
    help="Whether to compute orientations for picks.",
)
@optgroup.option(
    "--curvature-threshold",
    type=float,
    default=0.2,
    help="Largest sine of the turning angle between consecutive sampled points; a spline that turns more "
    "sharply is smoothed further.",
)
@optgroup.option(
    "--max-iterations",
    type=int,
    default=5,
    help="Maximum number of smoothing increases.",
)
@optgroup.option(
    "--label",
    type=int,
    default=None,
    help="Label to fit in a multilabel segmentation (default: every non-zero voxel).",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output picks.")
@add_output_option("picks", default_tool="spline")
@add_output_option(
    "filaments",
    flag="--filaments",
    short_flag="-of",
    param_name="filaments_uri",
    required=False,
    description="Also store the fitted splines as copick Filaments (exact B-spline curves) under this URI.",
)
@add_debug_option
def fit_spline(
    config,
    run_names,
    input_uri,
    voxel_spacing,
    spacing_distance,
    smoothing_factor,
    degree,
    connectivity_radius,
    compute_transforms,
    curvature_threshold,
    max_iterations,
    label,
    workers,
    output_uri,
    filaments_uri,
    debug,
):
    """Fit 3D splines to skeletons and generate oriented picks.

    Fits a smoothing 3D spline to every filament of a skeleton (or segmentation) volume and
    samples points along each spline exactly `--spacing-distance` voxels apart, producing picks.
    The skeleton is split into filaments between ends and junctions, continuing straight
    through junctions, so crossing or separate filaments each get their own spline. All picks go
    into one pick set: grouped by filament, in order along it, with the filament's number
    (1, 2, ... by length) as the pick's instance ID. With `--compute-transforms`, each pick's
    +Z axis follows the spline's direction, and the rotation about it changes as little as
    possible along the filament.

    Where a spline turns more sharply than `--curvature-threshold` between samples, its
    smoothing is increased (up to `--max-iterations` times). `--filaments` also stores the
    fitted splines as copick Filaments.

    For new work, `copick convert seg2fil` (tracing, with an instance segmentation and
    thickness-based filters) and `copick convert fil2picks` (sampling in angstroms) separate
    tracing from sampling.

    URI Format:

        \b
        Segmentations: name:user_id/session_id@voxel_spacing
        Picks: object_name:user_id/session_id

    Examples:

        \b
        # Fit splines to skeletonized components (voxel spacing from the @10.0 in -i)
        copick process fit-spline -i "skeleton:skel/inst-.*@10.0" \\
            -o "skeleton:spline/spline-{input_session_id}" --spacing-distance 4.4

        \b
        # Process a single skeleton component
        copick process fit-spline -i "skeleton:skel/skel-0@10.0" \\
            -o "skeleton:spline/spline-0" --spacing-distance 2.0

    See Also:

        \b
        copick convert seg2fil: trace filaments in a segmentation (Filaments and an instance segmentation)
        copick convert fil2picks: sample picks along filaments, in angstroms
        copick process skeletonize: produce the skeleton segmentations fed to this command
    """
    from copick_utils.process.spline_fitting import fit_spline_lazy_batch

    logger = get_logger(__name__, debug=debug)
    logger.info("fit-spline is kept for compatibility; see copick convert seg2fil and fil2picks for new work")
    if connectivity_radius is not None:
        logger.warning("--connectivity-radius is deprecated and has no effect")

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Create config directly from URIs with smart defaults
    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="segmentation",
            output_uri=output_uri,
            output_type="picks",
            command_name="spline",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Extract parameters for logging only
    input_params = parse_copick_uri(input_uri, "segmentation")

    # Voxel spacing comes from the @voxel_spacing in the input URI; the legacy
    # --voxel-spacing/-vs flag remains a deprecated override.
    uri_vs_raw = input_params.get("voxel_spacing")
    uri_vs = float(uri_vs_raw) if uri_vs_raw not in (None, "*") else None
    # TODO:remove once deprecation takes effect -- legacy -vs override (drop the voxel_spacing param; use uri_vs directly)
    voxel_spacing = resolve_deprecated_option(
        uri_vs,
        voxel_spacing,
        old_flag="--voxel-spacing/-vs",
        new_flag="the @voxel_spacing in -i/--input",
        logger=logger,
    )
    if voxel_spacing is None:
        raise click.BadParameter(
            "Input URI must include a specific voxel spacing (e.g., @10.0), or pass --voxel-spacing/-vs.",
        )

    filament_params = {}
    if filaments_uri:
        parsed = parse_copick_uri(
            expand_output_uri(
                output_uri=filaments_uri,
                input_uri=input_uri,
                input_type="segmentation",
                output_type="filaments",
                command_name="spline",
            ),
            "filaments",
        )
        filament_params = {
            "filaments_object_name": parsed["object_name"],
            "filaments_user_id": parsed["user_id"],
            "filaments_session_id": parsed["session_id"],
        }

    logger.info(f"Fitting splines to segmentations '{input_params['name']}'")
    logger.info(
        f"Source segmentation pattern: {input_params['name']} ({input_params['user_id']}/{input_params['session_id']})",
    )
    logger.info(f"Spacing distance: {spacing_distance}, degree: {degree}")
    logger.info(f"Smoothing factor: {smoothing_factor}")
    logger.info(f"Compute transforms: {compute_transforms}")
    logger.info(f"Curvature threshold: {curvature_threshold}, max iterations: {max_iterations}")
    logger.info(f"Voxel spacing: {voxel_spacing}")

    # Parallel discovery and processing - no sequential bottleneck!
    results = fit_spline_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        # Tool-specific kwargs passed to converter via converter_kwargs
        spacing_distance=spacing_distance,
        smoothing_factor=smoothing_factor,
        degree=degree,
        compute_transforms=compute_transforms,
        curvature_threshold=curvature_threshold,
        max_iterations=max_iterations,
        voxel_spacing=voxel_spacing,
        label=label,
        **filament_params,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_picks = sum(result.get("picks_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("processed", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total conversion tasks completed: {total_processed}")
    total_filaments = sum(result.get("filaments_fitted", 0) for result in results.values() if result)
    logger.info(f"Total picks created: {total_picks} on {total_filaments} filaments")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
