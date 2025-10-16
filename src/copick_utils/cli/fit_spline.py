import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import expand_output_uri, parse_copick_uri

from copick_utils.cli.util import add_input_option, add_output_option


@click.command(
    context_settings={"show_default": True},
    short_help="Fit 3D splines to skeletons and generate picks with orientations.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input segmentation.")
@optgroup.option(
    "--run-names",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_input_option("segmentation")
@optgroup.option(
    "--voxel-spacing",
    "-vs",
    type=float,
    required=True,
    help="Voxel spacing for coordinate scaling.",
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--spacing-distance",
    type=float,
    required=True,
    help="Distance between consecutive sampled points along the spline.",
)
@optgroup.option(
    "--smoothing-factor",
    type=float,
    help="Smoothing parameter for spline fitting (auto if not provided).",
)
@optgroup.option(
    "--degree",
    type=int,
    default=3,
    help="Degree of the spline (1-5).",
)
@optgroup.option(
    "--connectivity-radius",
    type=float,
    default=2.0,
    help="Maximum distance to consider skeleton points as connected.",
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
    help="Maximum allowed curvature before outlier removal.",
)
@optgroup.option(
    "--max-iterations",
    type=int,
    default=5,
    help="Maximum number of outlier removal iterations.",
)
@optgroup.option(
    "--workers",
    type=int,
    default=8,
    help="Number of worker processes.",
)
@optgroup.group("\nOutput Options", help="Options related to output picks.")
@add_output_option("picks", default_tool="spline")
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
    workers,
    output_uri,
    debug,
):
    """Fit 3D splines to skeletonized segmentations and generate picks with orientations.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing
        Picks: object_name:user_id/session_id

    \b
    This command fits regularized 3D parametric splines to skeleton volumes and samples
    points along the spline at regular intervals. Orientations are computed based on
    the spline direction.

    \b
    Examples:
        # Fit splines to skeletonized components
        copick process fit_spline -i "skeleton:skel/inst-.*@10.0" -o "skeleton:spline/spline-{input_session_id}" --spacing-distance 4.4 --voxel-spacing 10.0

        # Process specific skeleton
        copick process fit_spline -i "skeleton:skel/skel-0@10.0" -o "skeleton:spline/spline-0" --spacing-distance 2.0 --voxel-spacing 10.0
    """
    from copick_utils.process.spline_fitting import fit_spline_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Expand output URI with smart defaults
    try:
        output_uri = expand_output_uri(
            output_uri=output_uri,
            input_uri=input_uri,
            input_type="segmentation",
            output_type="picks",
            command_name="fit_spline",
            individual_outputs=False,
        )
    except ValueError as e:
        raise click.BadParameter(f"Error expanding output URI: {e}") from e

    # Parse input URI
    try:
        input_params = parse_copick_uri(input_uri, "segmentation")
    except ValueError as e:
        raise click.BadParameter(f"Invalid input URI: {e}") from e

    segmentation_name = input_params["name"]
    segmentation_user_id = input_params["user_id"]
    session_id_pattern = input_params["session_id"]

    # Parse output URI (now fully expanded)
    try:
        output_params = parse_copick_uri(output_uri, "picks")
    except ValueError as e:
        raise click.BadParameter(f"Invalid output URI: {e}") from e

    output_user_id = output_params["user_id"]
    output_session_id_template = output_params["session_id"]

    logger.info(f"Fitting splines to segmentations '{segmentation_name}'")
    logger.info(f"Source segmentations: {segmentation_user_id} matching pattern '{session_id_pattern}'")
    logger.info(f"Spacing distance: {spacing_distance}, degree: {degree}")
    logger.info(f"Smoothing factor: {smoothing_factor}, connectivity radius: {connectivity_radius}")
    logger.info(f"Compute transforms: {compute_transforms}, output user ID: {output_user_id}")
    logger.info(f"Curvature threshold: {curvature_threshold}, max iterations: {max_iterations}")
    logger.info(f"Voxel spacing: {voxel_spacing}")
    logger.info(f"Output session ID template: '{output_session_id_template}'")

    results = fit_spline_batch(
        root=root,
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        session_id_pattern=session_id_pattern,
        spacing_distance=spacing_distance,
        smoothing_factor=smoothing_factor,
        degree=degree,
        connectivity_radius=connectivity_radius,
        compute_transforms=compute_transforms,
        curvature_threshold=curvature_threshold,
        max_iterations=max_iterations,
        output_session_id_template=output_session_id_template,
        output_user_id=output_user_id,
        voxel_spacing=voxel_spacing,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_picks = sum(result.get("picks_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("segmentations_processed", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total segmentations processed: {total_processed}")
    logger.info(f"Total picks created: {total_picks}")
