"""CLI commands for segmentation processing operations."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger


@click.command(
    context_settings={"show_default": True},
    short_help="Separate connected components in segmentations.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input segmentation.")
@optgroup.option(
    "--segmentation-name",
    required=True,
    help="Name of the segmentation to process.",
)
@optgroup.option(
    "--segmentation-user-id",
    required=True,
    help="User ID of the segmentation to process.",
)
@optgroup.option(
    "--segmentation-session-id",
    required=True,
    help="Session ID of the segmentation to process.",
)
@optgroup.option(
    "--run-names",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--connectivity",
    type=click.Choice(["6", "18", "26"]),
    default="26",
    help="Connectivity for connected components analysis (6, 18, or 26).",
)
@optgroup.option(
    "--min-size",
    type=int,
    default=0,
    help="Minimum size of components to keep (in voxels).",
)
@optgroup.option(
    "--multilabel/--binary",
    is_flag=True,
    default=True,
    help="Process as multilabel segmentation (analyze each label separately).",
)
@optgroup.option(
    "--workers",
    type=int,
    default=8,
    help="Number of worker processes.",
)
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@optgroup.option(
    "--session-id-prefix",
    default="inst-",
    help="Prefix for output segmentation session IDs.",
)
@optgroup.option(
    "--output-user-id",
    default="components",
    help="User ID for output segmentations.",
)
@add_debug_option
def separate_components(
    config,
    segmentation_name,
    segmentation_user_id,
    segmentation_session_id,
    run_names,
    connectivity,
    min_size,
    multilabel,
    workers,
    session_id_prefix,
    output_user_id,
    debug,
):
    """Separate connected components in segmentations into individual segmentations.

    For multilabel segmentations, connected components analysis is performed on each
    label separately. Output segmentations are created with session IDs using the
    specified prefix and incremental numbering (e.g., "inst-0", "inst-1", etc.).
    """
    from copick_utils.process.connected_components import separate_components_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None
    connectivity_int = int(connectivity)

    logger.info(f"Separating connected components for segmentation '{segmentation_name}'")
    logger.info(f"Source segmentation: {segmentation_user_id}/{segmentation_session_id}")
    logger.info(f"Output prefix: {session_id_prefix}, user ID: {output_user_id}")
    logger.info(f"Connectivity: {connectivity_int}, min size: {min_size} voxels")
    logger.info(f"Processing as {'multilabel' if multilabel else 'binary'} segmentation")

    if run_names_list:
        logger.info(f"Processing {len(run_names_list)} specific runs")
    else:
        logger.info(f"Processing all {len(root.runs)} runs")

    results = separate_components_batch(
        root=root,
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        segmentation_session_id=segmentation_session_id,
        connectivity=connectivity_int,
        min_size=min_size,
        session_id_prefix=session_id_prefix,
        output_user_id=output_user_id,
        multilabel=multilabel,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_components = sum(result.get("components_created", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total components created: {total_components}")


@click.command(
    context_settings={"show_default": True},
    short_help="3D skeletonization of segmentations.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input segmentation.")
@optgroup.option(
    "--run-names",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@optgroup.option(
    "--segmentation-name",
    required=True,
    help="Name of the segmentations to process.",
)
@optgroup.option(
    "--segmentation-user-id",
    required=True,
    help="User ID of the segmentations to process.",
)
@optgroup.option(
    "--session-id-pattern",
    required=True,
    help="Session ID pattern (regex) or exact session ID to match segmentations.",
)
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
@optgroup.option(
    "--workers",
    type=int,
    default=8,
    help="Number of worker processes.",
)
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@optgroup.option(
    "--output-session-id-template",
    help="Template for output session IDs. Use {input_session_id} as placeholder. Default: same as input.",
)
@optgroup.option(
    "--output-user-id",
    default="skel",
    help="User ID for output segmentations.",
)
@add_debug_option
def skeletonize(
    config,
    run_names,
    segmentation_name,
    segmentation_user_id,
    session_id_pattern,
    method,
    remove_noise,
    min_object_size,
    remove_short_branches,
    min_branch_length,
    workers,
    output_session_id_template,
    output_user_id,
    debug,
):
    """3D skeletonization of segmentations using regex pattern matching.

    This command can process multiple segmentations by matching session IDs against
    a regex pattern. This is useful for processing the output of connected components
    separation (e.g., pattern "inst-.*" to match "inst-0", "inst-1", etc.).

    Examples:
    - Exact match: --session-id-pattern "inst-0"
    - Regex pattern: --session-id-pattern "inst-.*"
    - Regex pattern: --session-id-pattern "inst-[0-9]+"
    """
    from copick_utils.process.skeletonize import skeletonize_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    logger.info(f"Skeletonizing segmentations '{segmentation_name}'")
    logger.info(f"Source segmentations: {segmentation_user_id} matching pattern '{session_id_pattern}'")
    logger.info(f"Method: {method}, output user ID: {output_user_id}")
    logger.info(f"Preprocessing: remove_noise={remove_noise} (min_size={min_object_size})")
    logger.info(f"Post-processing: remove_short_branches={remove_short_branches} (min_length={min_branch_length})")

    if output_session_id_template:
        logger.info(f"Output session ID template: '{output_session_id_template}'")
    else:
        logger.info("Output session IDs: same as input")

    if run_names_list:
        logger.info(f"Processing {len(run_names_list)} specific runs")
    else:
        logger.info(f"Processing all {len(root.runs)} runs")

    results = skeletonize_batch(
        root=root,
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        session_id_pattern=session_id_pattern,
        method=method,
        remove_noise=remove_noise,
        min_object_size=min_object_size,
        remove_short_branches=remove_short_branches,
        min_branch_length=min_branch_length,
        output_session_id_template=output_session_id_template,
        output_user_id=output_user_id,
        run_names=run_names_list,
        workers=workers,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_skeletons = sum(result.get("skeletons_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("segmentations_processed", 0) for result in results.values() if result)

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total segmentations processed: {total_processed}")
    logger.info(f"Total skeletons created: {total_skeletons}")


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
@optgroup.option(
    "--segmentation-name",
    required=True,
    help="Name of the segmentations to process.",
)
@optgroup.option(
    "--segmentation-user-id",
    required=True,
    help="User ID of the segmentations to process.",
)
@optgroup.option(
    "--session-id-pattern",
    required=True,
    help="Session ID pattern (regex) or exact session ID to match segmentations.",
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
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@optgroup.option(
    "--output-session-id-template",
    help="Template for output session IDs. Use {input_session_id} as placeholder. Default: same as input.",
)
@optgroup.option(
    "--output-user-id",
    default="spline",
    help="User ID for output picks.",
)
@optgroup.option(
    "--voxel-spacing",
    type=float,
    required=True,
    help="Voxel spacing for coordinate scaling.",
)
@add_debug_option
def fit_spline(
    config,
    run_names,
    segmentation_name,
    segmentation_user_id,
    session_id_pattern,
    spacing_distance,
    smoothing_factor,
    degree,
    connectivity_radius,
    compute_transforms,
    curvature_threshold,
    max_iterations,
    workers,
    output_session_id_template,
    output_user_id,
    voxel_spacing,
    debug,
):
    """Fit 3D splines to skeletonized segmentations and generate picks with orientations.

    This command fits regularized 3D parametric splines to skeleton volumes and samples
    points along the spline at regular intervals. Orientations are computed based on
    the spline direction.

    This is designed to work with the output of the skeletonize command, using regex
    pattern matching to process multiple skeletons in batch.

    Examples:
    - Process skeletonized components: --session-id-pattern "inst-.*" --spacing-distance 4.4
    - Process specific skeleton: --session-id-pattern "skel-0" --spacing-distance 2.0
    - Custom output naming: --output-session-id-template "spline-{input_session_id}"
    """
    from copick_utils.process.spline_fitting import fit_spline_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    logger.info(f"Fitting splines to segmentations '{segmentation_name}'")
    logger.info(f"Source segmentations: {segmentation_user_id} matching pattern '{session_id_pattern}'")
    logger.info(f"Spacing distance: {spacing_distance}, degree: {degree}")
    logger.info(f"Smoothing factor: {smoothing_factor}, connectivity radius: {connectivity_radius}")
    logger.info(f"Compute transforms: {compute_transforms}, output user ID: {output_user_id}")
    logger.info(f"Curvature threshold: {curvature_threshold}, max iterations: {max_iterations}")
    logger.info(f"Voxel spacing: {voxel_spacing}")

    if output_session_id_template:
        logger.info(f"Output session ID template: '{output_session_id_template}'")
    else:
        logger.info("Output session IDs: same as input")

    if run_names_list:
        logger.info(f"Processing {len(run_names_list)} specific runs")
    else:
        logger.info(f"Processing all {len(root.runs)} runs")

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
