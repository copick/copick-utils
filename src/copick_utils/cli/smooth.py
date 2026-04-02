"""CLI commands for smoothing segmentations."""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger
from copick.util.uri import parse_copick_uri

from copick_utils.cli.util import add_input_option, add_output_option, add_workers_option
from copick_utils.util.config_models import create_simple_config


@click.group(
    short_help="Smooth segmentations using mesh-based methods.",
    no_args_is_help=True,
)
def smooth():
    """Smooth segmentations using mesh-based methods.

    Available methods:

    \b
      mesh      - Taubin mesh smoothing (marching cubes → smooth → re-rasterize)
      membrane  - Dilate-smooth-thin for thin membrane segmentations
    """
    pass


# ── mesh subcommand ───────────────────────────────────────────────────────────


@smooth.command(
    context_settings={"show_default": True},
    short_help="Taubin mesh smoothing for segmentations.",
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
@optgroup.group("\nTool Options", help="Options related to Taubin mesh smoothing.")
@optgroup.option(
    "--taubin-lambda",
    type=float,
    default=0.5,
    help="Positive shrink factor (0 < lambda < 1). Controls smoothing strength.",
)
@optgroup.option(
    "--taubin-nu",
    type=float,
    default=-0.53,
    help="Negative inflate factor (nu < -lambda). Prevents shrinkage.",
)
@optgroup.option(
    "--iterations",
    type=int,
    default=10,
    help="Number of Taubin smoothing iterations.",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="smooth-mesh")
@add_debug_option
def mesh(
    config,
    run_names,
    input_uri,
    taubin_lambda,
    taubin_nu,
    iterations,
    workers,
    output_uri,
    debug,
):
    """Smooth segmentations using Taubin mesh smoothing.

    Extracts a surface mesh via marching cubes, applies Taubin lambda/nu
    smoothing (tangential, volume-preserving), then re-rasterizes onto
    the original voxel grid.

    Only single-label (binary) segmentations are supported.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing

    \b
    Examples:
        # Smooth with default parameters
        copick process smooth mesh -i "membrane:user1/auto-001@10.0" -o "membrane_smooth"

        # Stronger smoothing with more iterations
        copick process smooth mesh -i "membrane:user1/auto-001@10.0" -o "smooth" --iterations 30

        # Custom Taubin parameters
        copick process smooth mesh -i "seg:user1/pred@10.0" -o "smooth" --taubin-lambda 0.6 --taubin-nu -0.63

        # Process specific runs
        copick process smooth mesh -i "membrane:user1/auto@10.0" -o "smooth" -r run1 -r run2
    """
    from copick_utils.process.smooth import smooth_mesh_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="segmentation",
            output_uri=output_uri,
            output_type="segmentation",
            command_name="smooth-mesh",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    input_params = parse_copick_uri(input_uri, "segmentation")
    logger.info(f"Smoothing segmentation '{input_params['name']}' (method: mesh)")
    logger.info(f"Taubin parameters: lambda={taubin_lambda}, nu={taubin_nu}, iterations={iterations}")

    results = smooth_mesh_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        taubin_lambda=taubin_lambda,
        taubin_nu=taubin_nu,
        iterations=iterations,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_changed = sum(result.get("voxels_changed", 0) for result in results.values() if result)

    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total voxels changed across all runs: {total_changed}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")


# ── membrane subcommand ───────────────────────────────────────────────────────


@smooth.command(
    context_settings={"show_default": True},
    short_help="Dilate-smooth-thin for thin membrane segmentations.",
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
@optgroup.group("\nTool Options", help="Options related to membrane smoothing.")
@optgroup.option(
    "--dilation-voxels",
    type=int,
    default=3,
    help="Number of voxels to dilate before smoothing. Ensures marching cubes sees a closed surface.",
)
@optgroup.option(
    "--taubin-lambda",
    type=float,
    default=0.5,
    help="Positive shrink factor (0 < lambda < 1). Controls smoothing strength.",
)
@optgroup.option(
    "--taubin-nu",
    type=float,
    default=-0.53,
    help="Negative inflate factor (nu < -lambda). Prevents shrinkage.",
)
@optgroup.option(
    "--iterations",
    type=int,
    default=10,
    help="Number of Taubin smoothing iterations.",
)
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output segmentations.")
@add_output_option("segmentation", default_tool="smooth-membrane")
@add_debug_option
def membrane(
    config,
    run_names,
    input_uri,
    dilation_voxels,
    taubin_lambda,
    taubin_nu,
    iterations,
    workers,
    output_uri,
    debug,
):
    """Smooth thin membrane segmentations using dilate-smooth-thin.

    For thin (~1-3 voxel) membrane segmentations: dilates the membrane
    to create a solid volume, applies Taubin mesh smoothing, then re-thins
    back to the original membrane thickness.

    Only single-label (binary) segmentations are supported.

    \b
    URI Format:
        Segmentations: name:user_id/session_id@voxel_spacing

    \b
    Examples:
        # Smooth a membrane segmentation with default parameters
        copick process smooth membrane -i "membrane:user1/auto-001@10.0" -o "membrane_smooth"

        # Increase dilation for thicker membranes
        copick process smooth membrane -i "membrane:user1/auto@10.0" -o "smooth" --dilation-voxels 5

        # Combine dilation and iteration tuning
        copick process smooth membrane -i "membrane:user1/auto@10.0" -o "smooth" --dilation-voxels 4 --iterations 20

        # Process specific runs
        copick process smooth membrane -i "membrane:user1/auto@10.0" -o "smooth" -r run1 -r run2
    """
    from copick_utils.process.smooth import smooth_membrane_lazy_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    try:
        task_config = create_simple_config(
            input_uri=input_uri,
            input_type="segmentation",
            output_uri=output_uri,
            output_type="segmentation",
            command_name="smooth-membrane",
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    input_params = parse_copick_uri(input_uri, "segmentation")
    logger.info(f"Smoothing segmentation '{input_params['name']}' (method: membrane)")
    logger.info(
        f"Parameters: dilation_voxels={dilation_voxels}, "
        f"taubin_lambda={taubin_lambda}, taubin_nu={taubin_nu}, iterations={iterations}",
    )

    results = smooth_membrane_lazy_batch(
        root=root,
        config=task_config,
        run_names=run_names_list,
        workers=workers,
        taubin_lambda=taubin_lambda,
        taubin_nu=taubin_nu,
        iterations=iterations,
        dilation_voxels=dilation_voxels,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_changed = sum(result.get("voxels_changed", 0) for result in results.values() if result)

    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total voxels changed across all runs: {total_changed}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
