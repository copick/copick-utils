"""CLI utilities for copick-utils commands."""

from typing import Callable

import click
from click_option_group import optgroup


def add_clustering_options(func: click.Command) -> click.Command:
    """
    Add common clustering options for picks-to-mesh conversion commands.

    Args:
        func (click.Command): The Click command to which the options will be added.

    Returns:
        click.Command: The Click command with the clustering options added.
    """
    opts = [
        optgroup.option(
            "--use-clustering/--no-use-clustering",
            "-cl",
            is_flag=True,
            default=False,
            help="Cluster points before mesh creation.",
        ),
        optgroup.option(
            "--clustering-method",
            type=click.Choice(["dbscan", "kmeans"]),
            default="dbscan",
            help="Clustering method.",
        ),
        optgroup.option(
            "--clustering-eps",
            type=float,
            default=1.0,
            help="DBSCAN eps parameter - maximum distance between points in a cluster (in angstroms).",
        ),
        optgroup.option(
            "--clustering-min-samples",
            type=int,
            default=3,
            help="DBSCAN min_samples parameter.",
        ),
        optgroup.option(
            "--clustering-n-clusters",
            type=int,
            default=1,
            help="K-means n_clusters parameter.",
        ),
        optgroup.option(
            "--all-clusters/--largest-cluster-only",
            "-mm",
            is_flag=True,
            default=True,
            help="Use all clusters (True) or only the largest cluster (False).",
        ),
    ]

    for opt in reversed(opts):
        func = opt(func)

    return func


def add_workers_option(func: click.Command) -> click.Command:
    """
    Add workers option for parallel processing.

    Args:
        func (click.Command): The Click command to which the option will be added.

    Returns:
        click.Command: The Click command with the workers option added.
    """
    opts = [
        optgroup.option(
            "--workers",
            "-w",
            type=int,
            default=8,
            help="Number of worker processes.",
        ),
    ]

    for opt in opts:
        func = opt(func)

    return func


def add_marching_cubes_options(func: click.Command) -> click.Command:
    """
    Add marching cubes options for segmentation-to-mesh conversion commands.

    Args:
        func (click.Command): The Click command to which the options will be added.

    Returns:
        click.Command: The Click command with the marching cubes options added.
    """
    opts = [
        optgroup.option(
            "--level",
            type=float,
            default=0.5,
            help="Isosurface level for marching cubes.",
        ),
        optgroup.option(
            "--step-size",
            type=int,
            default=1,
            help="Step size for marching cubes (higher = coarser mesh).",
        ),
    ]

    for opt in reversed(opts):
        func = opt(func)

    return func


def add_segmentation_processing_options(func: click.Command) -> click.Command:
    """
    Add segmentation processing options for segmentation-to-picks conversion commands.

    Args:
        func (click.Command): The Click command to which the options will be added.

    Returns:
        click.Command: The Click command with the segmentation processing options added.
    """
    opts = [
        optgroup.option(
            "--segmentation-idx",
            "-si",
            type=int,
            required=True,
            help="Label index to extract from segmentation.",
        ),
        optgroup.option(
            "--maxima-filter-size",
            type=int,
            default=9,
            help="Size of maximum detection filter.",
        ),
        optgroup.option(
            "--min-particle-size",
            type=int,
            default=1000,
            help="Minimum particle size threshold.",
        ),
        optgroup.option(
            "--max-particle-size",
            type=int,
            default=50000,
            help="Maximum particle size threshold.",
        ),
    ]

    for opt in reversed(opts):
        func = opt(func)

    return func


def add_mesh_voxelization_options(func: click.Command) -> click.Command:
    """
    Add mesh voxelization options for mesh-to-segmentation conversion commands.

    Args:
        func (click.Command): The Click command to which the options will be added.

    Returns:
        click.Command: The Click command with the mesh voxelization options added.
    """
    opts = [
        optgroup.option(
            "--mode",
            type=click.Choice(["watertight", "boundary"]),
            default="watertight",
            help="Voxelization mode: 'watertight' fills the entire mesh interior, 'boundary' only voxelizes the surface.",
        ),
        optgroup.option(
            "--boundary-sampling-density",
            type=float,
            default=1.0,
            help="Surface sampling density for boundary mode (samples per voxel edge length).",
        ),
        optgroup.option(
            "--invert/--no-invert",
            is_flag=True,
            default=False,
            help="Invert the volume (fill outside instead of inside for watertight mode).",
        ),
    ]

    for opt in reversed(opts):
        func = opt(func)

    return func


def add_picks_painting_options(func: click.Command) -> click.Command:
    """
    Add picks painting options for picks-to-segmentation conversion commands.

    Args:
        func (click.Command): The Click command to which the options will be added.

    Returns:
        click.Command: The Click command with the picks painting options added.
    """
    opts = [
        optgroup.option(
            "--radius",
            type=float,
            default=10.0,
            help="Radius of spheres to paint at pick locations (in angstroms).",
        ),
    ]

    for opt in reversed(opts):
        func = opt(func)

    return func


def add_boolean_operation_option(func: click.Command) -> click.Command:
    """
    Add boolean operation option for logical operation commands.

    Args:
        func (click.Command): The Click command to which the option will be added.

    Returns:
        click.Command: The Click command with the boolean operation option added.
    """
    opts = [
        optgroup.option(
            "--operation",
            "-op",
            type=click.Choice(["union", "difference", "intersection", "exclusion", "concatenate"]),
            required=True,
            help="Boolean operation to perform.",
        ),
    ]

    for opt in opts:
        func = opt(func)

    return func


def add_distance_options(func: click.Command) -> click.Command:
    """
    Add distance-related options for distance-based logical operations.

    Args:
        func (click.Command): The Click command to which the options will be added.

    Returns:
        click.Command: The Click command with the distance options added.
    """
    opts = [
        optgroup.option(
            "--max-distance",
            "-d",
            type=float,
            default=100.0,
            help="Maximum distance from reference surface (in angstroms).",
        ),
        optgroup.option(
            "--mesh-voxel-spacing",
            "-mvs",
            type=float,
            help="Voxel spacing for mesh voxelization when using mesh reference (defaults to target voxel spacing).",
        ),
    ]

    for opt in reversed(opts):
        func = opt(func)

    return func


# ============================================================================
# URI-Based Option Decorators (New Simplified Interface)
# ============================================================================


def add_input_option(object_type: str, func: click.Command = None) -> Callable:
    """
    Add --input/-i option for URI-based input selection.

    Supports copick URI format with pattern matching:
    - Picks/Meshes: object_name:user_id/session_id
    - Segmentations: name:user_id/session_id@voxel_spacing?multilabel=true

    Args:
        object_type (str): Type of object ('picks', 'mesh', 'segmentation').
        func (click.Command, optional): The Click command to which the option will be added.

    Returns:
        Callable: The Click command with the input option added.
    """

    def add_input_option_decorator(_func: click.Command) -> click.Command:
        """Add --input option to command."""
        # Determine help text based on object type
        format_examples = {
            "picks": "object_name:user_id/session_id",
            "mesh": "object_name:user_id/session_id",
            "segmentation": "name:user_id/session_id@voxel_spacing",
        }

        help_text = (
            f"Input {object_type} URI (format: {format_examples.get(object_type, 'URI')}). Supports glob patterns."
        )

        opt = optgroup.option(
            "--input",
            "-i",
            "input_uri",
            required=True,
            help=help_text,
        )
        return opt(_func)

    if func is None:
        return add_input_option_decorator
    else:
        return add_input_option_decorator(func)


def add_output_option(object_type: str, func: click.Command = None, default_tool: str = None) -> Callable:
    """
    Add --output/-o option for URI-based output specification with smart defaults.

    Supports copick URI format with smart defaults and pattern matching:
    - Full format: object_name:user_id/session_id or name:user_id/session_id@voxel_spacing
    - Smart defaults: Omit components to inherit from input (e.g., just "membrane")
    - Templates: session_id can include {input_session_id} or {instance_id}

    Smart defaults:
    - Name/object omitted → inherits from input
    - user_id omitted → uses command name (e.g., "mesh2seg")
    - session_id omitted → auto-template based on input pattern
    - voxel_spacing omitted → inherits from input (segmentation only)

    Args:
        object_type (str): Type of object ('picks', 'mesh', 'segmentation').
        func (click.Command, optional): The Click command to which the option will be added.
        default_tool (str, optional): Default user_id if not specified in URI (deprecated, auto-detected).

    Returns:
        Callable: The Click command with the output option added.
    """

    def add_output_option_decorator(_func: click.Command) -> click.Command:
        """Add --output option to command."""
        # Determine help text based on object type
        shorthand_examples = {
            "picks": '"ribosome", "ribosome/my-session", or "/my-session"',
            "mesh": '"membrane", "membrane/my-session", or "/my-session"',
            "segmentation": '"membrane", "membrane/my-session", or "/my-session"',
        }

        voxel_suffix = "@voxel_spacing" if object_type == "segmentation" else ""
        help_text = (
            f"Output {object_type} URI. "
            f"Supports smart defaults (e.g., {shorthand_examples.get(object_type, 'shorthand')}). "
            f"Full format: object_name:user_id/session_id{voxel_suffix}."
        )

        opt = optgroup.option(
            "--output",
            "-o",
            "output_uri",
            required=True,
            help=help_text,
        )
        return opt(_func)

    if func is None:
        return add_output_option_decorator
    else:
        return add_output_option_decorator(func)


def add_dual_input_options(object_type: str, func: click.Command = None) -> Callable:
    """
    Add --input1/-i1 and --input2/-i2 options for dual input commands.

    Supports copick URI format for both inputs:
    - Picks/Meshes: object_name:user_id/session_id
    - Segmentations: name:user_id/session_id@voxel_spacing?multilabel=true

    Args:
        object_type (str): Type of object ('mesh', 'segmentation').
        func (click.Command, optional): The Click command to which the options will be added.

    Returns:
        Callable: The Click command with both input options added.
    """

    def add_dual_input_options_decorator(_func: click.Command) -> click.Command:
        """Add --input1 and --input2 options to command."""
        # Determine help text based on object type
        format_examples = {
            "mesh": "object_name:user_id/session_id",
            "segmentation": "name:user_id/session_id@voxel_spacing",
        }

        format_str = format_examples.get(object_type, "URI")

        opts = [
            optgroup.option(
                "--input1",
                "-i1",
                "input1_uri",
                required=True,
                help=f"First input {object_type} URI (format: {format_str}). Supports glob patterns.",
            ),
            optgroup.option(
                "--input2",
                "-i2",
                "input2_uri",
                required=True,
                help=f"Second input {object_type} URI (format: {format_str}). Supports glob patterns.",
            ),
        ]

        for opt in reversed(opts):
            _func = opt(_func)

        return _func

    if func is None:
        return add_dual_input_options_decorator
    else:
        return add_dual_input_options_decorator(func)


def add_multi_input_options(object_type: str, func: click.Command = None) -> Callable:
    """
    Add --input/-i option for multiple URI-based inputs.

    Supports specifying the same flag multiple times for N-way operations.
    Example: -i input1 -i input2 -i input3

    Pattern Support:
    - Glob (default): Use * and ? wildcards (e.g., 'name:user*/session-*')
    - Regex: Prefix with 're:' (e.g., 're:name:user\\d+/session-\\d+')

    Args:
        object_type (str): Type of object ('mesh', 'segmentation').
        func (click.Command, optional): The Click command to which the option will be added.

    Returns:
        Callable: The Click command with the multi-input option added.
    """

    def add_multi_input_options_decorator(_func: click.Command) -> click.Command:
        """Add --input option to command."""
        format_examples = {
            "mesh": "object_name:user_id/session_id",
            "segmentation": "name:user_id/session_id@voxel_spacing",
        }

        help_text = (
            f"Input {object_type} URI (format: {format_examples.get(object_type, 'URI')}). "
            f"Can be specified multiple times for N-way operations. "
            f"Supports glob patterns (default) or regex patterns (re: prefix)."
        )

        opt = optgroup.option(
            "--input",
            "-i",
            "input_uris",
            multiple=True,
            required=True,
            help=help_text,
        )
        return opt(_func)

    if func is None:
        return add_multi_input_options_decorator
    else:
        return add_multi_input_options_decorator(func)


def add_reference_mesh_option(func: click.Command = None, required: bool = False) -> Callable:
    """
    Add --ref-mesh/-rm option for reference mesh input.

    Args:
        func (click.Command, optional): The Click command to which the option will be added.
        required (bool): Whether the option is required.

    Returns:
        Callable: The Click command with the reference mesh option added.
    """

    def add_reference_mesh_option_decorator(_func: click.Command) -> click.Command:
        """Add --ref-mesh option to command."""
        opt = optgroup.option(
            "--ref-mesh",
            "-rm",
            "ref_mesh_uri",
            required=required,
            help="Reference mesh URI (format: object_name:user_id/session_id). Supports glob patterns.",
        )
        return opt(_func)

    if func is None:
        return add_reference_mesh_option_decorator
    else:
        return add_reference_mesh_option_decorator(func)


def add_reference_seg_option(func: click.Command = None, required: bool = False) -> Callable:
    """
    Add --ref-seg/-rs option for reference segmentation input.

    Args:
        func (click.Command, optional): The Click command to which the option will be added.
        required (bool): Whether the option is required.

    Returns:
        Callable: The Click command with the reference segmentation option added.
    """

    def add_reference_seg_option_decorator(_func: click.Command) -> click.Command:
        """Add --ref-seg option to command."""
        opt = optgroup.option(
            "--ref-seg",
            "-rs",
            "ref_seg_uri",
            required=required,
            help="Reference segmentation URI (format: name:user_id/session_id@voxel_spacing). Supports glob patterns.",
        )
        return opt(_func)

    if func is None:
        return add_reference_seg_option_decorator
    else:
        return add_reference_seg_option_decorator(func)


def add_tomogram_option(func: click.Command = None, required: bool = True) -> Callable:
    """
    Add --tomogram/-t option for tomogram URI specification.

    Tomogram URI format: tomo_type@voxel_spacing
    Example: "wbp@10.0"

    Args:
        func (click.Command, optional): The Click command to which the option will be added.
        required (bool): Whether the option is required. Default is True.

    Returns:
        Callable: The Click command with the tomogram option added.
    """

    def add_tomogram_option_decorator(_func: click.Command) -> click.Command:
        """Add --tomogram option to command."""
        opt = optgroup.option(
            "--tomogram",
            "-t",
            "tomogram_uri",
            required=required,
            help="Tomogram URI (format: tomo_type@voxel_spacing). Example: 'wbp@10.0'",
        )
        return opt(_func)

    if func is None:
        return add_tomogram_option_decorator
    else:
        return add_tomogram_option_decorator(func)
