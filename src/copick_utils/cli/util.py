"""CLI utilities for copick-utils commands."""

from typing import Callable

import click
from click_option_group import optgroup


def add_pick_input_options(func: click.Command) -> click.Command:
    """
    Add common input options for picks-to-mesh conversion commands.

    Supports flexible input selection:
    - For exact match: provide exact session ID
    - For pattern match: provide regex pattern (enables many-to-many mode)

    Args:
        func (click.Command): The Click command to which the options will be added.

    Returns:
        click.Command: The Click command with the input options added.
    """
    opts = [
        optgroup.option(
            "--pick-object-name",
            "-po",
            required=True,
            help="Name of the pick object to convert.",
        ),
        optgroup.option(
            "--pick-user-id",
            "-pu",
            required=True,
            help="User ID of the picks to convert.",
        ),
        optgroup.option(
            "--pick-session-id",
            "-ps",
            required=True,
            help="Session ID or regex pattern of the picks to convert. Use regex for many-to-many mode.",
        ),
    ]

    for opt in reversed(opts):
        func = opt(func)

    return func


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


def add_mesh_input_options(func: click.Command) -> click.Command:
    """
    Add common input options for mesh-to-segmentation conversion commands.

    Args:
        func (click.Command): The Click command to which the options will be added.

    Returns:
        click.Command: The Click command with the input options added.
    """
    opts = [
        optgroup.option(
            "--mesh-object-name",
            "-mo",
            required=True,
            help="Name of the mesh object to convert.",
        ),
        optgroup.option(
            "--mesh-user-id",
            "-mu",
            required=True,
            help="User ID of the mesh to convert.",
        ),
        optgroup.option(
            "--mesh-session-id",
            "-ms",
            required=True,
            help="Session ID or regex pattern of the mesh to convert.",
        ),
    ]

    for opt in reversed(opts):
        func = opt(func)

    return func


def add_segmentation_input_options(func: click.Command) -> click.Command:
    """
    Add common input options for segmentation-to-mesh conversion commands.

    Args:
        func (click.Command): The Click command to which the options will be added.

    Returns:
        click.Command: The Click command with the input options added.
    """
    opts = [
        optgroup.option(
            "--seg-name",
            "-sn",
            required=True,
            help="Name of the segmentation to convert.",
        ),
        optgroup.option(
            "--seg-user-id",
            "-su",
            required=True,
            help="User ID of the segmentation to convert.",
        ),
        optgroup.option(
            "--seg-session-id",
            "-ss",
            required=True,
            help="Session ID or regex pattern of the segmentation to convert.",
        ),
        optgroup.option(
            "--voxel-spacing",
            "-vs",
            type=float,
            required=True,
            help="Voxel spacing of the segmentation.",
        ),
        optgroup.option(
            "--multilabel/--no-multilabel",
            is_flag=True,
            default=False,
            help="Source is multilabel segmentation.",
        ),
    ]

    for opt in reversed(opts):
        func = opt(func)

    return func


def add_segmentation_output_options(func: click.Command = None, *, default_tool: str = "from-mesh") -> Callable:
    """
    Add common output options for mesh-to-segmentation conversion commands.

    Args:
        func (click.Command): The Click command to which the options will be added.
        default_tool (str): Default user ID for created segmentation.

    Returns:
        click.Command: The Click command with the output options added.
    """

    def add_segmentation_output_options_decorator(func: click.Command) -> click.Command:
        """
        Add common output options for segmentation creation commands.

        Args:
            func (click.Command): The Click command to which the options will be added.

        Returns:
            click.Command: The Click command with the output options added.
        """
        opts = [
            optgroup.option(
                "--seg-name",
                "-sn",
                required=True,
                help="Name of the segmentation to create.",
            ),
            optgroup.option(
                "--seg-user-id",
                "-su",
                default=default_tool,
                help="User ID for created segmentation. Defaults to tool name.",
            ),
            optgroup.option(
                "--seg-session-id",
                "-ss",
                default="0",
                help="Session ID for created segmentation.",
            ),
            optgroup.option(
                "--voxel-spacing",
                "-vs",
                type=float,
                required=True,
                help="Voxel spacing for the segmentation.",
            ),
            optgroup.option(
                "--multilabel/--no-multilabel",
                is_flag=True,
                default=False,
                help="Create multilabel segmentation.",
            ),
            optgroup.option(
                "--tomo-type",
                default="wbp",
                help="Type of tomogram to use as reference.",
            ),
        ]

        for opt in reversed(opts):
            func = opt(func)

        return func

    if func is None:
        return add_segmentation_output_options_decorator
    else:
        return add_segmentation_output_options_decorator(func)


def add_mesh_output_options(func: click.Command = None, *, default_tool: str = "from-picks") -> Callable:
    """
    Add common output options for picks-to-mesh conversion commands.

    Supports flexible output selection:
    - One-to-one: exact session ID for single output
    - One-to-many: session ID template with {instance_id} for individual meshes
    - Many-to-many: session ID template with {input_session_id} and {instance_id}

    Args:
        func (click.Command): The Click command to which the options will be added.
        default_tool (str): Default user ID for created mesh.

    Returns:
        click.Command: The Click command with the output options added.
    """

    def add_mesh_output_options_decorator(func: click.Command) -> click.Command:
        """
        Add common output options for picks-to-mesh conversion commands.

        Args:
            func (click.Command): The Click command to which the options will be added.

        Returns:
            click.Command: The Click command with the output options added.
        """
        opts = [
            optgroup.option(
                "--mesh-object-name",
                "-mo",
                help="Name of the mesh object to create. If not specified, defaults to pick object name.",
            ),
            optgroup.option(
                "--mesh-user-id",
                "-mu",
                default=default_tool,
                help="User ID for created mesh. Defaults to tool name.",
            ),
            optgroup.option(
                "--mesh-session-id",
                "-ms",
                default="0",
                help="Session ID or template for created mesh. Supports placeholders: {input_session_id} for many-to-many, {instance_id} for individual meshes.",
            ),
            optgroup.option(
                "--individual-meshes/--no-individual-meshes",
                "-im",
                is_flag=True,
                default=False,
                help="Create individual mesh files for each mesh instead of combining them.",
            ),
        ]

        for opt in reversed(opts):
            func = opt(func)

        return func

    if func is None:
        return add_mesh_output_options_decorator
    else:
        return add_mesh_output_options_decorator(func)


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


def add_picks_output_options(func: click.Command = None, *, default_tool: str = "from-segmentation") -> Callable:
    """
    Add common output options for segmentation-to-picks conversion commands.

    Args:
        func (click.Command): The Click command to which the options will be added.
        default_tool (str): Default user ID for created picks.

    Returns:
        click.Command: The Click command with the output options added.
    """

    def add_picks_output_options_decorator(func: click.Command) -> click.Command:
        """
        Add common output options for picks creation commands.

        Args:
            func (click.Command): The Click command to which the options will be added.

        Returns:
            click.Command: The Click command with the output options added.
        """
        opts = [
            optgroup.option(
                "--pick-object-name",
                "-po",
                required=True,
                help="Name of the pick object to create.",
            ),
            optgroup.option(
                "--pick-user-id",
                "-pu",
                default=default_tool,
                help="User ID for created picks. Defaults to tool name.",
            ),
            optgroup.option(
                "--pick-session-id",
                "-ps",
                default="0",
                help="Session ID for created picks.",
            ),
        ]

        for opt in reversed(opts):
            func = opt(func)

        return func

    if func is None:
        return add_picks_output_options_decorator
    else:
        return add_picks_output_options_decorator(func)


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
