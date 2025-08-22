"""CLI utilities for copick-utils commands."""

import click
from click_option_group import optgroup


def add_pick_input_options(func: click.Command) -> click.Command:
    """
    Add common input options for picks-to-mesh conversion commands.

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
            help="Session ID of the picks to convert.",
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
            "--create-multiple/--no-create-multiple",
            "-mm",
            is_flag=True,
            default=False,
            help="Create separate meshes for each cluster.",
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


def add_mesh_output_options(func: click.Command) -> click.Command:
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
            required=True,
            help="Name of the mesh object to create.",
        ),
        optgroup.option(
            "--mesh-user-id",
            "-mu",
            default="from-picks",
            help="User ID for created mesh.",
        ),
        optgroup.option(
            "--mesh-session-id",
            "-ms",
            default="0",
            help="Session ID for created mesh. When using --individual-meshes, can contain placeholders.",
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
