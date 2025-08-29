"""CLI commands for converting picks to various mesh types.

Supports flexible input/output selection modes:
1. One-to-one: Single input picks → single output mesh
2. One-to-many: Single input picks → multiple output meshes (individual meshes)
3. Many-to-many: Multiple input picks (regex) → multiple output meshes (template)

Examples:
  # One-to-one: Convert specific picks to mesh
  picks2mesh --pick-session-id "manual-001" --mesh-session-id "mesh-001"

  # One-to-many: Create individual meshes from clusters
  picks2mesh --pick-session-id "manual-001" --mesh-session-id "mesh-{instance_id}" --individual-meshes

  # Many-to-many: Convert all manual picks to corresponding meshes
  picks2mesh --pick-session-id "manual-.*" --mesh-session-id "mesh-{input_session_id}"
"""

import click
import copick
from click_option_group import optgroup
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from copick_utils.cli.input_output_selection import InputOutputSelector, validate_placeholders
from copick_utils.cli.util import (
    add_clustering_options,
    add_mesh_output_options,
    add_pick_input_options,
    add_workers_option,
)


@click.command(
    context_settings={"show_default": True},
    short_help="Convert picks to mesh using convex hull or alpha shapes.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input picks.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_pick_input_options
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--mesh-type",
    "-t",
    type=click.Choice(["convex_hull", "alpha_shape"]),
    default="convex_hull",
    help="Type of mesh to create.",
)
@optgroup.option(
    "--alpha",
    "-a",
    type=float,
    help="Alpha parameter for alpha shapes (required if mesh-type=alpha_shape).",
)
@add_clustering_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_mesh_output_options(default_tool="picks2mesh")
@add_debug_option
def picks2mesh(
    config,
    run_names,
    pick_object_name,
    pick_user_id,
    pick_session_id,
    mesh_type,
    alpha,
    use_clustering,
    clustering_method,
    clustering_eps,
    clustering_min_samples,
    clustering_n_clusters,
    workers,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    all_clusters,
    individual_meshes,
    debug,
):
    """
    Convert picks to meshes using convex hull or alpha shapes.

    \b
    Supports flexible input/output selection modes:
    - One-to-one: exact session ID → exact session ID
    - One-to-many: exact session ID → template with {instance_id}
    - Many-to-many: regex pattern → template with {input_session_id} and {instance_id}

    \b
    Examples:
        # Convert single pick set to single mesh
        picks2mesh --pick-session-id "manual-001" --mesh-session-id "mesh-001"

        # Create individual meshes from clusters
        picks2mesh --pick-session-id "manual-001" --mesh-session-id "mesh-{instance_id}" --individual-meshes

        # Convert all manual picks using pattern matching
        picks2mesh --pick-session-id "manual-.*" --mesh-session-id "mesh-{input_session_id}"
    """
    from copick_utils.converters.mesh_from_picks import mesh_from_picks_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    if mesh_type == "alpha_shape" and alpha is None:
        raise click.BadParameter("Alpha parameter is required for alpha shapes")

    # Validate placeholder requirements
    try:
        validate_placeholders(pick_session_id, mesh_session_id, individual_meshes)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Prepare clustering parameters
    clustering_params = {}
    if clustering_method == "dbscan":
        clustering_params = {"eps": clustering_eps, "min_samples": clustering_min_samples}
    elif clustering_method == "kmeans":
        clustering_params = {"n_clusters": clustering_n_clusters}

    # Create input/output selector
    selector = InputOutputSelector(
        pick_object_name=pick_object_name,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        mesh_object_name=mesh_object_name,
        mesh_user_id=mesh_user_id,
        mesh_session_id=mesh_session_id,
        individual_meshes=individual_meshes,
    )

    logger.info(f"Converting picks to {mesh_type} mesh for object '{pick_object_name}'")
    logger.info(f"Selection mode: {selector.get_mode_description()}")
    logger.info(f"Source picks pattern: {pick_user_id}/{pick_session_id}")
    logger.info(f"Target mesh template: {selector.mesh_object_name} ({mesh_user_id}/{mesh_session_id})")

    # Collect all conversion tasks across runs
    all_tasks = []
    runs_to_process = root.runs if run_names_list is None else [root.get_run(name) for name in run_names_list]

    for run in runs_to_process:
        tasks = selector.get_conversion_tasks(run)
        all_tasks.extend(tasks)

    if not all_tasks:
        logger.warning("No matching picks found for conversion")
        return

    logger.info(f"Found {len(all_tasks)} conversion tasks across {len(runs_to_process)} runs")

    results = mesh_from_picks_batch(
        root=root,
        conversion_tasks=all_tasks,
        run_names=run_names_list,
        workers=workers,
        mesh_type=mesh_type,
        alpha=alpha,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        all_clusters=all_clusters,
        # individual_meshes=individual_meshes,
        # session_id_template=mesh_session_id,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_vertices = sum(result.get("vertices_created", 0) for result in results.values() if result)
    total_faces = sum(result.get("faces_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("processed", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total conversion tasks completed: {total_processed}")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")


@click.command(
    context_settings={"show_default": True},
    short_help="Convert picks to sphere meshes.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input picks.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_pick_input_options
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--subdivisions",
    type=int,
    default=2,
    help="Number of sphere subdivisions for mesh resolution.",
)
@optgroup.option(
    "--deduplicate-spheres/--no-deduplicate-spheres",
    is_flag=True,
    default=True,
    help="Merge overlapping spheres to avoid duplicates.",
)
@optgroup.option(
    "--min-sphere-distance",
    type=float,
    help="Minimum distance between sphere centers for deduplication (default: 0.5 * average radius).",
)
@add_clustering_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_mesh_output_options(default_tool="picks2sphere")
@add_debug_option
def picks2sphere(
    config,
    run_names,
    pick_object_name,
    pick_user_id,
    pick_session_id,
    use_clustering,
    clustering_method,
    clustering_eps,
    clustering_min_samples,
    clustering_n_clusters,
    subdivisions,
    deduplicate_spheres,
    min_sphere_distance,
    all_clusters,
    workers,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    individual_meshes,
    debug,
):
    """
    Convert picks to sphere meshes.

    \b
    Supports flexible input/output selection modes:
    - One-to-one: exact session ID → exact session ID
    - One-to-many: exact session ID → template with {instance_id}
    - Many-to-many: regex pattern → template with {input_session_id} and {instance_id}

    \b
    Examples:
        # Convert single pick set to single sphere mesh
        picks2sphere --pick-session-id "manual-001" --mesh-session-id "sphere-001"

        \b
    # Create individual sphere meshes from clusters
    picks2sphere --pick-session-id "manual-001" --mesh-session-id "sphere-{instance_id}" --individual-meshes

        \b
    # Convert all manual picks using pattern matching
    picks2sphere --pick-session-id "manual-.*" --mesh-session-id "sphere-{input_session_id}"
    """
    from copick_utils.converters.sphere_from_picks import sphere_from_picks_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Validate placeholder requirements
    try:
        validate_placeholders(pick_session_id, mesh_session_id, individual_meshes)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Prepare clustering parameters
    clustering_params = {}
    if clustering_method == "dbscan":
        clustering_params = {"eps": clustering_eps, "min_samples": clustering_min_samples}
    elif clustering_method == "kmeans":
        clustering_params = {"n_clusters": clustering_n_clusters}

    # Create input/output selector
    selector = InputOutputSelector(
        pick_object_name=pick_object_name,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        mesh_object_name=mesh_object_name,
        mesh_user_id=mesh_user_id,
        mesh_session_id=mesh_session_id,
        individual_meshes=individual_meshes,
    )

    logger.info(f"Converting picks to sphere mesh for object '{pick_object_name}'")
    logger.info(f"Selection mode: {selector.get_mode_description()}")
    logger.info(f"Source picks pattern: {pick_user_id}/{pick_session_id}")
    logger.info(f"Target mesh template: {selector.mesh_object_name} ({mesh_user_id}/{mesh_session_id})")

    # Collect all conversion tasks across runs
    all_tasks = []
    runs_to_process = root.runs if run_names_list is None else [root.get_run(name) for name in run_names_list]

    for run in runs_to_process:
        tasks = selector.get_conversion_tasks(run)
        all_tasks.extend(tasks)

    if not all_tasks:
        logger.warning("No matching picks found for conversion")
        return

    logger.info(f"Found {len(all_tasks)} conversion tasks across {len(runs_to_process)} runs")

    results = sphere_from_picks_batch(
        root=root,
        conversion_tasks=all_tasks,
        run_names=run_names_list,
        workers=workers,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        subdivisions=subdivisions,
        all_clusters=all_clusters,
        deduplicate_spheres_flag=deduplicate_spheres,
        min_sphere_distance=min_sphere_distance,
        # individual_meshes=individual_meshes,
        # session_id_template=mesh_session_id,
    )
    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_vertices = sum(result.get("vertices_created", 0) for result in results.values() if result)
    total_faces = sum(result.get("faces_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("processed", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total conversion tasks completed: {total_processed}")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")


@click.command(
    context_settings={"show_default": True},
    short_help="Convert picks to ellipsoid meshes.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input picks.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_pick_input_options
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--subdivisions",
    type=int,
    default=2,
    help="Number of ellipsoid subdivisions for mesh resolution.",
)
@optgroup.option(
    "--deduplicate-ellipsoids/--no-deduplicate-ellipsoids",
    is_flag=True,
    default=True,
    help="Merge overlapping ellipsoids to avoid duplicates.",
)
@optgroup.option(
    "--min-ellipsoid-distance",
    type=float,
    help="Minimum distance between ellipsoid centers for deduplication (default: 0.5 * average major axis).",
)
@add_clustering_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_mesh_output_options(default_tool="picks2ellipsoid")
@add_debug_option
def picks2ellipsoid(
    config,
    run_names,
    pick_object_name,
    pick_user_id,
    pick_session_id,
    use_clustering,
    clustering_method,
    clustering_eps,
    clustering_min_samples,
    clustering_n_clusters,
    subdivisions,
    deduplicate_ellipsoids,
    min_ellipsoid_distance,
    all_clusters,
    workers,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    individual_meshes,
    debug,
):
    """
    Convert picks to ellipsoid meshes.

    \b
    Supports flexible input/output selection modes:
    - One-to-one: exact session ID → exact session ID
    - One-to-many: exact session ID → template with {instance_id}
    - Many-to-many: regex pattern → template with {input_session_id} and {instance_id}

    \b
    Examples:
        # Convert single pick set to single ellipsoid mesh
        picks2ellipsoid --pick-session-id "manual-001" --mesh-session-id "ellipsoid-001"

        \b
    # Create individual ellipsoid meshes from clusters
    picks2ellipsoid --pick-session-id "manual-001" --mesh-session-id "ellipsoid-{instance_id}" --individual-meshes

        \b
    # Convert all manual picks using pattern matching
    picks2ellipsoid --pick-session-id "manual-.*" --mesh-session-id "ellipsoid-{input_session_id}"
    """
    from copick_utils.converters.ellipsoid_from_picks import ellipsoid_from_picks_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Validate placeholder requirements
    try:
        validate_placeholders(pick_session_id, mesh_session_id, individual_meshes)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Prepare clustering parameters
    clustering_params = {}
    if clustering_method == "dbscan":
        clustering_params = {"eps": clustering_eps, "min_samples": clustering_min_samples}
    elif clustering_method == "kmeans":
        clustering_params = {"n_clusters": clustering_n_clusters}

    # Create input/output selector
    selector = InputOutputSelector(
        pick_object_name=pick_object_name,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        mesh_object_name=mesh_object_name,
        mesh_user_id=mesh_user_id,
        mesh_session_id=mesh_session_id,
        individual_meshes=individual_meshes,
    )

    logger.info(f"Converting picks to ellipsoid mesh for object '{pick_object_name}'")
    logger.info(f"Selection mode: {selector.get_mode_description()}")
    logger.info(f"Source picks pattern: {pick_user_id}/{pick_session_id}")
    logger.info(f"Target mesh template: {selector.mesh_object_name} ({mesh_user_id}/{mesh_session_id})")

    # Collect all conversion tasks across runs
    all_tasks = []
    runs_to_process = root.runs if run_names_list is None else [root.get_run(name) for name in run_names_list]

    for run in runs_to_process:
        tasks = selector.get_conversion_tasks(run)
        all_tasks.extend(tasks)

    if not all_tasks:
        logger.warning("No matching picks found for conversion")
        return

    logger.info(f"Found {len(all_tasks)} conversion tasks across {len(runs_to_process)} runs")

    results = ellipsoid_from_picks_batch(
        root=root,
        conversion_tasks=all_tasks,
        run_names=run_names_list,
        workers=workers,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        subdivisions=subdivisions,
        all_clusters=all_clusters,
        deduplicate_ellipsoids_flag=deduplicate_ellipsoids,
        min_ellipsoid_distance=min_ellipsoid_distance,
        # individual_meshes=individual_meshes,
        # session_id_template=mesh_session_id,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_vertices = sum(result.get("vertices_created", 0) for result in results.values() if result)
    total_faces = sum(result.get("faces_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("processed", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total conversion tasks completed: {total_processed}")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")


@click.command(
    context_settings={"show_default": True},
    short_help="Convert picks to plane meshes.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input picks.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_pick_input_options
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--padding",
    type=float,
    default=1.2,
    help="Padding factor for plane size (1.0=exact fit, >1.0=larger plane).",
)
@add_clustering_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_mesh_output_options(default_tool="picks2plane")
@add_debug_option
def picks2plane(
    config,
    run_names,
    pick_object_name,
    pick_user_id,
    pick_session_id,
    use_clustering,
    clustering_method,
    clustering_eps,
    clustering_min_samples,
    clustering_n_clusters,
    padding,
    all_clusters,
    workers,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    individual_meshes,
    debug,
):
    """
    Convert picks to plane meshes.

    \b
    Supports flexible input/output selection modes:
    - One-to-one: exact session ID → exact session ID
    - One-to-many: exact session ID → template with {instance_id}
    - Many-to-many: regex pattern → template with {input_session_id} and {instance_id}

    \b
    Examples:
        \b
        # Convert single pick set to single plane mesh
        picks2plane --pick-session-id "manual-001" --mesh-session-id "plane-001"

        \b
    # Create individual plane meshes from clusters
    picks2plane --pick-session-id "manual-001" --mesh-session-id "plane-{instance_id}" --individual-meshes

        \b
    # Convert all manual picks using pattern matching
    picks2plane --pick-session-id "manual-.*" --mesh-session-id "plane-{input_session_id}"
    """
    from copick_utils.converters.plane_from_picks import plane_from_picks_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Validate placeholder requirements
    try:
        validate_placeholders(pick_session_id, mesh_session_id, individual_meshes)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Prepare clustering parameters
    clustering_params = {}
    if clustering_method == "dbscan":
        clustering_params = {"eps": clustering_eps, "min_samples": clustering_min_samples}
    elif clustering_method == "kmeans":
        clustering_params = {"n_clusters": clustering_n_clusters}

    # Create input/output selector
    selector = InputOutputSelector(
        pick_object_name=pick_object_name,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        mesh_object_name=mesh_object_name,
        mesh_user_id=mesh_user_id,
        mesh_session_id=mesh_session_id,
        individual_meshes=individual_meshes,
    )

    logger.info(f"Converting picks to plane mesh for object '{pick_object_name}'")
    logger.info(f"Selection mode: {selector.get_mode_description()}")
    logger.info(f"Source picks pattern: {pick_user_id}/{pick_session_id}")
    logger.info(f"Target mesh template: {selector.mesh_object_name} ({mesh_user_id}/{mesh_session_id})")

    # Collect all conversion tasks across runs
    all_tasks = []
    runs_to_process = root.runs if run_names_list is None else [root.get_run(name) for name in run_names_list]

    for run in runs_to_process:
        tasks = selector.get_conversion_tasks(run)
        all_tasks.extend(tasks)

    if not all_tasks:
        logger.warning("No matching picks found for conversion")
        return

    logger.info(f"Found {len(all_tasks)} conversion tasks across {len(runs_to_process)} runs")

    results = plane_from_picks_batch(
        root=root,
        conversion_tasks=all_tasks,
        run_names=run_names_list,
        workers=workers,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        padding=padding,
        all_clusters=all_clusters,
        # individual_meshes=individual_meshes,
        # session_id_template=mesh_session_id,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_vertices = sum(result.get("vertices_created", 0) for result in results.values() if result)
    total_faces = sum(result.get("faces_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("processed", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total conversion tasks completed: {total_processed}")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")


@click.command(
    context_settings={"show_default": True},
    short_help="Convert picks to 2D surface meshes.",
    no_args_is_help=True,
)
@add_config_option
@optgroup.group("\nInput Options", help="Options related to the input picks.")
@optgroup.option(
    "--run-names",
    "-r",
    multiple=True,
    help="Specific run names to process (default: all runs).",
)
@add_pick_input_options
@optgroup.group("\nTool Options", help="Options related to this tool.")
@optgroup.option(
    "--surface-method",
    type=click.Choice(["delaunay", "rbf", "grid"]),
    default="delaunay",
    help="Surface fitting method.",
)
@optgroup.option(
    "--grid-resolution",
    type=int,
    default=50,
    help="Resolution for grid-based surface methods.",
)
@add_clustering_options
@add_workers_option
@optgroup.group("\nOutput Options", help="Options related to output meshes.")
@add_mesh_output_options(default_tool="picks2surface")
@add_debug_option
def picks2surface(
    config,
    run_names,
    pick_object_name,
    pick_user_id,
    pick_session_id,
    surface_method,
    grid_resolution,
    use_clustering,
    clustering_method,
    clustering_eps,
    clustering_min_samples,
    clustering_n_clusters,
    all_clusters,
    workers,
    mesh_object_name,
    mesh_user_id,
    mesh_session_id,
    individual_meshes,
    debug,
):
    """
    Convert picks to 2D surface meshes.

    Supports flexible input/output selection modes:
    - One-to-one: exact session ID → exact session ID
    - One-to-many: exact session ID → template with {instance_id}
    - Many-to-many: regex pattern → template with {input_session_id} and {instance_id}

    Examples:
        # Convert single pick set to single surface mesh
        picks2surface --pick-session-id "manual-001" --mesh-session-id "surface-001"

        # Create individual surface meshes from clusters
        picks2surface --pick-session-id "manual-001" --mesh-session-id "surface-{instance_id}" --individual-meshes

        # Convert all manual picks using pattern matching
        picks2surface --pick-session-id "manual-.*" --mesh-session-id "surface-{input_session_id}"
    """
    from copick_utils.converters.surface_from_picks import surface_from_picks_batch

    logger = get_logger(__name__, debug=debug)

    root = copick.from_file(config)
    run_names_list = list(run_names) if run_names else None

    # Validate placeholder requirements
    try:
        validate_placeholders(pick_session_id, mesh_session_id, individual_meshes)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    # Prepare clustering parameters
    clustering_params = {}
    if clustering_method == "dbscan":
        clustering_params = {"eps": clustering_eps, "min_samples": clustering_min_samples}
    elif clustering_method == "kmeans":
        clustering_params = {"n_clusters": clustering_n_clusters}

    # Create input/output selector
    selector = InputOutputSelector(
        pick_object_name=pick_object_name,
        pick_user_id=pick_user_id,
        pick_session_id=pick_session_id,
        mesh_object_name=mesh_object_name,
        mesh_user_id=mesh_user_id,
        mesh_session_id=mesh_session_id,
        individual_meshes=individual_meshes,
    )

    logger.info(f"Converting picks to {surface_method} surface mesh for object '{pick_object_name}'")
    logger.info(f"Selection mode: {selector.get_mode_description()}")
    logger.info(f"Source picks pattern: {pick_user_id}/{pick_session_id}")
    logger.info(f"Target mesh template: {selector.mesh_object_name} ({mesh_user_id}/{mesh_session_id})")

    # Collect all conversion tasks across runs
    all_tasks = []
    runs_to_process = root.runs if run_names_list is None else [root.get_run(name) for name in run_names_list]

    for run in runs_to_process:
        tasks = selector.get_conversion_tasks(run)
        all_tasks.extend(tasks)

    if not all_tasks:
        logger.warning("No matching picks found for conversion")
        return

    logger.info(f"Found {len(all_tasks)} conversion tasks across {len(runs_to_process)} runs")

    results = surface_from_picks_batch(
        root=root,
        conversion_tasks=all_tasks,
        run_names=run_names_list,
        workers=workers,
        surface_method=surface_method,
        grid_resolution=grid_resolution,
        use_clustering=use_clustering,
        clustering_method=clustering_method,
        clustering_params=clustering_params,
        all_clusters=all_clusters,
        # individual_meshes=individual_meshes,
        # session_id_template=mesh_session_id,
    )

    successful = sum(1 for result in results.values() if result and result.get("processed", 0) > 0)
    total_vertices = sum(result.get("vertices_created", 0) for result in results.values() if result)
    total_faces = sum(result.get("faces_created", 0) for result in results.values() if result)
    total_processed = sum(result.get("processed", 0) for result in results.values() if result)

    # Collect all errors
    all_errors = []
    for result in results.values():
        if result and result.get("errors"):
            all_errors.extend(result["errors"])

    logger.info(f"Completed: {successful}/{len(results)} runs processed successfully")
    logger.info(f"Total conversion tasks completed: {total_processed}")
    logger.info(f"Total vertices created: {total_vertices}")
    logger.info(f"Total faces created: {total_faces}")

    if all_errors:
        logger.warning(f"Encountered {len(all_errors)} errors during processing")
        for error in all_errors[:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
        if len(all_errors) > 5:
            logger.warning(f"  ... and {len(all_errors) - 5} more errors")
