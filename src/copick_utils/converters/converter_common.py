from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import trimesh as tm
from copick.util.log import get_logger
from sklearn.cluster import DBSCAN, KMeans

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickRoot, CopickRun

logger = get_logger(__name__)


def validate_points(points: np.ndarray, min_count: int, shape_name: str) -> bool:
    """Validate that we have enough points for the given shape type.

    Args:
        points: Nx3 array of points.
        min_count: Minimum number of points required.
        shape_name: Name of the shape for error messages.

    Returns:
        True if valid, False otherwise.
    """
    if len(points) < min_count:
        logger.warning(f"Need at least {min_count} points to fit a {shape_name}, got {len(points)}")
        return False
    return True


def cluster(
    points: np.ndarray,
    method: str = "dbscan",
    min_points_per_cluster: int = 3,
    **kwargs,
) -> List[np.ndarray]:
    """Cluster points using the specified method.

    Args:
        points: Nx3 array of points.
        method: Clustering method ('dbscan', 'kmeans').
        min_points_per_cluster: Minimum points required per cluster.
        **kwargs: Additional parameters for clustering.

    Returns:
        List of point arrays, one per cluster.
    """
    if method == "dbscan":
        eps = kwargs.get("eps", 1.0)
        min_samples = kwargs.get("min_samples", 3)

        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(points)

        # Group points by cluster label (excluding noise points labeled as -1)
        clusters = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label != -1:  # Skip noise points
                cluster_points = points[labels == label]
                if len(cluster_points) >= min_points_per_cluster:
                    clusters.append(cluster_points)

        return clusters

    elif method == "kmeans":
        n_clusters = kwargs.get("n_clusters", 1)

        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clustering.fit_predict(points)

        clusters = []
        for i in range(n_clusters):
            cluster_points = points[labels == i]
            if len(cluster_points) >= min_points_per_cluster:
                clusters.append(cluster_points)

        return clusters

    else:
        raise ValueError(f"Unknown clustering method: {method}")


def store_mesh_with_stats(
    run: "CopickRun",
    mesh: tm.Trimesh,
    object_name: str,
    session_id: str,
    user_id: str,
    shape_name: str,
) -> Tuple["CopickMesh", Dict[str, int]]:
    """Store a mesh and return statistics.

    Args:
        run: Copick run object.
        mesh: Trimesh object to store.
        object_name: Name of the mesh object.
        session_id: Session ID for the mesh.
        user_id: User ID for the mesh.
        shape_name: Name of the shape for logging.

    Returns:
        Tuple of (CopickMesh object, stats dict).

    Raises:
        Exception: If mesh creation fails.
    """
    copick_mesh = run.new_mesh(object_name, session_id, user_id, exist_ok=True)
    copick_mesh.mesh = mesh
    copick_mesh.store()

    stats = {
        "vertices_created": len(mesh.vertices),
        "faces_created": len(mesh.faces),
    }
    logger.info(
        f"Created {shape_name} mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces",
    )
    return copick_mesh, stats


def create_batch_worker(
    converter_func: Callable,
    output_type: str,
    input_type: str = "picks",
    min_points: int = 3,
) -> Callable:
    """Create a batch worker function for a specific converter.

    Args:
        converter_func: The main converter function to call.
        output_type: Type of output being created (e.g., "mesh", "segmentation").
        input_type: Type of input being processed (e.g., "picks", "mesh", "segmentation").
        min_points: Minimum points required (only relevant for picks input).

    Returns:
        Worker function that can be used with map_runs.
    """

    def worker(
        run: "CopickRun",
        input_object_name: str,
        input_user_id: str,
        input_session_id: str,
        output_object_name: str,
        output_session_id: str,
        output_user_id: str,
        **converter_kwargs,
    ) -> Dict[str, Any]:
        """Worker function for batch conversion."""
        try:
            # Get input data based on input type
            if input_type == "picks":
                input_list = run.get_picks(
                    object_name=input_object_name,
                    user_id=input_user_id,
                    session_id=input_session_id,
                )
                if not input_list:
                    return {"processed": 0, "errors": [f"No picks found for {run.name}"]}

                input_obj = input_list[0]
                points, transforms = input_obj.numpy()

                if points is None or len(points) == 0:
                    return {"processed": 0, "errors": [f"Could not load pick data for {run.name}"]}

                # Use points directly - copick coordinates are already in angstroms
                positions = points[:, :3]

                # Validate minimum points
                if not validate_points(positions, min_points, output_type):
                    return {"processed": 0, "errors": [f"Insufficient points for {run.name}"]}

                # Call converter with points
                result = converter_func(
                    points=positions,
                    run=run,
                    object_name=output_object_name,
                    session_id=output_session_id,
                    user_id=output_user_id,
                    **converter_kwargs,
                )

            elif input_type == "mesh":
                input_list = run.get_meshes(
                    object_name=input_object_name,
                    user_id=input_user_id,
                    session_id=input_session_id,
                )
                if not input_list:
                    return {"processed": 0, "errors": [f"No meshes found for {run.name}"]}

                input_obj = input_list[0]

                # Call converter with mesh object
                result = converter_func(
                    mesh=input_obj,
                    run=run,
                    object_name=output_object_name,
                    session_id=output_session_id,
                    user_id=output_user_id,
                    **converter_kwargs,
                )

            elif input_type == "segmentation":
                input_list = run.get_segmentations(
                    name=input_object_name,
                    user_id=input_user_id,
                    session_id=input_session_id,
                    **converter_kwargs,  # Pass through voxel_size, is_multilabel, etc.
                )
                if not input_list:
                    return {"processed": 0, "errors": [f"No segmentations found for {run.name}"]}

                input_obj = input_list[0]

                # Call converter with segmentation object
                result = converter_func(
                    segmentation=input_obj,
                    run=run,
                    object_name=output_object_name,
                    session_id=output_session_id,
                    user_id=output_user_id,
                    **converter_kwargs,
                )
            else:
                return {"processed": 0, "errors": [f"Unknown input type: {input_type}"]}

            if result:
                output_obj, stats = result
                return {
                    "processed": 1,
                    "errors": [],
                    "result": output_obj,
                    **stats,  # Include all stats (vertices_created, faces_created, voxels_created, etc.)
                }
            else:
                return {"processed": 0, "errors": [f"No {output_type} generated for {run.name}"]}

        except Exception as e:
            return {"processed": 0, "errors": [f"Error processing {run.name}: {e}"]}

    return worker


def create_batch_converter(
    converter_func: Callable,
    task_description: str,
    output_type: str,
    input_type: str = "picks",
    min_points: int = 3,
    dual_input: bool = False,
) -> Callable:
    """
    Create a batch converter function that supports flexible input/output selection.

    Args:
        converter_func: The main converter function to call.
        task_description: Description for the progress bar.
        output_type: Type of output being created (e.g., "mesh", "segmentation").
        input_type: Type of input being processed (e.g., "picks", "mesh", "segmentation").
        min_points: Minimum points required (only relevant for picks input).
        dual_input: If True, expects tasks with dual inputs (e.g., mesh boolean operations).

    Returns:
        Batch converter function.
    """

    def batch_converter(
        root: "CopickRoot",
        conversion_tasks: List[Dict[str, Any]],
        run_names: Optional[List[str]] = None,
        workers: int = 8,
        **converter_kwargs,
    ) -> Dict[str, Any]:
        """
        Batch convert with flexible input/output selection.

        Args:
            root: The copick root containing runs to process.
            conversion_tasks: List of conversion task dictionaries.
            run_names: List of run names to process. If None, processes all runs.
            workers: Number of worker processes. Default is 8.
            **converter_kwargs: Additional arguments passed to the converter function.

        Returns:
            Dictionary with processing results and statistics.
        """
        from copick.ops.run import map_runs

        runs_to_process = [run.name for run in root.runs] if run_names is None else run_names

        # Group tasks by run - determine input object key dynamically
        # ConversionSelector always uses 'input_object' as the key
        input_key = "input_object"

        tasks_by_run = {}
        for task in conversion_tasks:
            # Get run name from input object
            input_obj = task.get(input_key)
            if input_obj is None:
                # Try alternate keys for backward compatibility
                input_obj = task.get("input_picks") or task.get("input_mesh") or task.get("input_segmentation")

            if input_obj:
                run_name = input_obj.run.name
                if run_name not in tasks_by_run:
                    tasks_by_run[run_name] = []
                tasks_by_run[run_name].append(task)

        # Create a modified worker that processes multiple tasks per run
        def multi_task_worker(
            run: "CopickRun",
            **kwargs,
        ) -> Dict[str, Any]:
            """Worker function that processes multiple conversion tasks for a single run."""
            run_tasks = tasks_by_run.get(run.name, [])

            if not run_tasks:
                return {"processed": 0, "errors": [f"No tasks for {run.name}"]}

            total_processed = 0
            all_errors = []
            accumulated_stats = {}

            for task in run_tasks:
                try:
                    input_obj = task.get(input_key)
                    if input_obj is None:
                        # Try alternate keys for backward compatibility
                        input_obj = task.get("input_picks") or task.get("input_mesh") or task.get("input_segmentation")

                    if not input_obj:
                        all_errors.append(f"No input object found in task for {run.name}")
                        continue

                    # Handle different input types
                    if input_type == "picks":
                        points, transforms = input_obj.numpy()
                        if points is None or len(points) == 0:
                            all_errors.append(f"Could not load pick data from {input_obj.session_id} in {run.name}")
                            continue

                        positions = points[:, :3]
                        if not validate_points(positions, min_points, output_type):
                            all_errors.append(
                                f"Insufficient points for {output_type} in {input_obj.session_id}/{run.name}",
                            )
                            continue

                        # Call converter with points
                        result = converter_func(
                            points=positions,
                            run=run,
                            object_name=task.get(f"{output_type}_object_name", task.get("mesh_object_name")),
                            session_id=task.get(f"{output_type}_session_id", task.get("mesh_session_id")),
                            user_id=task.get(f"{output_type}_user_id", task.get("mesh_user_id")),
                            individual_meshes=task.get("individual_meshes", False),
                            session_id_template=task.get("session_id_template"),
                            **converter_kwargs,
                        )

                    else:
                        # For mesh or segmentation input, pass the object directly
                        if input_type == "mesh":
                            if dual_input:
                                # For dual-input operations like mesh boolean operations
                                input2_obj = task.get("input2_mesh")
                                if not input2_obj:
                                    all_errors.append(f"Missing second input mesh for task in {run.name}")
                                    continue

                                result = converter_func(
                                    mesh1=input_obj,
                                    mesh2=input2_obj,
                                    run=run,
                                    object_name=task.get("mesh_object_name"),
                                    session_id=task.get("mesh_session_id"),
                                    user_id=task.get("mesh_user_id"),
                                    **converter_kwargs,
                                )
                            else:
                                # Single-input mesh operations
                                result = converter_func(
                                    mesh=input_obj,
                                    run=run,
                                    object_name=task.get("output_object_name"),
                                    session_id=task.get("output_session_id"),
                                    user_id=task.get("output_user_id"),
                                    **converter_kwargs,
                                )
                        elif input_type == "segmentation":
                            if dual_input:
                                # For dual-input operations like segmentation boolean operations
                                input2_obj = task.get("input2_segmentation")
                                if not input2_obj:
                                    all_errors.append(f"Missing second input segmentation for task in {run.name}")
                                    continue

                                result = converter_func(
                                    segmentation1=input_obj,
                                    segmentation2=input2_obj,
                                    run=run,
                                    object_name=task.get("segmentation_object_name"),
                                    session_id=task.get("segmentation_session_id"),
                                    user_id=task.get("segmentation_user_id"),
                                    voxel_spacing=task.get("voxel_spacing"),
                                    tomo_type=task.get("tomo_type", "wbp"),
                                    is_multilabel=task.get("is_multilabel", False),
                                    **converter_kwargs,
                                )
                            else:
                                # Single-input segmentation operations
                                # Pass all task parameters to the converter function
                                task_params = dict(task)
                                task_params["segmentation"] = input_obj
                                task_params["run"] = run
                                task_params.update(converter_kwargs)

                                result = converter_func(**task_params)

                    if result:
                        output_obj, stats = result
                        total_processed += 1

                        # Accumulate stats dynamically
                        for key, value in stats.items():
                            if key not in accumulated_stats:
                                accumulated_stats[key] = 0
                            accumulated_stats[key] += value
                    else:
                        session_id = getattr(input_obj, "session_id", "unknown")
                        all_errors.append(f"No {output_type} generated for {session_id} in {run.name}")

                except Exception as e:
                    logger.exception(f"Error processing task in {run.name}: {e}")
                    all_errors.append(f"Error processing task in {run.name}: {e}")

            return {
                "processed": total_processed,
                "errors": all_errors,
                **accumulated_stats,
            }

        # Only process runs that have tasks
        relevant_runs = [run for run in runs_to_process if run in tasks_by_run]

        if not relevant_runs:
            input_type + "s" if not input_type.endswith("s") else input_type
            # Fix pluralization for common cases
            if input_type == "mesh":
                pass
            # Return empty results dict to match map_runs format
            return {}

        results = map_runs(
            callback=multi_task_worker,
            root=root,
            runs=relevant_runs,
            workers=workers,
            task_desc=task_description,
        )

        return results

    return batch_converter


def handle_clustering_workflow(
    points: np.ndarray,
    use_clustering: bool,
    clustering_method: str,
    clustering_params: Dict[str, Any],
    all_clusters: bool,
    min_points_per_cluster: int,
    shape_creation_func: Callable[..., tm.Trimesh],
    shape_name: str,
    **shape_kwargs,
) -> Tuple[Optional[tm.Trimesh], List[np.ndarray]]:
    """Handle the common clustering workflow for all converters.

    Args:
        points: Input points to process.
        use_clustering: Whether to cluster points first.
        clustering_method: Clustering method ('dbscan', 'kmeans').
        clustering_params: Parameters for clustering.
        all_clusters: If True, use all clusters; if False, use only largest.
        min_points_per_cluster: Minimum points required per cluster.
        shape_creation_func: Function to create shapes from point clusters.
        shape_name: Name of shape for logging.
        **shape_kwargs: Additional arguments for shape creation.

    Returns:
        Tuple of (combined_mesh, points_used_for_logging).
    """
    if use_clustering:
        point_clusters = cluster(
            points,
            clustering_method,
            min_points_per_cluster,
            **clustering_params,
        )

        if not point_clusters:
            logger.warning("No valid clusters found")
            return None, []

        logger.info(f"Found {len(point_clusters)} clusters")

        if all_clusters and len(point_clusters) > 1:
            # Create shapes from all clusters and combine them
            all_meshes = []
            for i, cluster_points in enumerate(point_clusters):
                try:
                    cluster_mesh = shape_creation_func(cluster_points, **shape_kwargs)
                    all_meshes.append(cluster_mesh)
                    logger.info(f"Cluster {i}: created {shape_name} with {len(cluster_mesh.vertices)} vertices")
                except Exception as e:
                    logger.critical(f"Failed to create {shape_name} from cluster {i}: {e}")
                    continue

            if not all_meshes:
                logger.warning(f"No valid {shape_name}s created from clusters")
                return None, []

            # Combine all meshes
            combined_mesh = tm.util.concatenate(all_meshes)
            return combined_mesh, points  # Return original points for logging
        else:
            # Use largest cluster
            cluster_sizes = [len(cluster) for cluster in point_clusters]
            largest_cluster_idx = np.argmax(cluster_sizes)
            points_to_use = point_clusters[largest_cluster_idx]
            logger.info(f"Using largest cluster with {len(points_to_use)} points")

            combined_mesh = shape_creation_func(points_to_use, **shape_kwargs)
            return combined_mesh, points_to_use
    else:
        # Use all points without clustering
        combined_mesh = shape_creation_func(points, **shape_kwargs)
        logger.info(f"Created {shape_name} from {len(points)} points")
        return combined_mesh, points
