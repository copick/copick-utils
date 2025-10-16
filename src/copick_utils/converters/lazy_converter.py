"""Lazy task discovery architecture for parallel object discovery and processing."""

import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from copick.util.log import get_logger

from copick_utils.util.config_models import ReferenceConfig, SelectorConfig, TaskConfig

if TYPE_CHECKING:
    from copick.models import CopickRoot, CopickRun

logger = get_logger(__name__)


def create_selector_config(
    input_type: str,
    output_type: str,
    input_object_name: str,
    input_user_id: str,
    input_session_id: str,
    output_object_name: Optional[str] = None,
    output_user_id: str = "converter",
    output_session_id: str = "0",
    individual_outputs: bool = False,
    segmentation_name: Optional[str] = None,
    voxel_spacing: Optional[float] = None,
) -> SelectorConfig:
    """
    Create selector configuration using Pydantic model with validation.

    Args:
        input_type: Type of input ('picks', 'mesh', 'segmentation')
        output_type: Type of output ('picks', 'mesh', 'segmentation')
        input_object_name: Name of the input object
        input_user_id: User ID of the input
        input_session_id: Session ID or regex pattern of the input
        output_object_name: Name of the output object (defaults to input_object_name)
        output_user_id: User ID for created output
        output_session_id: Session ID or template for created output
        individual_outputs: Whether to create individual output files
        segmentation_name: Name for segmentation (when input or output is segmentation)
        voxel_spacing: Voxel spacing for segmentation

    Returns:
        Validated SelectorConfig model
    """
    return SelectorConfig(
        input_type=input_type,
        output_type=output_type,
        input_object_name=input_object_name,
        input_user_id=input_user_id,
        input_session_id=input_session_id,
        output_object_name=output_object_name or input_object_name,
        output_user_id=output_user_id,
        output_session_id=output_session_id,
        individual_outputs=individual_outputs,
        segmentation_name=segmentation_name,
        voxel_spacing=voxel_spacing,
    )


def create_reference_config(
    reference_type: str,  # "mesh" or "segmentation"
    object_name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    voxel_spacing: Optional[float] = None,
    **additional_params,
) -> ReferenceConfig:
    """
    Create reference discovery configuration using Pydantic model with validation.

    Args:
        reference_type: Type of reference ("mesh" or "segmentation")
        object_name: Name of reference object
        user_id: User ID of reference
        session_id: Session ID of reference
        voxel_spacing: Voxel spacing for segmentation references
        **additional_params: Additional parameters (max_distance, etc.)

    Returns:
        Validated ReferenceConfig model
    """
    return ReferenceConfig(
        reference_type=reference_type,
        object_name=object_name,
        user_id=user_id,
        session_id=session_id,
        voxel_spacing=voxel_spacing,
        additional_params=additional_params,
    )


def _is_regex_pattern(pattern: str) -> bool:
    """Check if string is a regex pattern."""
    regex_chars = r"[.*+?^${}()|[\]\\"
    has_regex_chars = any(char in pattern for char in regex_chars)

    if not has_regex_chars:
        return False

    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


def discover_tasks_for_run(run: "CopickRun", selector_config: SelectorConfig) -> List[Dict[str, Any]]:
    """
    Discover conversion tasks for a single run using selector configuration.

    Args:
        run: CopickRun to discover tasks for
        selector_config: Pydantic SelectorConfig model

    Returns:
        List of task dictionaries for this run
    """
    # Use copick's official URI resolution for proper pattern matching
    from copick.util.uri import get_copick_objects_by_type

    # Determine pattern type based on session_id
    pattern_type = "regex" if _is_regex_pattern(selector_config.input_session_id) else "glob"

    # Build filter dict based on input type
    filters = {"pattern_type": pattern_type}

    if selector_config.input_type == "picks" or selector_config.input_type == "mesh":
        filters["object_name"] = selector_config.input_object_name
        filters["user_id"] = selector_config.input_user_id
        filters["session_id"] = selector_config.input_session_id
    elif selector_config.input_type == "segmentation":
        filters["name"] = selector_config.input_object_name
        filters["user_id"] = selector_config.input_user_id
        filters["session_id"] = selector_config.input_session_id
        filters["voxel_spacing"] = selector_config.voxel_spacing

    # Find matching input objects using copick's official resolution
    matching_inputs = get_copick_objects_by_type(
        root=run.root,
        object_type=selector_config.input_type,
        run_name=run.name,
        **filters,
    )

    if not matching_inputs:
        return []

    # Generate type-specific input parameter name
    if selector_config.input_type == "mesh":
        input_param_name = "mesh"
    elif selector_config.input_type == "segmentation":
        input_param_name = "segmentation"
    elif selector_config.input_type == "picks":
        input_param_name = "picks"
    else:
        input_param_name = "input_object"  # fallback

    tasks = []
    for input_object in matching_inputs:
        # Resolve output session ID from template
        resolved_session_id = selector_config.output_session_id.replace(
            "{input_session_id}",
            input_object.session_id,
        )

        task = {
            input_param_name: input_object,  # Use type-specific parameter name
            "output_object_name": selector_config.output_object_name,
            "output_user_id": selector_config.output_user_id,
            "output_session_id": resolved_session_id,
            "individual_outputs": selector_config.individual_outputs,
            "input_type": selector_config.input_type,
            "output_type": selector_config.output_type,
            "segmentation_name": selector_config.segmentation_name,
            "voxel_spacing": selector_config.voxel_spacing,
        }

        # Add session ID template for individual outputs
        if selector_config.individual_outputs:
            task["session_id_template"] = resolved_session_id

        tasks.append(task)

    return tasks


def add_references_to_tasks(
    run: "CopickRun",
    tasks: List[Dict[str, Any]],
    reference_config: ReferenceConfig,
) -> List[Dict[str, Any]]:
    """
    Add reference object information to tasks for distance operations.

    Args:
        run: CopickRun to search for references
        tasks: List of tasks to augment
        reference_config: Pydantic ReferenceConfig model

    Returns:
        List of tasks with reference information added
    """
    reference_type = reference_config.reference_type

    # Find reference objects
    if reference_type == "mesh":
        ref_objects = run.get_meshes(
            object_name=reference_config.object_name,
            user_id=reference_config.user_id,
            session_id=reference_config.session_id,
        )
        ref_key = "reference_mesh"
        alt_key = "reference_segmentation"
    else:  # segmentation
        ref_objects = run.get_segmentations(
            name=reference_config.object_name,
            user_id=reference_config.user_id,
            session_id=reference_config.session_id,
            voxel_size=reference_config.voxel_spacing,
        )
        ref_key = "reference_segmentation"
        alt_key = "reference_mesh"

    if not ref_objects:
        logger.warning(f"No reference {reference_type} found in run {run.name}")
        return []

    # Add reference information to all tasks
    augmented_tasks = []
    for task in tasks:
        task[ref_key] = ref_objects[0]
        task[alt_key] = None

        # Add additional reference parameters
        for key, value in reference_config.additional_params.items():
            task[key] = value

        augmented_tasks.append(task)

    return augmented_tasks


def pair_tasks_within_run(
    tasks1: List[Dict[str, Any]],
    tasks2: List[Dict[str, Any]],
    input_type: str = "segmentation",
) -> List[Dict[str, Any]]:
    """
    Pair tasks from two selectors within a single run for boolean operations.

    Args:
        tasks1: Tasks from first selector
        tasks2: Tasks from second selector
        input_type: Type of input objects to determine parameter names

    Returns:
        List of paired tasks for boolean operations
    """
    # Generate type-specific parameter names
    if input_type == "mesh":
        param1, param2 = "mesh1", "mesh2"
    elif input_type == "segmentation":
        param1, param2 = "segmentation1", "segmentation2"
    elif input_type == "picks":
        param1, param2 = "picks1", "picks2"
    else:
        # Fallback to generic names
        param1, param2 = "input_object1", "input_object2"

    paired_tasks = []

    # Determine input key based on type (matches discover_tasks_for_run)
    if input_type == "mesh":
        input_key = "mesh"
    elif input_type == "segmentation":
        input_key = "segmentation"
    elif input_type == "picks":
        input_key = "picks"
    else:
        input_key = "input_object"

    # Pair in order (same logic as current segop.py)
    for i, task1 in enumerate(tasks1):
        if i < len(tasks2):
            task2 = tasks2[i]

            # Create combined task for boolean operation with type-specific parameter names
            paired_task = {
                param1: task1[input_key],
                param2: task2[input_key],
                "object_name": task1["output_object_name"],
                "user_id": task1["output_user_id"],
                "session_id": task1["output_session_id"],
                # Copy other parameters from task1
                "voxel_spacing": task1.get("voxel_spacing"),
                "is_multilabel": False,  # Boolean ops work on binary
            }
            paired_tasks.append(paired_task)

    return paired_tasks


def pair_multi_tasks_within_run(
    tasks_list: List[List[Dict[str, Any]]],
    input_type: str = "segmentation",
) -> List[Dict[str, Any]]:
    """
    Pair tasks from N selectors within a single run for N-way operations.

    Args:
        tasks_list: List of task lists from N selectors
        input_type: Type of input objects

    Returns:
        List of N-way paired tasks with inputs as a list
    """
    # Use plural parameter name for N inputs
    if input_type == "mesh":
        param_name = "meshes"
    elif input_type == "segmentation":
        param_name = "segmentations"
    else:
        param_name = "inputs"

    paired_tasks = []

    # Pair in order across all selectors
    min_length = min(len(tasks) for tasks in tasks_list) if tasks_list else 0

    for i in range(min_length):
        # Collect all input objects at this index
        input_objects = []
        for tasks in tasks_list:
            task = tasks[i]
            # Extract input object using various possible keys
            input_obj = task.get("segmentation") or task.get("mesh") or task.get("picks") or task.get("input_object")
            input_objects.append(input_obj)

        # Create combined task with list of inputs
        first_task = tasks_list[0][i]
        paired_task = {
            param_name: input_objects,  # List of N input objects
            "object_name": first_task["output_object_name"],
            "user_id": first_task["output_user_id"],
            "session_id": first_task["output_session_id"],
            "voxel_spacing": first_task.get("voxel_spacing"),
            "is_multilabel": False,
        }

        paired_tasks.append(paired_task)

    return paired_tasks


def lazy_conversion_worker(
    run: "CopickRun",
    config: TaskConfig,
    converter_func: Callable,
    **converter_kwargs,
) -> Dict[str, Any]:
    """
    Universal lazy worker that discovers and processes tasks for a single run.

    Args:
        run: CopickRun to process
        config: Pydantic TaskConfig model with validated configuration
        converter_func: Converter function to call
        **converter_kwargs: Additional arguments for converter

    Returns:
        Processing results dictionary
    """
    try:
        if config.type == "single_selector":
            # Simple conversion command
            tasks = discover_tasks_for_run(run, config.selector)

        elif config.type == "single_selector_with_reference":
            # Distance-based command
            tasks = discover_tasks_for_run(run, config.selector)
            tasks = add_references_to_tasks(run, tasks, config.reference)

        elif config.type == "dual_selector":
            # Boolean operation command
            tasks1 = discover_tasks_for_run(run, config.selectors[0])
            tasks2 = discover_tasks_for_run(run, config.selectors[1])
            # Use input type from first selector to determine parameter names
            input_type = config.selectors[0].input_type
            tasks = pair_tasks_within_run(tasks1, tasks2, input_type)

            # Add additional parameters to all tasks
            if config.additional_params:
                for task in tasks:
                    task.update(config.additional_params)

        elif config.type == "multi_selector":
            # N-way operation (N≥2)
            # Discover tasks for all selectors
            tasks_list = []
            for selector in config.selectors:
                selector_tasks = discover_tasks_for_run(run, selector)
                tasks_list.append(selector_tasks)

            # Use input type from first selector
            input_type = config.selectors[0].input_type
            tasks = pair_multi_tasks_within_run(tasks_list, input_type)

            # Add additional parameters
            if config.additional_params:
                for task in tasks:
                    task.update(config.additional_params)

        elif config.type == "single_selector_multi_union":
            # Single input pattern that expands to N-way union
            discovered_tasks = discover_tasks_for_run(run, config.selector)

            if len(discovered_tasks) < 2:
                # Not enough matches for union operation
                return {
                    "processed": 0,
                    "errors": [
                        f"Pattern matched {len(discovered_tasks)} segmentation(s) in {run.name}, but union requires at least 2",
                    ],
                }

            # Extract all input objects from discovered tasks
            input_type = config.selector.input_type
            if input_type == "segmentation":
                param_name = "segmentations"
                input_key = "segmentation"
            elif input_type == "mesh":
                param_name = "meshes"
                input_key = "mesh"
            else:
                param_name = "inputs"
                input_key = "input_object"

            input_objects = [task[input_key] for task in discovered_tasks]

            # Create single N-way task from all matched objects
            first_task = discovered_tasks[0]
            tasks = [
                {
                    param_name: input_objects,
                    "object_name": first_task["output_object_name"],
                    "user_id": first_task["output_user_id"],
                    "session_id": first_task["output_session_id"],
                    "voxel_spacing": first_task.get("voxel_spacing"),
                    "is_multilabel": False,
                },
            ]

        else:
            raise ValueError(f"Unknown config type: {config.type}")

        if not tasks:
            return {"processed": 0, "errors": [f"No tasks found for {run.name}"]}

        # Process all discovered tasks for this run
        total_processed = 0
        all_errors = []
        accumulated_stats = {}

        for task in tasks:
            try:
                # Call converter function with task parameters
                task_params = dict(task)
                task_params["run"] = run
                task_params.update(converter_kwargs)

                result = converter_func(**task_params)

                if result:
                    output_obj, stats = result
                    total_processed += 1

                    # Accumulate stats
                    for key, value in stats.items():
                        if key not in accumulated_stats:
                            accumulated_stats[key] = 0
                        accumulated_stats[key] += value
                else:
                    # Try to find the input object using different possible parameter names
                    input_obj = (
                        task.get("input_object")
                        or task.get("segmentation")
                        or task.get("mesh")
                        or task.get("picks")
                        or task.get("segmentation1")
                        or task.get("mesh1")
                        or task.get("picks1")
                    )
                    session_id = getattr(input_obj, "session_id", "unknown")
                    all_errors.append(f"No output generated for {session_id} in {run.name}")

            except Exception as e:
                logger.exception(f"Error processing task in {run.name}: {e}")
                all_errors.append(f"Error processing task in {run.name}: {e}")

        return {
            "processed": total_processed,
            "errors": all_errors,
            **accumulated_stats,
        }

    except Exception as e:
        logger.exception(f"Error in lazy worker for {run.name}: {e}")
        return {"processed": 0, "errors": [f"Worker error in {run.name}: {e}"]}


def create_lazy_batch_converter(
    converter_func: Callable,
    task_description: str,
) -> Callable:
    """
    Create a lazy batch converter that does parallel task discovery and processing.

    Args:
        converter_func: The converter function to call for each task
        task_description: Description for progress bar

    Returns:
        Lazy batch converter function
    """

    def lazy_batch_converter(
        root: "CopickRoot",
        config: TaskConfig,
        run_names: Optional[List[str]] = None,
        workers: int = 8,
        **converter_kwargs,
    ) -> Dict[str, Any]:
        """
        Lazy batch converter with parallel task discovery.

        Args:
            root: The copick root containing runs to process
            config: Validated TaskConfig Pydantic model
            run_names: List of run names to process. If None, processes all runs.
            workers: Number of worker processes
            **converter_kwargs: Additional arguments passed to converter function

        Returns:
            Dictionary with processing results and statistics
        """
        from copick.ops.run import map_runs

        runs_to_process = [run.name for run in root.runs] if run_names is None else run_names

        if not runs_to_process:
            return {}

        # Create worker function for this specific converter
        def run_worker(run: "CopickRun", **kwargs) -> Dict[str, Any]:
            return lazy_conversion_worker(
                run=run,
                config=config,
                converter_func=converter_func,
                **converter_kwargs,
            )

        # Execute in parallel - no sequential discovery!
        results = map_runs(
            callback=run_worker,
            root=root,
            runs=runs_to_process,
            workers=workers,
            task_desc=task_description,
        )

        return results

    return lazy_batch_converter
