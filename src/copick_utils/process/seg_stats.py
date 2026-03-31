"""Analyze connected component sizes per label in segmentations."""

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from copick.util.log import get_logger
from scipy.ndimage import generate_binary_structure, label

if TYPE_CHECKING:
    from copick.models import CopickRoot, CopickRun, CopickSegmentation

logger = get_logger(__name__)


def _analyze_components_single(
    seg: np.ndarray,
    voxel_spacing: float,
    connectivity: str = "all",
) -> List[Dict[str, Any]]:
    """
    Analyze connected components per label in a segmentation.

    Args:
        seg: Label image (integer numpy array).
        voxel_spacing: Voxel spacing in angstroms.
        connectivity: Connectivity for connected components.
                     "face" = 6-connected, "face-edge" = 18-connected, "all" = 26-connected.

    Returns:
        List of dicts, one per component:
        [{label, component_id, volume_voxels, volume_angstroms3}, ...]
    """
    connectivity_map = {
        "face": 1,
        "face-edge": 2,
        "all": 3,
    }
    connectivity_value = connectivity_map.get(connectivity, 3)
    struct = generate_binary_structure(seg.ndim, connectivity_value)
    voxel_volume = voxel_spacing**3

    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels != 0]  # Skip background

    components = []

    for label_value in unique_labels:
        binary_mask = seg == label_value
        labeled_array, num_components = label(binary_mask, structure=struct)

        for component_id in range(1, num_components + 1):
            component_voxels = int(np.sum(labeled_array == component_id))
            components.append({
                "label": int(label_value),
                "component_id": component_id,
                "volume_voxels": component_voxels,
                "volume_angstroms3": component_voxels * voxel_volume,
            })

    return components


def analyze_segmentation_components(
    segmentation: "CopickSegmentation",
    voxel_spacing: float,
    connectivity: str = "all",
) -> Optional[List[Dict[str, Any]]]:
    """
    Analyze connected components in a CopickSegmentation.

    Args:
        segmentation: Input CopickSegmentation object.
        voxel_spacing: Voxel spacing in angstroms.
        connectivity: Connectivity for connected components.

    Returns:
        List of component dicts, or None if loading failed.
    """
    try:
        seg_array = segmentation.numpy()

        if seg_array is None:
            logger.error("Could not load segmentation data")
            return None

        if seg_array.size == 0:
            logger.error("Empty segmentation data")
            return None

        return _analyze_components_single(seg_array, voxel_spacing=voxel_spacing, connectivity=connectivity)

    except Exception as e:
        logger.error(f"Error analyzing segmentation components: {e}")
        return None


def _seg_stats_worker(
    run: "CopickRun",
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str,
    voxel_spacing: float,
    connectivity: str,
    root: "CopickRoot",
) -> Dict[str, Any]:
    """Worker function for batch segmentation stats."""
    try:
        segmentations = run.get_segmentations(
            name=segmentation_name,
            user_id=segmentation_user_id,
            session_id=segmentation_session_id,
            voxel_size=voxel_spacing,
        )

        if not segmentations:
            return {"processed": 0, "components": [], "errors": [f"No segmentation found for {run.name}"]}

        segmentation = segmentations[0]

        components = analyze_segmentation_components(
            segmentation=segmentation,
            voxel_spacing=voxel_spacing,
            connectivity=connectivity,
        )

        if components is None:
            return {"processed": 0, "components": [], "errors": [f"Failed to analyze {run.name}"]}

        # Tag each component with the run name
        for comp in components:
            comp["run"] = run.name

        return {
            "processed": 1,
            "components": components,
            "errors": [],
        }

    except Exception as e:
        return {"processed": 0, "components": [], "errors": [f"Error processing {run.name}: {e}"]}


def seg_stats_batch(
    root: "CopickRoot",
    segmentation_name: str,
    segmentation_user_id: str,
    segmentation_session_id: str,
    voxel_spacing: float,
    connectivity: str = "all",
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Batch analyze connected component sizes across multiple runs.

    Args:
        root: The copick root containing runs to process.
        segmentation_name: Name of the segmentation to analyze.
        segmentation_user_id: User ID of the segmentation to analyze.
        segmentation_session_id: Session ID of the segmentation to analyze.
        voxel_spacing: Voxel spacing in angstroms.
        connectivity: Connectivity for connected components.
        run_names: List of run names to process. If None, processes all runs.
        workers: Number of worker processes.

    Returns:
        Dictionary with processing results per run.
    """
    from copick.ops.run import map_runs

    runs_to_process = [run.name for run in root.runs] if run_names is None else run_names

    results = map_runs(
        callback=_seg_stats_worker,
        root=root,
        runs=runs_to_process,
        workers=workers,
        task_desc="Analyzing segmentation components",
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        segmentation_session_id=segmentation_session_id,
        voxel_spacing=voxel_spacing,
        connectivity=connectivity,
    )

    return results


def export_stats_csv(results: Dict[str, Any], output_path: str) -> None:
    """
    Export component statistics to a CSV file.

    Args:
        results: Results from seg_stats_batch.
        output_path: Path to output CSV file.
    """
    all_components = []
    for run_result in results.values():
        if run_result and run_result.get("components"):
            all_components.extend(run_result["components"])

    if not all_components:
        logger.warning("No components to export")
        return

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["run", "label", "component_id", "volume_voxels", "volume_angstroms3"]

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_components)

    logger.info(f"Exported {len(all_components)} components to {output_path}")


def export_stats_plot(results: Dict[str, Any], output_path: str) -> None:
    """
    Export component statistics as a plot.

    Creates a histogram of component volumes per label. Output format is
    inferred from the file extension (.html, .png, .pdf, .svg).

    Args:
        results: Results from seg_stats_batch.
        output_path: Path to output plot file.
    """
    import plotly.graph_objects as go

    all_components = []
    for run_result in results.values():
        if run_result and run_result.get("components"):
            all_components.extend(run_result["components"])

    if not all_components:
        logger.warning("No components to plot")
        return

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Group by label
    labels = sorted(set(c["label"] for c in all_components))

    fig = go.Figure()

    for label_value in labels:
        volumes = [c["volume_angstroms3"] for c in all_components if c["label"] == label_value]
        fig.add_trace(go.Histogram(
            x=volumes,
            name=f"Label {label_value}",
            opacity=0.7,
        ))

    fig.update_layout(
        title="Connected Component Volume Distribution",
        xaxis_title="Volume (Å³)",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_white",
        legend_title="Label",
    )

    ext = output.suffix.lower()
    if ext == ".html":
        fig.write_html(str(output))
    elif ext in (".png", ".pdf", ".svg", ".jpeg", ".webp"):
        fig.write_image(str(output))
    else:
        # Default to HTML
        fig.write_html(str(output))
        logger.warning(f"Unknown extension '{ext}', defaulting to HTML output")

    logger.info(f"Exported plot to {output_path}")
