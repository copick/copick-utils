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

        # Use bincount for O(n) counting of all components in one pass
        counts = np.bincount(labeled_array.ravel())
        # counts[0] is background within this label mask, skip it
        for component_id in range(1, num_components + 1):
            component_voxels = int(counts[component_id])
            components.append(
                {
                    "label": int(label_value),
                    "component_id": component_id,
                    "volume_voxels": component_voxels,
                    "volume_angstroms3": component_voxels * voxel_volume,
                },
            )

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
    input_uri: str,
    connectivity: str,
) -> Dict[str, Any]:
    """Worker function for batch segmentation stats.

    Uses resolve_copick_objects for proper URI resolution with pattern support.
    """
    from copick.util.uri import resolve_copick_objects

    try:
        segmentations = resolve_copick_objects(input_uri, run.root, "segmentation", run_name=run.name)

        if not segmentations:
            return {"processed": 0, "components": [], "errors": [f"No segmentation found for {run.name}"]}

        all_components = []
        for segmentation in segmentations:
            # Use the actual voxel_size from the segmentation for volume calculations
            components = analyze_segmentation_components(
                segmentation=segmentation,
                voxel_spacing=segmentation.voxel_size,
                connectivity=connectivity,
            )

            if components:
                for comp in components:
                    comp["run"] = run.name
                    comp["voxel_spacing"] = segmentation.voxel_size
                all_components.extend(components)

        return {
            "processed": 1,
            "components": all_components,
            "errors": [],
        }

    except Exception as e:
        return {"processed": 0, "components": [], "errors": [f"Error processing {run.name}: {e}"]}


def seg_stats_batch(
    root: "CopickRoot",
    input_uri: str,
    connectivity: str = "all",
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Batch analyze connected component sizes across multiple runs.

    Args:
        root: The copick root containing runs to process.
        input_uri: Copick URI for the segmentation(s) to analyze. Supports patterns.
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
        input_uri=input_uri,
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

    fieldnames = ["run", "label", "component_id", "volume_voxels", "volume_angstroms3", "voxel_spacing"]

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_components)

    logger.info(f"Exported {len(all_components)} components to {output_path}")


def _compute_bin_size(all_components: List[Dict[str, Any]]) -> float:
    """Compute histogram bin size as the volume of 10 cubic voxels.

    Uses the voxel_spacing from the first component. Falls back to a reasonable default.
    """
    voxel_spacing = all_components[0].get("voxel_spacing")
    if voxel_spacing and voxel_spacing > 0:
        return 10.0 * (voxel_spacing**3)
    # Fallback: use 1% of the range
    volumes = [c["volume_angstroms3"] for c in all_components]
    vol_range = max(volumes) - min(volumes)
    return max(vol_range / 100.0, 1.0)


def _suppress_kaleido_logs():
    """Suppress verbose kaleido/chromium logging."""
    import logging

    for name in ("kaleido", "chromium", "kaleido._kaleido_tab", "kaleido.kaleido", "kaleido.browser_async"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _write_figure(fig, output: Path) -> None:
    """Write a plotly figure to file, format inferred from extension."""
    _suppress_kaleido_logs()

    ext = output.suffix.lower()
    if ext == ".html":
        fig.write_html(str(output))
    elif ext in (".png", ".pdf", ".svg", ".jpeg", ".webp"):
        fig.write_image(str(output))
    else:
        fig.write_html(str(output))
        logger.warning(f"Unknown extension '{ext}', defaulting to HTML output")


def export_stats_plot(results: Dict[str, Any], output_path: str) -> None:
    """
    Export component statistics as histogram plots.

    Creates a combined histogram of all labels plus individual per-label histograms.
    For PDF output, all plots are arranged vertically in a single document.
    Uses consistent bin sizes (10 cubic voxels) and logarithmic y-axis.

    Args:
        results: Results from seg_stats_batch.
        output_path: Path to output plot file.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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
    labels = sorted({c["label"] for c in all_components})
    bin_size = _compute_bin_size(all_components)

    # Compute global x-axis range for consistent axes
    all_volumes = [c["volume_angstroms3"] for c in all_components]
    x_min = 0
    x_max = max(all_volumes) * 1.05

    # Create subplots: combined + one per label
    n_plots = 1 + len(labels)
    subplot_titles = ["All Labels (Combined)"] + [f"Label {lv}" for lv in labels]

    fig = make_subplots(
        rows=n_plots,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.3 / n_plots,
    )

    # Combined plot (row 1)
    for label_value in labels:
        volumes = [c["volume_angstroms3"] for c in all_components if c["label"] == label_value]
        fig.add_trace(
            go.Histogram(
                x=volumes,
                name=f"Label {label_value}",
                opacity=0.7,
                xbins=dict(start=x_min, end=x_max, size=bin_size),
            ),
            row=1,
            col=1,
        )

    # Individual per-label plots
    for i, label_value in enumerate(labels):
        volumes = [c["volume_angstroms3"] for c in all_components if c["label"] == label_value]
        fig.add_trace(
            go.Histogram(
                x=volumes,
                name=f"Label {label_value}",
                opacity=0.7,
                xbins=dict(start=x_min, end=x_max, size=bin_size),
                showlegend=False,
            ),
            row=i + 2,
            col=1,
        )

    # Apply log y-axis and consistent x-axis range to all subplots
    for i in range(1, n_plots + 1):
        fig.update_xaxes(title_text="Volume (Å³)", range=[x_min, x_max], row=i, col=1)
        fig.update_yaxes(title_text="Count", type="log", row=i, col=1)

    fig.update_layout(
        barmode="overlay",
        template="plotly_white",
        legend_title="Label",
        height=400 * n_plots,
        width=1000,
    )

    _write_figure(fig, output)
    logger.info(f"Exported plot ({n_plots} panels) to {output_path}")
