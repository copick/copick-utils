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
    include_background: bool = True,
) -> List[Dict[str, Any]]:
    """
    Analyze connected components per label in a segmentation.

    Args:
        seg: Label image (integer numpy array).
        voxel_spacing: Voxel spacing in angstroms.
        connectivity: Connectivity for connected components.
                     "face" = 6-connected, "face-edge" = 18-connected, "all" = 26-connected.
        include_background: If True, also analyze connected components of the background (label 0).

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

    all_unique_labels = np.unique(seg)
    foreground_labels = all_unique_labels[all_unique_labels != 0]

    components = []

    for label_value in foreground_labels:
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

    # Analyze background connected components (label 0)
    if include_background and 0 in all_unique_labels:
        background_mask = seg == 0
        labeled_bg, num_bg_components = label(background_mask, structure=struct)

        if num_bg_components > 0:
            bg_counts = np.bincount(labeled_bg.ravel())
            for component_id in range(1, num_bg_components + 1):
                component_voxels = int(bg_counts[component_id])
                components.append(
                    {
                        "label": 0,
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
    include_background: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    """
    Analyze connected components in a CopickSegmentation.

    Args:
        segmentation: Input CopickSegmentation object.
        voxel_spacing: Voxel spacing in angstroms.
        connectivity: Connectivity for connected components.
        include_background: If True, also analyze background (label 0) components.

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

        return _analyze_components_single(
            seg_array,
            voxel_spacing=voxel_spacing,
            connectivity=connectivity,
            include_background=include_background,
        )

    except Exception as e:
        logger.error(f"Error analyzing segmentation components: {e}")
        return None


def _seg_stats_worker(
    run: "CopickRun",
    input_uri: str,
    connectivity: str,
    include_background: bool = True,
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
                include_background=include_background,
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
    include_background: bool = True,
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Batch analyze connected component sizes across multiple runs.

    Args:
        root: The copick root containing runs to process.
        input_uri: Copick URI for the segmentation(s) to analyze. Supports patterns.
        connectivity: Connectivity for connected components.
        include_background: If True, also analyze background (label 0) components.
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
        include_background=include_background,
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


def _compute_log_bins(all_components: List[Dict[str, Any]], n_bins: int = 50):
    """Compute logarithmically spaced bin edges for histograms.

    Log-spaced bins handle the typical distribution where there are many small
    components and few very large ones.
    """
    all_volumes = np.array([c["volume_angstroms3"] for c in all_components])
    positive = all_volumes[all_volumes > 0]
    if len(positive) == 0:
        return np.linspace(0, 1, n_bins + 1)
    v_min = positive.min()
    v_max = positive.max() * 1.05
    return np.geomspace(v_min, v_max, n_bins + 1)


def export_stats_plot(results: Dict[str, Any], output_path: str, root: "CopickRoot" = None) -> None:
    """
    Export component statistics as histogram plots using matplotlib.

    Creates a combined histogram (all labels overlaid) plus individual per-label
    histograms. For PDF, each plot is a separate page. For image formats (png, svg),
    all plots are arranged as subplots in a single figure.

    Labels are named and colored according to the copick project's pickable objects.

    Args:
        results: Results from seg_stats_batch.
        output_path: Path to output file (.pdf, .png, .svg, .jpg).
        root: CopickRoot for looking up object names and colors by label.
    """
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    all_components = []
    for run_result in results.values():
        if run_result and run_result.get("components"):
            all_components.extend(run_result["components"])

    if not all_components:
        logger.warning("No components to plot")
        return

    # Separate foreground and background components — background gets its own plot
    # but is excluded from the combined overview plot
    foreground_components = [c for c in all_components if c["label"] != 0]
    background_components = [c for c in all_components if c["label"] == 0]

    if not foreground_components and not background_components:
        logger.warning("No components to plot")
        return

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fg_labels = sorted({c["label"] for c in foreground_components}) if foreground_components else []
    fg_bins = _compute_log_bins(foreground_components) if foreground_components else None
    bg_bins = _compute_log_bins(background_components) if background_components else None

    # Build label → name and label → color mappings from copick objects
    label_names = {}
    label_colors = {}
    if root is not None:
        for obj in root.pickable_objects:
            if obj.label is not None:
                label_names[obj.label] = obj.name
                if obj.color is not None:
                    # RGBA (0-255) → matplotlib RGBA (0-1)
                    label_colors[obj.label] = tuple(c / 255.0 for c in obj.color)

    def _label_display(lv):
        return label_names.get(lv, f"Label {lv}")

    def _label_color(lv):
        return label_colors.get(lv, None)

    # Group foreground volumes by label
    volumes_by_label = {}
    for label_value in fg_labels:
        volumes_by_label[label_value] = [
            c["volume_angstroms3"] for c in foreground_components if c["label"] == label_value
        ]

    bg_volumes = [c["volume_angstroms3"] for c in background_components] if background_components else []

    # Get voxel volume for secondary axis (cubic voxels)
    ref_components = foreground_components or background_components
    voxel_spacing = ref_components[0].get("voxel_spacing")
    voxel_volume = voxel_spacing**3 if voxel_spacing and voxel_spacing > 0 else None

    def _add_voxel_axis(ax):
        """Add a secondary x-axis showing volume in cubic voxels."""
        if voxel_volume is None:
            return
        ax2 = ax.secondary_xaxis("top", functions=(lambda x: x / voxel_volume, lambda x: x * voxel_volume))
        ax2.set_xlabel("Volume (voxels³)")

    def _setup_ax(ax, title):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Volume (Å³)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        _add_voxel_axis(ax)

    ext = output.suffix.lower()

    n_panels = (1 + len(fg_labels) if fg_labels else 0) + (1 if bg_volumes else 0)

    if ext == ".pdf":
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(str(output)) as pdf:
            # Combined foreground plot
            if fg_labels:
                fig, ax = plt.subplots(figsize=(10, 5))
                for lv in fg_labels:
                    ax.hist(
                        volumes_by_label[lv],
                        bins=fg_bins,
                        alpha=0.7,
                        label=_label_display(lv),
                        color=_label_color(lv),
                    )
                _setup_ax(ax, "All Labels (Combined)")
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

                # Individual per-label plots
                for lv in fg_labels:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.hist(volumes_by_label[lv], bins=fg_bins, alpha=0.7, color=_label_color(lv))
                    _setup_ax(ax, _label_display(lv))
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

            # Background plot (separate page, own bins)
            if bg_volumes:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(bg_volumes, bins=bg_bins, alpha=0.7, color="gray")
                _setup_ax(ax, "Background (Label 0)")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    else:
        # Image formats: subplots in one figure
        n_plots = n_panels
        fig, axes = plt.subplots(max(n_plots, 1), 1, figsize=(10, 5 * max(n_plots, 1)))
        if n_plots <= 1:
            axes = [axes]

        idx = 0

        # Combined foreground plot
        if fg_labels:
            for lv in fg_labels:
                axes[idx].hist(
                    volumes_by_label[lv],
                    bins=fg_bins,
                    alpha=0.7,
                    label=_label_display(lv),
                    color=_label_color(lv),
                )
            _setup_ax(axes[idx], "All Labels (Combined)")
            axes[idx].legend()
            idx += 1

            # Individual per-label plots
            for lv in fg_labels:
                axes[idx].hist(volumes_by_label[lv], bins=fg_bins, alpha=0.7, color=_label_color(lv))
                _setup_ax(axes[idx], _label_display(lv))
                idx += 1

        # Background plot (separate, own bins)
        if bg_volumes:
            axes[idx].hist(bg_volumes, bins=bg_bins, alpha=0.7, color="gray")
            _setup_ax(axes[idx], "Background (Label 0)")

        fig.tight_layout()
        fig.savefig(str(output), dpi=150)
        plt.close(fig)

    logger.info(f"Exported plot ({n_panels} panels) to {output_path}")
