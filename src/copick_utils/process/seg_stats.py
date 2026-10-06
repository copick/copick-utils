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


def _component_rows(
    labeled: np.ndarray,
    voxel_spacing: float,
    skeleton: bool = False,
    ids: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    One row per positive value of a labeled array (a component, or an instance).

    Args:
        labeled: Integer array; each positive value is one component or instance.
        voxel_spacing: Voxel spacing in angstroms.
        skeleton: Also measure each one's skeleton (length, label radius, branches, junctions, endpoints).
        ids: The values to report (default: every positive value present).

    Returns:
        List of dicts with component_id, volume_voxels, volume_angstroms3 (and skeleton columns).
    """
    from scipy.ndimage import find_objects

    counts = np.bincount(labeled.ravel()) if labeled.size else np.zeros(1, dtype=int)
    if ids is None:
        ids = np.flatnonzero(counts[1:]) + 1
    slices = find_objects(labeled) if skeleton else None
    voxel_volume = voxel_spacing**3
    rows = []
    for value in ids:
        value = int(value)
        voxels = int(counts[value]) if value < len(counts) else 0
        row = {"component_id": value, "volume_voxels": voxels, "volume_angstroms3": voxels * voxel_volume}
        if skeleton:
            from copick_utils.process.skeleton_graph import summarize

            box = slices[value - 1] if value - 1 < len(slices) else None
            summary = summarize(labeled[box] == value) if box is not None else summarize(np.zeros((1, 1, 1), bool))
            row.update(
                {
                    "skeleton_length_angstroms": summary.length * voxel_spacing,
                    "label_radius_angstroms": None if summary.radius is None else summary.radius * voxel_spacing,
                    "n_branches": summary.n_branches,
                    "n_junctions": summary.n_junctions,
                    "n_endpoints": summary.n_endpoints,
                },
            )
        rows.append(row)
    return rows


def _analyze_components_single(
    seg: np.ndarray,
    voxel_spacing: float,
    connectivity: str = "all",
    include_background: bool = True,
    skeleton: bool = False,
) -> List[Dict[str, Any]]:
    """
    Analyze connected components per label in a segmentation.

    Args:
        seg: Label image (integer numpy array).
        voxel_spacing: Voxel spacing in angstroms.
        connectivity: Connectivity for connected components.
                     "face" = 6-connected, "face-edge" = 18-connected, "all" = 26-connected.
        include_background: If True, also analyze connected components of the background (label 0).
        skeleton: If True, also measure each foreground component's skeleton.

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

    all_unique_labels = np.unique(seg)
    foreground_labels = all_unique_labels[all_unique_labels != 0]

    components = []
    for label_value in foreground_labels:
        labeled_array, _ = label(seg == label_value, structure=struct)
        for row in _component_rows(labeled_array, voxel_spacing, skeleton=skeleton):
            components.append({"label": int(label_value), **row})

    # Analyze background connected components (label 0)
    if include_background and 0 in all_unique_labels:
        labeled_bg, _ = label(seg == 0, structure=struct)
        for row in _component_rows(labeled_bg, voxel_spacing):
            components.append({"label": 0, **row})

    return components


def _analyze_instances(
    instances: np.ndarray,
    voxel_spacing: float,
    label_value: int,
    skeleton: bool = False,
) -> List[Dict[str, Any]]:
    """One row per instance ID of an instance segmentation (an instance may span several pieces)."""
    return [
        {"label": int(label_value), "instance_id": row["component_id"], **row}
        for row in _component_rows(instances, voxel_spacing, skeleton=skeleton)
    ]


def _analyze_panoptic(
    panoptic: np.ndarray,
    voxel_spacing: float,
    connectivity: str = "all",
    include_background: bool = True,
    skeleton: bool = False,
) -> List[Dict[str, Any]]:
    """Rows for a panoptic segmentation: one per (label, instance) segment, and the connected components of each
    label's region without instances (instance 0) and of the background."""
    labels, instances = panoptic[0], panoptic[1]
    rows = []
    for label_value in np.unique(labels[labels > 0]):
        in_label = labels == label_value
        ids = np.unique(instances[in_label])
        ids = ids[ids > 0]
        if len(ids):
            per_label = np.where(in_label, instances, 0)
            rows.extend(_analyze_instances(per_label, voxel_spacing, int(label_value), skeleton=skeleton))
        stuff = in_label & (instances == 0)
        if stuff.any():
            rows.extend(
                _analyze_components_single(
                    stuff.astype(np.uint8) * int(label_value),
                    voxel_spacing,
                    connectivity=connectivity,
                    include_background=False,
                    skeleton=skeleton,
                ),
            )
    if include_background:
        rows.extend(
            r
            for r in _analyze_components_single(
                (labels > 0).astype(np.uint8),
                voxel_spacing,
                connectivity=connectivity,
                include_background=True,
            )
            if r["label"] == 0
        )
    return rows


def analyze_segmentation_components(
    segmentation: "CopickSegmentation",
    voxel_spacing: float,
    connectivity: str = "all",
    include_background: bool = True,
    skeleton: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """
    Analyze connected components in a CopickSegmentation.

    Binary and multilabel segmentations give one row per connected component of each label. An instance
    segmentation gives one row per instance ID, and a panoptic segmentation one row per (label, instance) segment
    plus the connected components of regions without instances.

    Args:
        segmentation: Input CopickSegmentation object.
        voxel_spacing: Voxel spacing in angstroms.
        connectivity: Connectivity for connected components.
        include_background: If True, also analyze background (label 0) components.
        skeleton: If True, also measure each component's or instance's skeleton.

    Returns:
        List of component dicts, or None if loading failed.
    """
    from copick_utils.util.segmentations import segmentation_type

    try:
        seg_array = segmentation.numpy()
        if seg_array is None:
            logger.error("Could not load segmentation data")
            return None
        if seg_array.size == 0:
            logger.error("Empty segmentation data")
            return None

        seg_type = segmentation_type(segmentation)
        if seg_type == "instance":
            obj = segmentation.run.root.get_object(segmentation.name)
            label_value = obj.label if obj is not None else 0
            rows = _analyze_instances(seg_array, voxel_spacing, label_value, skeleton=skeleton)
            if include_background:
                rows.extend(
                    r
                    for r in _analyze_components_single(
                        (seg_array > 0).astype(np.uint8),
                        voxel_spacing,
                        connectivity=connectivity,
                        include_background=True,
                    )
                    if r["label"] == 0
                )
        elif seg_type == "panoptic":
            rows = _analyze_panoptic(seg_array, voxel_spacing, connectivity, include_background, skeleton)
        else:
            rows = _analyze_components_single(
                seg_array,
                voxel_spacing=voxel_spacing,
                connectivity=connectivity,
                include_background=include_background,
                skeleton=skeleton,
            )
        for row in rows:
            row["segmentation_type"] = seg_type
        return rows

    except Exception as e:
        logger.error(f"Error analyzing segmentation components: {e}")
        return None


def _seg_stats_worker(
    run: "CopickRun",
    input_uri: str,
    connectivity: str,
    include_background: bool = True,
    skeleton: bool = False,
) -> Dict[str, Any]:
    """Worker function for batch segmentation stats.

    Uses resolve_copick_objects for proper URI resolution with pattern support. A URI without a type flag selects
    binary and multilabel segmentations only; ``?instance=true`` or ``?panoptic=true`` selects those types.
    """
    from copick_utils.util.segmentations import resolve_segmentations

    try:
        segmentations = resolve_segmentations(input_uri, run.root, run_name=run.name)

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
                skeleton=skeleton,
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
    skeleton: bool = False,
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
        skeleton: If True, also measure each component's or instance's skeleton.

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
        skeleton=skeleton,
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
    for optional in (
        "segmentation_type",
        "instance_id",
        "skeleton_length_angstroms",
        "label_radius_angstroms",
        "n_branches",
        "n_junctions",
        "n_endpoints",
    ):
        if any(optional in c for c in all_components):
            fieldnames.append(optional)

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
