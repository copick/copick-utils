"""Trace filaments in segmentations: copick Filaments (fitted B-spline curves) and an instance segmentation."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
from copick.util.log import get_logger

from copick_utils.converters.lazy_converter import create_lazy_batch_converter
from copick_utils.process.filament_tracing import TraceParameters, trace_centrelines

if TYPE_CHECKING:
    from copick.models import CopickFilaments, CopickRun, CopickSegmentation

logger = get_logger(__name__)

#: Default minimum label radius, as a fraction of the filament object's tube radius (see ``min_radius``).
MIN_RADIUS_FRACTION = 1.0 / 3.0


def _volume_to_trace(segmentation: "CopickSegmentation", object_name: str, label: Optional[int]):
    """The volume to trace and whether it holds instances, from the segmentation's type."""
    from copick_utils.util.segmentations import segmentation_type

    seg_type = segmentation_type(segmentation)
    if seg_type == "instance":
        return segmentation.numpy(), True
    if seg_type == "binary":
        return segmentation.numpy() > 0, False

    if label is None:
        obj = segmentation.run.root.get_object(object_name)
        if obj is None or obj.label is None:
            raise ValueError(
                f"Choose the label to trace in the {seg_type} segmentation {segmentation.name} with --label "
                f"({object_name} is not a pickable object with a label).",
            )
        label = int(obj.label)
    if seg_type == "multilabel":
        return segmentation.numpy() == label, False

    # Panoptic: the object's instances if it has any, otherwise its label region
    channels = segmentation.numpy()
    labels, instances = channels[0], channels[1]
    in_label = labels == label
    if np.any(instances[in_label] > 0):
        return np.where(in_label, instances, 0), True
    return in_label, False


def filaments_from_segmentation(
    segmentation: "CopickSegmentation",
    run: "CopickRun",
    object_name: str,
    session_id: str,
    user_id: str,
    min_volume: Optional[float] = None,
    min_length: Optional[float] = None,
    min_aspect: float = 3.0,
    min_radius: Optional[float] = None,
    prune_length: Optional[float] = None,
    junction_merge: Optional[float] = None,
    max_bend: float = 45.0,
    smoothing: Optional[float] = None,
    extend_ends: bool = True,
    fill_lumen: Optional[float] = None,
    label: Optional[int] = None,
    instances_object_name: Optional[str] = None,
    instances_user_id: Optional[str] = None,
    instances_session_id: Optional[str] = None,
    **kwargs,
) -> Optional[Tuple["CopickFilaments", Dict[str, float]]]:
    """
    Trace the filaments of a segmentation and store them as copick Filaments.

    Each filament is stored with its fitted spline as a ``bspline`` curve (copick regenerates the filament's points
    from it) and the fit's settings in ``metadata["fit"]``. IDs are 1..K by length (longest first) for a binary,
    multilabel or panoptic-region input; an instance segmentation (or a panoptic segmentation's instances) keeps
    its IDs. With ``instances_*`` set, the instance segmentation of the traced filaments is stored as well: each
    label voxel holds the ID of the nearest filament of its connected component, with the same IDs as the
    filaments.

    Args:
        segmentation: Input segmentation (binary, multilabel, instance or panoptic).
        run: CopickRun object.
        object_name: Object name of the output filaments (a filament object).
        session_id: Session ID of the output filaments.
        user_id: User ID of the output filaments.
        min_volume: Drop components smaller than this (Å³). None: no volume filter.
        min_length: Reject filaments shorter than this (Å). None: no length filter.
        min_aspect: Reject filaments shorter than this many label diameters.
        min_radius: Reject filaments whose label radius is below this (Å). None: a third of the object's radius
            (the filament's tube radius), or no radius filter if the object has none.
        prune_length: Prune skeleton spurs shorter than this (Å). None: one label diameter.
        junction_merge: Merge junctions joined by a bridge up to this long (Å). None: one label diameter.
        max_bend: Largest deviation from a straight line (degrees) to continue through a junction.
        smoothing: RMS deviation of the fitted spline from the skeleton (Å). None: half a voxel.
        extend_ends: Extend free ends to the edge of the label.
        fill_lumen: Fill holes up to this radius (Å) in each axis-aligned slice before tracing, the empty lumen of
            a tube whose wall alone is labelled. None: the object's radius; 0: no filling.
        label: Label to trace in a multilabel or panoptic segmentation (default: the object's label).
        instances_object_name: Object name of the instance segmentation to store, or None for none.
        instances_user_id: User ID of the instance segmentation.
        instances_session_id: Session ID of the instance segmentation ({input_session_id} is replaced).
        **kwargs: Additional keyword arguments from the lazy converter.

    Returns:
        Tuple of (CopickFilaments, stats dict) or None if the operation failed.
    """
    from copick.models import CopickFilamentCurve

    import copick_utils
    from copick_utils.util.segmentations import new_segmentation

    try:
        voxel_size = float(segmentation.voxel_size)
        volume, instances = _volume_to_trace(segmentation, object_name, label)
        obj = run.root.get_object(object_name)
        tube_radius = float(obj.radius) if obj is not None and obj.radius else None
        if min_radius is None and tube_radius:
            # Calibrated on dataset 10521 (easymode microtubule labels): real filaments have a label radius of
            # 40-76 Å against a 120 Å tube radius; slivers and stubs of segmentation noise 10-35 Å.
            min_radius = MIN_RADIUS_FRACTION * tube_radius
        if fill_lumen is None:
            fill_lumen = tube_radius  # a hollow label's lumen is narrower than the tube
        params = TraceParameters(
            min_volume=min_volume,
            min_length=min_length,
            min_aspect=min_aspect,
            min_radius=min_radius,
            prune_length=prune_length,
            junction_merge=junction_merge,
            max_bend=max_bend,
            smoothing=smoothing,
            extend_ends=extend_ends,
            fill_lumen=fill_lumen or None,
        )
        centrelines, assignment, report = trace_centrelines(volume, voxel_size, params, instances=instances)

        source = f"{segmentation.name}:{segmentation.user_id}/{segmentation.session_id}@{voxel_size}"
        curves, metadata = [], []
        for centreline in centrelines:
            curves.append(
                CopickFilamentCurve.from_tck(
                    centreline.tck,
                    step=voxel_size,
                    smoothing=centreline.smoothing,
                    scale=1.0,
                ),
            )
            fit = dict(centreline.metadata)
            fit.update({"tool": "copick-utils seg2fil", "version": copick_utils.__version__, "source": source})
            metadata.append({"fit": fit})

        filaments = run.new_filaments(object_name, session_id, user_id, exist_ok=True)
        filaments.from_curves(
            curves,
            instance_ids=[c.instance_id for c in centrelines],
            scores=[c.score for c in centrelines],
            radii=[c.radius for c in centrelines],
            voxel_spacing=voxel_size,
            metadata=metadata,
        )

        stats = {key: value for key, value in report.as_dict().items() if isinstance(value, (int, float))}
        stats["filaments_written"] = len(centrelines)
        if instances_object_name and not instances:
            instance_session = (instances_session_id or session_id).replace(
                "{input_session_id}",
                segmentation.session_id,
            )
            output = new_segmentation(
                run,
                voxel_size,
                instances_object_name,
                instance_session,
                instances_user_id or user_id,
                "instance",
            )
            output.from_numpy(assignment)
            stats["instance_segmentations_written"] = 1

        logger.info(
            f"Traced {len(centrelines)} filaments ({report.total_length / 10:.0f} nm) in {run.name}; "
            f"rejected {report.rejected_length} short, {report.rejected_aspect} blob-like and "
            f"{report.rejected_radius} thin",
        )
        return filaments, stats

    except Exception as e:
        logger.error(f"Error tracing filaments in {run.name}: {e}")
        return None


filaments_from_segmentation_lazy_batch = create_lazy_batch_converter(
    converter_func=filaments_from_segmentation,
    task_description="Tracing filaments",
)
