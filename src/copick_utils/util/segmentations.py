"""Segmentation types: how a URI's type flags select and create binary, multilabel, instance and panoptic stores.

copick has four segmentation types. A URI names one with a query flag (``?multilabel=true``, ``?instance=true``,
``?panoptic=true``). copick-utils treats a URI without a flag as binary or multilabel, so a tool that does not ask for
instance or panoptic segmentations never reads one by accident (an instance store can share its name, user and session
with a binary one, and a panoptic store is 4D).
"""

from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

if TYPE_CHECKING:
    from copick.models import CopickRun, CopickSegmentation

SegmentationType = Literal["binary", "multilabel", "instance", "panoptic"]


def selection_filters(
    multilabel: Optional[bool] = None,
    instance: Optional[bool] = None,
    panoptic: Optional[bool] = None,
) -> Dict[str, Optional[bool]]:
    """The type filters for copick's segmentation queries (``get_copick_objects_by_type``), from a URI's flags.

    Args:
        multilabel: The URI's multilabel flag, or None.
        instance: The URI's instance flag, or None.
        panoptic: The URI's panoptic flag, or None.

    Returns:
        ``{"multilabel": ..., "instance": ..., "panoptic": ...}``. Without an instance or panoptic flag, both are False.
    """
    return {"multilabel": multilabel, "instance": bool(instance), "panoptic": bool(panoptic)}


def query_kwargs(
    multilabel: Optional[bool] = None,
    instance: Optional[bool] = None,
    panoptic: Optional[bool] = None,
) -> Dict[str, Optional[bool]]:
    """The type keywords for ``CopickRun.get_segmentations``, from a URI's flags (see ``selection_filters``)."""
    filters = selection_filters(multilabel, instance, panoptic)
    return {
        "is_multilabel": filters["multilabel"],
        "is_instance": filters["instance"],
        "is_panoptic": filters["panoptic"],
    }


def type_from_flags(
    multilabel: Optional[bool] = None,
    instance: Optional[bool] = None,
    panoptic: Optional[bool] = None,
) -> Optional[SegmentationType]:
    """The segmentation type a URI's flags name, or None when it names none.

    Raises:
        ValueError: If more than one flag is set.
    """
    named = [t for t, flag in (("multilabel", multilabel), ("instance", instance), ("panoptic", panoptic)) if flag]
    if len(named) > 1:
        raise ValueError(f"A segmentation has one type; the URI names {', '.join(named)}.")
    return named[0] if named else None


def type_from_uri(uri: str) -> Optional[SegmentationType]:
    """The segmentation type a segmentation URI names with its query flags, or None."""
    from copick.util.uri import parse_copick_uri

    params = parse_copick_uri(uri, "segmentation")
    return type_from_flags(params.get("multilabel"), params.get("instance"), params.get("panoptic"))


def resolve_segmentations(uri: str, root, run_name: Optional[str] = None) -> list:
    """The segmentations a URI selects, of the type its flags name (binary or multilabel when it names none).

    Args:
        uri: Segmentation URI (patterns allowed).
        root: Copick root.
        run_name: Restrict to one run.

    Returns:
        List of segmentations.
    """
    from copick.util.uri import parse_copick_uri, resolve_copick_objects

    params = parse_copick_uri(uri, "segmentation")
    wanted = selection_filters(params.get("multilabel"), params.get("instance"), params.get("panoptic"))
    found = resolve_copick_objects(uri, root, "segmentation", run_name=run_name)
    return [
        s
        for s in found
        if (wanted["multilabel"] is None or bool(s.is_multilabel) == wanted["multilabel"])
        and bool(getattr(s, "is_instance", False)) == wanted["instance"]
        and bool(getattr(s, "is_panoptic", False)) == wanted["panoptic"]
    ]


def segmentation_type(segmentation: "CopickSegmentation") -> SegmentationType:
    """The type of an existing segmentation."""
    if getattr(segmentation, "is_panoptic", False):
        return "panoptic"
    if getattr(segmentation, "is_instance", False):
        return "instance"
    return "multilabel" if segmentation.is_multilabel else "binary"


def new_segmentation(
    run: "CopickRun",
    voxel_size: float,
    name: str,
    session_id: str,
    user_id: str,
    seg_type: SegmentationType,
    exist_ok: bool = True,
    **kwargs: Any,
) -> "CopickSegmentation":
    """Create (or reuse) a segmentation of the given type.

    Args:
        run: Run to write into.
        voxel_size: Voxel size in Angstrom.
        name: Object name (binary, instance) or descriptive name (multilabel, panoptic).
        session_id: Session ID.
        user_id: User ID.
        seg_type: ``"binary"``, ``"multilabel"``, ``"instance"`` or ``"panoptic"``.
        exist_ok: Reuse an existing segmentation with the same key.
        **kwargs: Passed to ``CopickRun.new_segmentation``.

    Returns:
        The segmentation.
    """
    if seg_type not in ("binary", "multilabel", "instance", "panoptic"):
        raise ValueError(f"Unknown segmentation type {seg_type!r}.")
    type_kwargs = {}
    if seg_type == "instance":
        type_kwargs["is_instance"] = True
    elif seg_type == "panoptic":
        type_kwargs["is_panoptic"] = True
    return run.new_segmentation(
        voxel_size=voxel_size,
        name=name,
        session_id=session_id,
        is_multilabel=seg_type == "multilabel",
        user_id=user_id,
        exist_ok=exist_ok,
        **type_kwargs,
        **kwargs,
    )
