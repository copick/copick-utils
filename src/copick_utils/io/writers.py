import numpy as np


def tomogram(run, input_volume, voxel_size=10, algorithm="wbp"):
    """Write a volumetric tomogram into a copick run as OME-Zarr.

    Reuses the voxel spacing and tomogram for `algorithm` if they exist, creating
    them otherwise, then writes `input_volume` (building the multiscale pyramid).

    Args:
        run: The copick run to write into.
        input_volume: The tomogram volume to write, shape (Z, Y, X).
        voxel_size: Voxel spacing in angstroms.
        algorithm: Tomogram algorithm / type to write under (e.g. `wbp`).
    """

    # Retrieve or create voxel spacing
    voxel_spacing = run.get_voxel_spacing(voxel_size)
    if voxel_spacing is None:
        voxel_spacing = run.new_voxel_spacing(voxel_size=voxel_size)

    # Check if We Need to Create a New Tomogram for Given Algorithm
    tomo = voxel_spacing.get_tomogram(algorithm)
    if tomo is None:
        tomo = voxel_spacing.new_tomogram(tomo_type=algorithm)

    # Write the tomogram data
    tomo.from_numpy(input_volume)


def segmentation(
    run,
    seg_vol,
    user_id,
    name="segmentation",
    session_id="0",
    voxel_size=10,
    multilabel=True,
):
    """Write a segmentation into a copick run as OME-Zarr.

    Reuses an existing segmentation matching `name` / `user_id` / `session_id` at
    the given voxel spacing if present, creating a new one otherwise, then writes
    `seg_vol` as `uint8`.

    Args:
        run: The copick run to write into.
        seg_vol: The label volume to write, shape (Z, Y, X).
        user_id: User id to attribute the segmentation to.
        name: Segmentation name. Defaults to `segmentation`.
        session_id: Session id for the segmentation. Defaults to `0`.
        voxel_size: Voxel spacing in angstroms.
        multilabel: Whether the segmentation holds multiple labels.
    """

    # Retrieve or create a segmentation
    segmentations = run.get_segmentations(name=name, user_id=user_id, session_id=session_id)

    # If no segmentation exists or no segmentation at the given voxel size, create a new one
    if len(segmentations) == 0 or any(seg.voxel_size != voxel_size for seg in segmentations):
        seg = run.new_segmentation(
            voxel_size=voxel_size,
            name=name,
            session_id=session_id,
            is_multilabel=multilabel,
            user_id=user_id,
        )
    else:
        # Overwrite the current segmentation at the specified voxel size if it exists
        seg = next(seg for seg in segmentations if seg.voxel_size == voxel_size)

    # Write the segmentation data
    seg.from_numpy(seg_vol, dtype=np.uint8)
