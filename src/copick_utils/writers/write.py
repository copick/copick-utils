from typing import Any, Dict, List
import numpy as np

def tomogram(
    run,
    inputVol,
    voxelSize=10,
    tomo_algorithm="wbp"
    ):
    """
    Write a OME-Zarr tomogram into a Copick Directory.

    Parameters:
    - run: The run object, which provides a method to create a new tomogram.
    - inputVol: The volumetric tomgoram data to be written.
    - voxelsize (float): The size of the voxels. Default is 10.
    """

    # Create a new segmentation or Read Previous Segmentation
    vs = run.get_voxel_spacing(voxelSize)
    if vs is None:
        vs = run.new_voxel_spacing(
            voxel_size=voxelSize,
        )
        tomogram = vs.new_tomogram(tomo_algorithm)
    else:
        tomogram = vs.get_tomogram(tomo_algorithm)
        
    tomogram.from_numpy(inputVol, voxelSize)

def segmentation(
    run,
    inputSegmentVol,
    userID,
    segmentationName="segmentation",
    sessionID="0",
    voxelSize=10,
    multilabel_seg = True
    ):
    """
    Write a OME-Zarr segmentation into a Copick Directory.

    Parameters:
    - run: The run object, which provides a method to create a new segmentation.
    - segmentation: The segmentation data to be written.
    - voxelsize (float): The size of the voxels. Default is 10.
    """

    # Create a new segmentation or Read Previous Segmentation
    seg = run.get_segmentations(name=segmentationName, user_id=userID, session_id=sessionID)

    if len(seg) == 0 or seg[0].voxel_size != voxelSize:
        seg = run.new_segmentation(
            voxel_size=voxelSize,
            name=segmentationName,
            session_id=sessionID,
            is_multilabel=multilabel_seg,
            user_id=userID,
        )
    else:
        seg = seg[0]

    seg.from_numpy(inputSegmentVol, dtype=np.uint8)

