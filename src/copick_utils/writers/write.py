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
        vs = run.new_voxel_spacing(voxel_size=voxelSize)
        tomogram = vs.new_tomogram(tomo_algorithm)
    else:
        tomogram = vs.get_tomogram(tomo_algorithm)
        
    # Write the Tomogram
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
    - userID: The User ID with the Associated 
    - voxelsize (float): The size of the voxels. Default is 10.
    """

    # Create a new segmentation or Read Previous Segmentation
    seg = run.get_segmentations(name=segmentationName, user_id=userID, session_id=sessionID)

    # Write New Segmentation if Neither Any Segmentations Exist, 
    # Or Any for the Given Voxel Size
    if len(seg) == 0 or any(s.voxel_size != voxelSize for s in seg):
        seg = run.new_segmentation(
            voxel_size=voxelSize,
            name=segmentationName,
            session_id=sessionID,
            is_multilabel=multilabel_seg,
            user_id=userID,
        )
    else:
        # Overwrite Current Segmentation at that Resolution 
        # if it Exists
        seg = next(s for s in seg if s.voxel_size == voxelSize)

    # Write the Segmentation
    seg.from_numpy(inputSegmentVol, dtype=np.uint8)

