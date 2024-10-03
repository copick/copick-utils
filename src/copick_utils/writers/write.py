from typing import Any, Dict, List
import ome_zarr.writer
import zarr

def tomogram(
    run,
    inputVol,
    voxelSize=10,
    tomo_algorithm="wbp"
    ):
    """
    Write a OME-Zarr segmentation into a Copick Directory.

    Parameters:
    - run: The run object, which provides a method to create a new segmentation.
    - segmentation: The segmentation data to be written.
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
        
    # Write the zarr file
    loc = tomogram.zarr()
    root_group = zarr.group(loc, overwrite=True)

    ome_zarr.writer.write_multiscale(
        [inputVol],
        group=root_group,
        axes=ome_zarr_axes(),
        coordinate_transformations=[ome_zarr_transforms(voxelSize)],
        storage_options=dict(chunks=(256, 256, 256), overwrite=True),
        compute=True,
    )

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


    # Write the zarr file
    loc = seg.zarr()
    root_group = zarr.group(loc, overwrite=True)

    ome_zarr.writer.write_multiscale(
        [inputSegmentVol],
        group=root_group,
        axes=ome_zarr_axes(),
        coordinate_transformations=[ome_zarr_transforms(voxelSize)],
        storage_options=dict(chunks=(256, 256, 256), overwrite=True),
        compute=True,
    )


def ome_zarr_feature_axes() -> List[Dict[str, str]]:
    """
    Returns a list of dictionaries defining the axes information for an OME-Zarr dataset.

    Returns:
    - List[Dict[str, str]]: A list of dictionaries, each specifying the name, type, and unit of an axis.
      The axes are 'z', 'y', and 'x', all of type 'space' and unit 'angstrom'.
    """
    return [
        {
            "name": "c",
            "type": "channel",
        },
        {
            "name": "z",
            "type": "space",
            "unit": "angstrom",
        },
        {
            "name": "y",
            "type": "space",
            "unit": "angstrom",
        },
        {
            "name": "x",
            "type": "space",
            "unit": "angstrom",
        },
    ]


def ome_zarr_feature_transforms(voxel_size: float) -> List[Dict[str, Any]]:
    """
    Return a list of dictionaries defining the coordinate transformations of OME-Zarr dataset.

    Parameters:
    - voxel_size (float): The size of a voxel.

    Returns:
    - List[Dict[str, Any]]: A list containing a single dictionary with the 'scale' transformation,
      specifying the voxel size for each axis and the transformation type as 'scale'.
    """
    return [{"scale": [voxel_size, voxel_size, voxel_size, voxel_size], "type": "scale"}]

def ome_zarr_axes() -> List[Dict[str, str]]:
    """
    Returns a list of dictionaries defining the axes information for an OME-Zarr dataset.

    Returns:
    - List[Dict[str, str]]: A list of dictionaries, each specifying the name, type, and unit of an axis.
      The axes are 'z', 'y', and 'x', all of type 'space' and unit 'angstrom'.
    """
    return [
        {
            "name": "z",
            "type": "space",
            "unit": "angstrom",
        },
        {
            "name": "y",
            "type": "space",
            "unit": "angstrom",
        },
        {
            "name": "x",
            "type": "space",
            "unit": "angstrom",
        },
    ]


def ome_zarr_transforms(voxel_size: float) -> List[Dict[str, Any]]:
    """
    Return a list of dictionaries defining the coordinate transformations of OME-Zarr dataset.

    Parameters:
    - voxel_size (float): The size of a voxel.

    Returns:
    - List[Dict[str, Any]]: A list containing a single dictionary with the 'scale' transformation,
      specifying the voxel size for each axis and the transformation type as 'scale'.
    """
    return [{"scale": [voxel_size, voxel_size, voxel_size], "type": "scale"}]

