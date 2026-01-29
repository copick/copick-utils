from copick.util.uri import resolve_copick_objects
import numpy as np


def tomogram(run, voxel_size: float = 10, algorithm: str = "wbp", raise_error: bool = False):
    """
    Reads a tomogram from a Copick run.

    Parameters:
    -----------
    run: copick.Run
    voxel_size: float
    algorithm: str
    raise_error: bool

    Returns:
    --------
    vol: np.ndarray - The tomogram.
    """

    # Get the tomogram from the Copick URI
    try:
        uri = f'{algorithm}@{voxel_size}'
        vol = resolve_copick_objects(uri, run.root, 'tomogram', run_name = run.name)
        return vol[0].numpy()
    except: # Report which orbject is missing

        # Try to resolve the tomogram using the Copick URI
        voxel_spacing_obj = run.get_voxel_spacing(voxel_size)

        if voxel_spacing_obj is None:
            # Query Avaiable Voxel Spacings
            availableVoxelSpacings = [tomo.voxel_size for tomo in run.voxel_spacings]

            # Report to the user which voxel spacings they can use
            message = (
                f"[Warning] No tomogram found for {run.name} with voxel size {voxel_size} and tomogram type {algorithm}"
                f"Available spacings are: {', '.join(map(str, availableVoxelSpacings))}"
            )
            if raise_error:
                raise ValueError(message)
            else:
                print(message)
                return None

        tomogram = voxel_spacing_obj.get_tomogram(algorithm)
        if tomogram is None:
            # Get available algorithms
            availableAlgorithms = [tomo.tomo_type for tomo in run.get_voxel_spacing(voxel_size).tomograms]

            # Report to the user which algorithms are available
            message = (
                f"[Warning] No tomogram found for {run.name} with voxel size {voxel_size} and tomogram type {algorithm}"
                f"Available algorithms are: {', '.join(availableAlgorithms)}"
            )
            if raise_error:
                raise ValueError(message)
            else:
                print(message)
                return None


def segmentation(run, voxel_spacing: float, name: str, user_id=None,  session_id=None, raise_error=False):
    """
    Reads a segmentation from a Copick run.

    Parameters:
    -----------
    run: copick.Run
    voxel_spacing: float
    name: str
    user_id: str
    session_id: str
    raise_error: bool

    Returns:
    --------
    seg: np.ndarray - The segmentation.
    """

    # Fill in the missing values with wildcards
    if user_id is None: user_id = '*'
    if session_id is None: session_id = '*'

    # Try to resolve the segmentation using the Copick URI
    try:
        uri = f'{name}:{user_id}/{session_id}@{voxel_spacing}'
        segs = resolve_copick_objects(uri, run.root, 'segmentation', run_name = run.name)
        return segs[0].numpy()
    except:
        # If the query was unavailable, set the user_id and session_id to None
        user_id, session_id = None, None

        # Query Was Unavailable, Let's List Out All Available Segmentations
        seg = run.get_segmentations(
            name=name,
            session_id=session_id,
            user_id=user_id,
            voxel_size=voxel_spacing,
        )

        # No Segmentations Are Available, Result in Error
        if len(seg) == 0:
            # Get all available segmentations with their metadata
            available_segs = run.get_segmentations(voxel_size=voxel_spacing)
            seg_info = [(s.name, s.user_id, s.session_id) for s in available_segs]

            # Format the information for display
            seg_details = [f"(name: {name}, user_id: {uid}, session_id: {sid})" for name, uid, sid in seg_info]

            message = (
                f"\nNo segmentation found matching:\n"
                f"  name: {name}, user_id: {user_id}, session_id: {session_id}\n"
                f"Available segmentations in {run.name} are:\n  " + "\n  ".join(seg_details)
            )
            if raise_error:
                raise ValueError(message)
            else:
                print(message)
                return None

        # No Segmentations Are Available, Result in Error
        if len(seg) > 1:
            print(
                f"[Warning] More Than 1 Segmentation is Available for the Query Information. "
                f"Available Segmentations are: {seg} "
                f"Defaulting to Loading: {seg[0]}\n",
            )


def coordinates(
    run,  # CoPick run object containing the segmentation data
    name: str,  # Name of the object or protein for which coordinates are being extracted
    user_id: str,  # Identifier of the user that generated the picks
    session_id: str = None,  # Identifier of the session that generated the picks
    voxel_size: float = 10,  # Voxel size of the tomogram, used for scaling the coordinates
    raise_error: bool = False,
):
    """
    Reads the coordinates of the picks from a Copick run.

    Parameters:
    -----------
    run: copick.Run
    name: str
    user_id: str
    session_id: str
    voxel_size: float
    raise_error: bool

    Returns:
    --------
    coordinates: np.ndarray - The 3D coordinates of the picks in voxel space.
    """
    # Retrieve the pick points associated with the specified object and user ID
    picks = run.get_picks(object_name=name, user_id=user_id, session_id=session_id)

    if len(picks) == 0:
        # Get all available segmentations with their metadata

        available_picks = run.get_picks()
        picks_info = [(s.pickable_object_name, s.user_id, s.session_id) for s in available_picks]

        # Format the information for display
        picks_details = [f"(name: {name}, user_id: {uid}, session_id: {sid})" for name, uid, sid in picks_info]

        message = (
            f"\nNo picks found matching:\n"
            f"  name: {name}, user_id: {user_id}, session_id: {session_id}\n"
            f"Available picks are:\n  " + "\n  ".join(picks_details)
        )
        if raise_error:
            raise ValueError(message)
        else:
            print(message)
            return None
    elif len(picks) > 1:
        # Format pick information for display
        picks_info = [(p.pickable_object_name, p.user_id, p.session_id) for p in picks]
        picks_details = [f"(name: {name}, user_id: {uid}, session_id: {sid})" for name, uid, sid in picks_info]

        print(
            "[Warning] More than 1 pick is available for the query information."
            "\nAvailable picks are:\n  " + "\n  ".join(picks_details) + f"\nDefaulting to loading:\n {picks[0]}\n",
        )
    points = picks[0].points

    # Initialize an array to store the coordinates
    nPoints = len(picks[0].points)  # Number of points retrieved
    coordinates = np.zeros([len(picks[0].points), 3])  # Create an empty array to hold the (z, y, x) coordinates

    # Iterate over all points and convert their locations to coordinates in voxel space
    for ii in range(nPoints):
        coordinates[ii,] = [
            points[ii].location.z / voxel_size,  # Scale z-coordinate by voxel size
            points[ii].location.y / voxel_size,  # Scale y-coordinate by voxel size
            points[ii].location.x / voxel_size,
        ]  # Scale x-coordinate by voxel size

    # Return the array of coordinates
    return coordinates
