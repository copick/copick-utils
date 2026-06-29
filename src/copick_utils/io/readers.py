import numpy as np
from copick.util.uri import resolve_copick_objects


def tomogram(run, voxel_size: float = 10, algorithm: str = "wbp", raise_error: bool = False, verbose=True):
    """Read a tomogram from a copick run as a NumPy array.

    Resolves the tomogram `{algorithm}@{voxel_size}` for the run. If it is not
    found, reports the available voxel spacings / algorithms (raising or printing
    depending on `raise_error` / `verbose`).

    Args:
        run: The copick run to read from.
        voxel_size: Voxel spacing to query, in angstroms.
        algorithm: Tomogram algorithm / type to read (e.g. `wbp`).
        raise_error: Raise `ValueError` if the tomogram is missing instead of
            returning `None`.
        verbose: Print a diagnostic message listing what is available when the
            tomogram is missing.

    Returns:
        The tomogram as a NumPy array of shape (Z, Y, X), or `None` if it is
        missing and `raise_error` is False.
    """

    # Get the tomogram from the Copick URI
    try:
        uri = f"{algorithm}@{voxel_size}"
        vol = resolve_copick_objects(uri, run.root, "tomogram", run_name=run.name)
        return vol[0].numpy()
    except Exception as err:  # Report which orbject is missing
        # Try to resolve the tomogram using the Copick URI
        voxel_spacing_obj = run.get_voxel_spacing(voxel_size)

        if voxel_spacing_obj is None:
            # Query Avaiable Voxel Spacings
            availableVoxelSpacings = [tomo.voxel_size for tomo in run.voxel_spacings]

            # Report to the user which voxel spacings they can use
            message = (
                f"[Warning] No tomogram found for {run.name} with uri: {uri}\n"
                f"Available voxel sizes are: {', '.join(map(str, availableVoxelSpacings))}"
            )
            if raise_error:
                raise ValueError(message) from err
            elif verbose:
                print(message)
            return None

        tomogram = voxel_spacing_obj.get_tomogram(algorithm)
        if tomogram is None:
            # Get available algorithms
            availableAlgorithms = [tomo.tomo_type for tomo in run.get_voxel_spacing(voxel_size).tomograms]

            # Report to the user which algorithms are available
            message = (
                f"[Warning] No tomogram found for {run.name} with uri: {uri}\n"
                f"Available algorithms @{voxel_size}A are: {', '.join(availableAlgorithms)}"
            )
            if raise_error:
                raise ValueError(message) from err
            elif verbose:
                print(message)
            return None


def segmentation(run, voxel_spacing: float, name: str, user_id=None, session_id=None, raise_error=False, verbose=True):
    """Read a segmentation from a copick run as a NumPy array.

    Resolves the segmentation matching `name` (optionally filtered by `user_id`
    and `session_id`) at the given voxel spacing. If none matches, reports the
    available segmentations.

    Args:
        run: The copick run to read from.
        voxel_spacing: Voxel spacing of the segmentation, in angstroms.
        name: Segmentation / pickable-object name.
        user_id: Restrict to this user id (optional).
        session_id: Restrict to this session id (optional).
        raise_error: Raise `ValueError` if no segmentation matches instead of
            returning `None`.
        verbose: Print a diagnostic message listing what is available when no
            segmentation matches.

    Returns:
        The segmentation as a NumPy array of shape (Z, Y, X), or `None` if none
        matches and `raise_error` is False.
    """

    # Construct the Target URI
    if session_id is None and user_id is None:
        uri = f"{name}@{voxel_spacing}"
    elif session_id is None:
        uri = f"{name}:{user_id}@{voxel_spacing}"
    else:
        uri = f"{name}:{user_id}/{session_id}@{voxel_spacing}"

    # Try to resolve the segmentation using the Copick URI
    try:
        segs = resolve_copick_objects(uri, run.root, "segmentation", run_name=run.name)
        return segs[0].numpy()
    except Exception as err:
        # Force the voxel spacing to be a float
        voxel_spacing = float(voxel_spacing)

        # Get all available segmentations with their metadata
        available_segs = run.get_segmentations(voxel_size=voxel_spacing)

        if len(available_segs) == 0:
            available_segs = run.get_segmentations()
            message = (
                f"No segmentation found for URI: {uri}\n"
                f"Available segmentations avaiable w/following voxel sizes: {', '.join(map(str, [s.voxel_size for s in available_segs]))}"
            )
        else:
            seg_info = [(s.name, s.user_id, s.session_id) for s in available_segs]

            # Format the information for display
            seg_details = [f"(name: {name}, user_id: {uid}, session_id: {sid})" for name, uid, sid in seg_info]

            message = (
                f"\nNo segmentation at {voxel_spacing} A found matching:\n"
                f"  name: {name}, user_id: {user_id}, session_id: {session_id}\n"
                f"Available segmentations in {run.name} are:\n  " + "\n  ".join(seg_details)
            )
        if raise_error:
            raise ValueError(message) from err
        elif verbose:
            print(message)
        else:
            return None


def coordinates(
    run,  # CoPick run object containing the segmentation data
    name: str,  # Name of the object or protein for which coordinates are being extracted
    user_id: str,  # Identifier of the user that generated the picks
    session_id: str = None,  # Identifier of the session that generated the picks
    voxel_size: float = 10,  # Voxel size of the tomogram, used for scaling the coordinates
    raise_error: bool = False,
    verbose: bool = True,
):
    """Read pick coordinates from a copick run as an array of voxel indices.

    Looks up the picks for `name` (optionally filtered by `user_id` and
    `session_id`) and returns their locations divided by `voxel_size`. If several
    pick sets match, the first is used; if none match, reports the available picks.

    Args:
        run: The copick run to read from.
        name: Pickable-object / protein name.
        user_id: Identifier of the user that generated the picks.
        session_id: Identifier of the session that generated the picks (optional).
        voxel_size: Voxel spacing used to scale physical coordinates to voxels.
        raise_error: Raise `ValueError` if no picks match instead of returning `None`.
        verbose: Print a diagnostic message when no picks (or several) match.

    Returns:
        An (N, 3) array of (z, y, x) coordinates in voxel space, or `None` if no
        picks match and `raise_error` is False.
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
        elif verbose:
            print(message)
        return None

    elif len(picks) > 1:
        # Format pick information for display
        picks_info = [(p.pickable_object_name, p.user_id, p.session_id) for p in picks]
        picks_details = [f"(name: {name}, user_id: {uid}, session_id: {sid})" for name, uid, sid in picks_info]

        if verbose:
            print(
                "[Warning] More than 1 pick is available for the query information."
                "\nAvailable picks are:\n  " + "\n  ".join(picks_details) + f"\n"
                f"Defaulting to loading:\n {picks[0]}\n",
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
