"""Utility functions for pattern matching in copick objects."""

import re
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from copick.models import CopickMesh, CopickPicks, CopickRun, CopickSegmentation


def find_matching_segmentations(
    run: "CopickRun",
    segmentation_name: str,
    segmentation_user_id: str,
    session_id_pattern: str,
) -> List["CopickSegmentation"]:
    """
    Find segmentations matching a session ID pattern.

    Parameters:
    -----------
    run : CopickRun
        Copick run to search in
    segmentation_name : str
        Name of the segmentations
    segmentation_user_id : str
        User ID of the segmentations
    session_id_pattern : str
        Regex pattern or exact session ID to match

    Returns:
    --------
    matching_segmentations : list
        List of segmentations matching the pattern
    """
    # Get all segmentations for this name and user
    all_segmentations = run.get_segmentations(name=segmentation_name, user_id=segmentation_user_id)

    if not all_segmentations:
        return []

    # Check if pattern is a regex or exact match
    try:
        # Try to compile as regex
        pattern = re.compile(session_id_pattern)
        is_regex = True
    except re.error:
        # Not a valid regex, treat as exact string
        is_regex = False

    matching_segmentations = []

    for seg in all_segmentations:
        if is_regex:
            if pattern.match(seg.session_id):
                matching_segmentations.append(seg)
        else:
            if seg.session_id == session_id_pattern:
                matching_segmentations.append(seg)

    return matching_segmentations


def find_matching_picks(
    run: "CopickRun",
    object_name: str,
    pick_user_id: str,
    session_id_pattern: str,
) -> List["CopickPicks"]:
    """
    Find picks matching a session ID pattern.

    Parameters:
    -----------
    run : CopickRun
        Copick run to search in
    object_name : str
        Name of the object/picks
    pick_user_id : str
        User ID of the picks
    session_id_pattern : str
        Regex pattern or exact session ID to match

    Returns:
    --------
    matching_picks : list
        List of picks matching the pattern
    """
    # Get all picks for this object and user
    all_picks = run.get_picks(object_name=object_name, user_id=pick_user_id)

    if not all_picks:
        return []

    # Check if pattern is a regex or exact match
    try:
        # Try to compile as regex
        pattern = re.compile(session_id_pattern)
        is_regex = True
    except re.error:
        # Not a valid regex, treat as exact string
        is_regex = False

    matching_picks = []

    for picks in all_picks:
        if is_regex:
            if pattern.match(picks.session_id):
                matching_picks.append(picks)
        else:
            if picks.session_id == session_id_pattern:
                matching_picks.append(picks)

    return matching_picks


def find_matching_meshes(
    run: "CopickRun",
    object_name: str,
    mesh_user_id: str,
    session_id_pattern: str,
) -> List["CopickMesh"]:
    """
    Find meshes matching a session ID pattern.

    Parameters:
    -----------
    run : CopickRun
        Copick run to search in
    object_name : str
        Name of the object/meshes
    mesh_user_id : str
        User ID of the meshes
    session_id_pattern : str
        Regex pattern or exact session ID to match

    Returns:
    --------
    matching_meshes : list
        List of meshes matching the pattern
    """
    # Get all meshes for this object and user
    all_meshes = run.get_meshes(object_name=object_name, user_id=mesh_user_id)

    if not all_meshes:
        return []

    # Check if pattern is a regex or exact match
    try:
        # Try to compile as regex
        pattern = re.compile(session_id_pattern)
        is_regex = True
    except re.error:
        # Not a valid regex, treat as exact string
        is_regex = False

    matching_meshes = []

    for mesh in all_meshes:
        if is_regex:
            if pattern.match(mesh.session_id):
                matching_meshes.append(mesh)
        else:
            if mesh.session_id == session_id_pattern:
                matching_meshes.append(mesh)

    return matching_meshes
