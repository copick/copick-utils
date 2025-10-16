"""Input/output selection logic for conversion CLI commands."""

import re


def _is_regex_pattern(pattern: str) -> bool:
    """Check if string is a regex pattern by trying to compile it and seeing if it has special chars."""
    # Check for common regex special characters
    regex_chars = r"[.*+?^${}()|[\]\\"
    has_regex_chars = any(char in pattern for char in regex_chars)

    if not has_regex_chars:
        return False

    # Try to compile as regex
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


def validate_placeholders(
    pick_session_id: str,
    mesh_session_id: str,
    individual_meshes: bool,
) -> None:
    """
    Validate that session ID templates contain required placeholders.

    Args:
        pick_session_id: Input session ID or pattern
        mesh_session_id: Output session ID template
        individual_meshes: Whether individual meshes are being created

    Raises:
        ValueError: If template validation fails
    """
    if individual_meshes and "{instance_id}" not in mesh_session_id:
        raise ValueError(
            "Session ID template must contain {instance_id} placeholder when individual-meshes is enabled",
        )

    # Check if this is many-to-many mode (input has regex pattern)
    if _is_regex_pattern(pick_session_id) and "{input_session_id}" not in mesh_session_id:
        raise ValueError(
            "Session ID template must contain {input_session_id} placeholder when using regex input pattern",
        )


def validate_conversion_placeholders(
    input_session_id: str,
    output_session_id: str,
    individual_outputs: bool,
) -> None:
    """
    Validate that session ID templates contain required placeholders.

    Args:
        input_session_id: Input session ID or pattern
        output_session_id: Output session ID template
        individual_outputs: Whether individual outputs are being created

    Raises:
        ValueError: If template validation fails
    """
    if individual_outputs and "{instance_id}" not in output_session_id:
        raise ValueError(
            "Session ID template must contain {instance_id} placeholder when individual outputs are enabled",
        )

    # Check if this is many-to-many mode (input has regex pattern)
    if _is_regex_pattern(input_session_id) and "{input_session_id}" not in output_session_id:
        raise ValueError(
            "Session ID template must contain {input_session_id} placeholder when using regex input pattern",
        )
