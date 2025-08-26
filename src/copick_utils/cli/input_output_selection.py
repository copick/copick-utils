"""Input/output selection logic for picks-to-mesh CLI commands."""

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from copick_utils.util.pattern_matching import find_matching_picks

if TYPE_CHECKING:
    from copick.models import CopickRun


class InputOutputSelector:
    """Handle flexible input/output selection for picks-to-mesh conversion."""

    def __init__(
        self,
        pick_object_name: str,
        pick_user_id: str,
        pick_session_id: str,
        mesh_object_name: Optional[str] = None,
        mesh_user_id: str = "picks2mesh",
        mesh_session_id: str = "0",
        individual_meshes: bool = False,
    ):
        """
        Initialize the selector.

        Args:
            pick_object_name: Name of the pick object to convert
            pick_user_id: User ID of the picks to convert
            pick_session_id: Session ID or regex pattern of the picks
            mesh_object_name: Name of the mesh object to create (defaults to pick_object_name)
            mesh_user_id: User ID for created mesh
            mesh_session_id: Session ID or template for created mesh
            individual_meshes: Whether to create individual mesh files
        """
        self.pick_object_name = pick_object_name
        self.pick_user_id = pick_user_id
        self.pick_session_id = pick_session_id
        self.mesh_object_name = mesh_object_name or pick_object_name
        self.mesh_user_id = mesh_user_id
        self.mesh_session_id = mesh_session_id
        self.individual_meshes = individual_meshes

        # Validate session ID template
        self._validate_session_id_template()

    def _validate_session_id_template(self) -> None:
        """Validate that session ID template contains required placeholders."""
        if self.individual_meshes and "{instance_id}" not in self.mesh_session_id:
            raise ValueError(
                "Session ID template must contain {instance_id} placeholder when individual-meshes is enabled",
            )

        # Check if this is many-to-many mode (input has regex pattern)
        if self._is_regex_pattern(self.pick_session_id) and "{input_session_id}" not in self.mesh_session_id:
            raise ValueError(
                "Session ID template must contain {input_session_id} placeholder when using regex input pattern",
            )

    def _is_regex_pattern(self, pattern: str) -> bool:
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

    def get_conversion_tasks(self, run: "CopickRun") -> List[Dict[str, Any]]:
        """
        Get list of conversion tasks based on input/output selection.

        Args:
            run: CopickRun object to search for picks

        Returns:
            List of conversion task dictionaries with keys:
            - input_picks: CopickPicks object
            - mesh_object_name: str
            - mesh_user_id: str
            - mesh_session_id: str (resolved from template)
            - individual_meshes: bool
            - session_id_template: str (for individual meshes)
        """
        # Find matching input picks
        matching_picks = find_matching_picks(
            run=run,
            object_name=self.pick_object_name,
            pick_user_id=self.pick_user_id,
            session_id_pattern=self.pick_session_id,
        )

        if not matching_picks:
            return []

        tasks = []

        for picks in matching_picks:
            # Resolve mesh session ID from template
            resolved_session_id = self._resolve_session_id(picks.session_id)

            # Create session ID template for individual meshes
            session_id_template = None
            if self.individual_meshes:
                session_id_template = resolved_session_id

            task = {
                "input_picks": picks,
                "mesh_object_name": self.mesh_object_name,
                "mesh_user_id": self.mesh_user_id,
                "mesh_session_id": resolved_session_id,
                "individual_meshes": self.individual_meshes,
                "session_id_template": session_id_template,
            }
            tasks.append(task)

        return tasks

    def _resolve_session_id(self, input_session_id: str) -> str:
        """
        Resolve mesh session ID from template using input session ID.

        Args:
            input_session_id: Session ID of the input picks

        Returns:
            Resolved session ID for the output mesh
        """
        resolved = self.mesh_session_id

        # Replace input session ID placeholder
        resolved = resolved.replace("{input_session_id}", input_session_id)

        # Note: {instance_id} placeholder is handled by the individual mesh creation logic
        # in the converter functions, not here

        return resolved

    def get_mode_description(self) -> str:
        """Get description of the current selection mode."""
        is_regex = self._is_regex_pattern(self.pick_session_id)

        if is_regex and self.individual_meshes:
            return "many-to-many (regex input → template output with individual meshes)"
        elif is_regex:
            return "many-to-one (regex input → template output)"
        elif self.individual_meshes:
            return "one-to-many (single input → template output with individual meshes)"
        else:
            return "one-to-one (single input → single output)"


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
    InputOutputSelector(
        pick_object_name="dummy",
        pick_user_id="dummy",
        pick_session_id=pick_session_id,
        mesh_session_id=mesh_session_id,
        individual_meshes=individual_meshes,
    )
    # Validation happens in __init__, so if we get here it's valid
