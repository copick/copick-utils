"""Pydantic models for lazy task discovery configuration."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SelectorConfig(BaseModel):
    """Pydantic model for selector configuration with validation."""

    input_type: Literal["picks", "mesh", "segmentation"]
    output_type: Literal["picks", "mesh", "segmentation"]
    input_object_name: str
    input_user_id: str
    input_session_id: str
    output_object_name: str
    output_user_id: str = "converter"
    output_session_id: str = "0"
    individual_outputs: bool = False
    segmentation_name: Optional[str] = None
    voxel_spacing: Optional[float] = None

    @field_validator("segmentation_name")
    @classmethod
    def validate_segmentation_name(cls, v, info):
        """Ensure segmentation_name is provided when needed."""
        values = info.data
        input_type = values.get("input_type")
        output_type = values.get("output_type")

        if (input_type == "segmentation" or output_type == "segmentation") and v is None:
            raise ValueError("segmentation_name is required when input_type or output_type is 'segmentation'")
        return v

    @field_validator("voxel_spacing")
    @classmethod
    def validate_voxel_spacing(cls, v, info):
        """Ensure voxel_spacing is provided when working with segmentations."""
        values = info.data
        input_type = values.get("input_type")
        output_type = values.get("output_type")

        if (input_type == "segmentation" or output_type == "segmentation") and v is None:
            raise ValueError("voxel_spacing is required when working with segmentations")
        return v

    @field_validator("output_session_id")
    @classmethod
    def validate_output_session_id(cls, v, info):
        """Validate session ID templates contain required placeholders."""
        import re

        values = info.data
        input_session_id = values.get("input_session_id", "")
        individual_outputs = values.get("individual_outputs", False)

        # Check if input is a regex pattern
        regex_chars = r"[.*+?^${}()|[\]\\"
        has_regex_chars = any(char in input_session_id for char in regex_chars)
        is_regex = False
        if has_regex_chars:
            try:
                re.compile(input_session_id)
                is_regex = True
            except re.error:
                pass

        # Validate placeholders
        if individual_outputs and "{instance_id}" not in v:
            raise ValueError("output_session_id must contain {instance_id} placeholder when individual_outputs=True")

        if is_regex and "{input_session_id}" not in v:
            raise ValueError(
                "output_session_id must contain {input_session_id} placeholder when using regex input pattern",
            )

        return v

    @classmethod
    def from_uris(
        cls,
        input_uri: str,
        input_type: Literal["picks", "mesh", "segmentation"],
        output_uri: str,
        output_type: Literal["picks", "mesh", "segmentation"],
        individual_outputs: bool = False,
        command_name: Optional[str] = None,
    ) -> "SelectorConfig":
        """Create SelectorConfig from input/output URIs.

        Args:
            input_uri: Input copick URI string.
            input_type: Type of input object ('picks', 'mesh', 'segmentation').
            output_uri: Output copick URI string (supports smart defaults).
            output_type: Type of output object ('picks', 'mesh', 'segmentation').
            individual_outputs: Whether to create individual outputs.
            command_name: Name of the command (used for smart defaults in output_uri).

        Returns:
            SelectorConfig instance with parsed URI components.

        Raises:
            ValueError: If URI parsing fails or required fields are missing.
        """
        from copick.util.uri import expand_output_uri, parse_copick_uri

        # Expand output URI with smart defaults
        output_uri = expand_output_uri(
            output_uri=output_uri,
            input_uri=input_uri,
            input_type=input_type,
            output_type=output_type,
            command_name=command_name,
            individual_outputs=individual_outputs,
        )

        # Parse input URI
        input_params = parse_copick_uri(input_uri, input_type)

        # Parse output URI (now fully expanded)
        output_params = parse_copick_uri(output_uri, output_type)

        # Extract common fields
        input_object_name = input_params.get("object_name") or input_params.get("name")
        output_object_name = output_params.get("object_name") or output_params.get("name")

        # Build config dict
        config_dict = {
            "input_type": input_type,
            "output_type": output_type,
            "input_object_name": input_object_name,
            "input_user_id": input_params["user_id"],
            "input_session_id": input_params["session_id"],
            "output_object_name": output_object_name,
            "output_user_id": output_params["user_id"],
            "output_session_id": output_params["session_id"],
            "individual_outputs": individual_outputs,
        }

        # Add segmentation-specific fields if needed
        if input_type == "segmentation" or output_type == "segmentation":
            seg_name = input_params.get("name") or output_params.get("name")
            voxel_spacing = input_params.get("voxel_spacing") or output_params.get("voxel_spacing")

            config_dict["segmentation_name"] = seg_name

            # Convert voxel_spacing to float if it's a string
            if isinstance(voxel_spacing, str) and voxel_spacing != "*":
                voxel_spacing = float(voxel_spacing)
            config_dict["voxel_spacing"] = voxel_spacing

        return cls(**config_dict)


class ReferenceConfig(BaseModel):
    """Pydantic model for reference discovery configuration."""

    reference_type: Literal["mesh", "segmentation"]
    object_name: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    voxel_spacing: Optional[float] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("voxel_spacing")
    @classmethod
    def validate_segmentation_voxel_spacing(cls, v, info):
        """Ensure voxel_spacing is provided for segmentation references."""
        values = info.data
        if values.get("reference_type") == "segmentation" and v is None:
            raise ValueError("voxel_spacing is required for segmentation references")
        return v

    @field_validator("object_name")
    @classmethod
    def validate_required_fields(cls, v, info):
        """Ensure required fields are provided."""
        if v is None:
            raise ValueError("object_name is required for reference configuration")
        return v

    @classmethod
    def from_uri(
        cls,
        uri: str,
        reference_type: Literal["mesh", "segmentation"],
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> "ReferenceConfig":
        """Create ReferenceConfig from a URI.

        Args:
            uri: Copick URI string for the reference object.
            reference_type: Type of reference ('mesh' or 'segmentation').
            additional_params: Additional parameters to include in the config.

        Returns:
            ReferenceConfig instance with parsed URI components.

        Raises:
            ValueError: If URI parsing fails or required fields are missing.
        """
        from copick.util.uri import parse_copick_uri

        # Parse URI
        params = parse_copick_uri(uri, reference_type)

        # Extract fields based on type
        object_name = params.get("object_name") or params.get("name")
        user_id = params["user_id"]
        session_id = params["session_id"]
        voxel_spacing = params.get("voxel_spacing")

        # Convert voxel_spacing to float if it's a string
        if voxel_spacing and isinstance(voxel_spacing, str) and voxel_spacing != "*":
            voxel_spacing = float(voxel_spacing)

        return cls(
            reference_type=reference_type,
            object_name=object_name,
            user_id=user_id,
            session_id=session_id,
            voxel_spacing=voxel_spacing,
            additional_params=additional_params or {},
        )


class TaskConfig(BaseModel):
    """Pydantic model for complete task configuration."""

    type: Literal[
        "single_selector",
        "dual_selector",
        "multi_selector",
        "single_selector_with_reference",
        "single_selector_multi_union",
    ]
    selector: Optional[SelectorConfig] = None
    selectors: Optional[List[SelectorConfig]] = None
    reference: Optional[ReferenceConfig] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)
    pairing_method: Optional[str] = "index_order"

    @field_validator("selector")
    @classmethod
    def validate_single_selector(cls, v, info):
        """Validate single selector configuration."""
        values = info.data
        config_type = values.get("type")
        if config_type == "single_selector" and v is None:
            raise ValueError("selector is required for single_selector type")
        elif config_type == "single_selector_with_reference" and v is None:
            raise ValueError("selector is required for single_selector_with_reference type")
        elif config_type == "single_selector_multi_union" and v is None:
            raise ValueError("selector is required for single_selector_multi_union type")
        return v

    @field_validator("selectors")
    @classmethod
    def validate_dual_selectors(cls, v, info):
        """Validate selector configuration."""
        values = info.data
        config_type = values.get("type")
        if config_type == "dual_selector" and (v is None or len(v) != 2):
            raise ValueError("exactly 2 selectors required for dual_selector type")
        if config_type == "multi_selector" and (v is None or len(v) < 2):
            raise ValueError("at least 2 selectors required for multi_selector type")
        return v

    @field_validator("reference")
    @classmethod
    def validate_reference(cls, v, info):
        """Validate reference configuration."""
        values = info.data
        config_type = values.get("type")
        if config_type == "single_selector_with_reference" and v is None:
            raise ValueError("reference is required for single_selector_with_reference type")
        return v


# URI-based convenience functions (simplified interface)
def create_simple_config(
    input_uri: str,
    input_type: Literal["picks", "mesh", "segmentation"],
    output_uri: str,
    output_type: Literal["picks", "mesh", "segmentation"],
    individual_outputs: bool = False,
    command_name: Optional[str] = None,
) -> TaskConfig:
    """Create a simple single-selector task configuration from URIs.

    Args:
        input_uri: Input copick URI string.
        input_type: Type of input object.
        output_uri: Output copick URI string (supports smart defaults).
        output_type: Type of output object.
        individual_outputs: Whether to create individual outputs.
        command_name: Name of the command (used for smart defaults in output_uri).

    Returns:
        TaskConfig instance ready for use.

    Example:
        config = create_simple_config(
            input_uri="ribosome:user1/manual-001",
            input_type="picks",
            output_uri="ribosome",
            output_type="mesh",
            command_name="picks2mesh",
        )
    """
    selector_config = SelectorConfig.from_uris(
        input_uri=input_uri,
        input_type=input_type,
        output_uri=output_uri,
        output_type=output_type,
        individual_outputs=individual_outputs,
        command_name=command_name,
    )

    return TaskConfig(type="single_selector", selector=selector_config)


def create_single_selector_config(
    input_uri: str,
    input_type: Literal["mesh", "segmentation"],
    output_uri: str,
    output_type: Literal["mesh", "segmentation"],
    command_name: Optional[str] = None,
    operation: str = "union",
) -> TaskConfig:
    """
    Create a single-selector config for pattern expansion to N-way operations.

    Used when a single input URI with pattern should expand to multiple objects
    within each run, then perform an N-way operation on all matched objects.

    Args:
        input_uri: Input copick URI string (may contain patterns)
        input_type: Type of input objects
        output_uri: Output copick URI string
        output_type: Type of output object
        command_name: Name of command for smart defaults
        operation: Operation type (currently only "union" supported)

    Returns:
        TaskConfig with single_selector_multi_union type

    Raises:
        ValueError: If operation is not "union"

    Example:
        config = create_single_selector_config(
            input_uri="membrane:user*/manual-*@10.0",
            input_type="segmentation",
            output_uri="merged",
            output_type="segmentation",
            command_name="segop",
            operation="union",
        )
    """
    if operation not in ["union", "concatenate"]:
        raise ValueError(
            f"Single-input pattern expansion only supports 'union' and 'concatenate' operations, got '{operation}'",
        )

    selector = SelectorConfig.from_uris(
        input_uri=input_uri,
        input_type=input_type,
        output_uri=output_uri,
        output_type=output_type,
        individual_outputs=False,
        command_name=command_name,
    )

    return TaskConfig(
        type="single_selector_multi_union",
        selector=selector,
    )


def create_dual_selector_config(
    input1_uri: str,
    input2_uri: str,
    input_type: Literal["mesh", "segmentation"],
    output_uri: str,
    output_type: Literal["mesh", "segmentation"],
    pairing_method: str = "index_order",
    individual_outputs: bool = False,
    command_name: Optional[str] = None,
) -> TaskConfig:
    """Create a dual-selector task configuration from URIs.

    Args:
        input1_uri: First input copick URI string.
        input2_uri: Second input copick URI string.
        input_type: Type of input objects (both inputs must be same type).
        output_uri: Output copick URI string (supports smart defaults).
        output_type: Type of output object.
        pairing_method: How to pair inputs ("index_order", etc.).
        individual_outputs: Whether to create individual outputs.
        command_name: Name of the command (used for smart defaults in output_uri).

    Returns:
        TaskConfig instance ready for use.

    Example:
        config = create_dual_selector_config(
            input1_uri="membrane:user1/manual-001",
            input2_uri="vesicle:user1/auto-001",
            input_type="mesh",
            output_uri="combined",
            output_type="mesh",
            command_name="meshop",
        )
    """
    from copick.util.uri import parse_copick_uri

    # Parse both inputs
    parse_copick_uri(input1_uri, input_type)
    input2_params = parse_copick_uri(input2_uri, input_type)
    parse_copick_uri(output_uri, output_type)

    # Create first selector from URIs
    selector1_config = SelectorConfig.from_uris(
        input_uri=input1_uri,
        input_type=input_type,
        output_uri=output_uri,
        output_type=output_type,
        individual_outputs=individual_outputs,
        command_name=command_name,
    )

    # Create second selector manually (output fields not used)
    input2_object_name = input2_params.get("object_name") or input2_params.get("name")

    selector2_dict = {
        "input_type": input_type,
        "output_type": output_type,
        "input_object_name": input2_object_name,
        "input_user_id": input2_params["user_id"],
        "input_session_id": input2_params["session_id"],
        "output_object_name": input2_object_name,  # Not used
        "output_user_id": input2_params["user_id"],  # Not used
        "output_session_id": input2_params["session_id"],  # Not used
        "individual_outputs": False,
    }

    # Add segmentation-specific fields if needed
    if input_type == "segmentation":
        seg_name = input2_params.get("name")
        voxel_spacing = input2_params.get("voxel_spacing")

        if isinstance(voxel_spacing, str) and voxel_spacing != "*":
            voxel_spacing = float(voxel_spacing)

        selector2_dict["segmentation_name"] = seg_name
        selector2_dict["voxel_spacing"] = voxel_spacing

    selector2_config = SelectorConfig(**selector2_dict)

    return TaskConfig(
        type="dual_selector",
        selectors=[selector1_config, selector2_config],
        pairing_method=pairing_method,
    )


def create_multi_selector_config(
    input_uris: List[str],
    input_type: Literal["mesh", "segmentation"],
    output_uri: str,
    output_type: Literal["mesh", "segmentation"],
    pairing_method: str = "n_way",
    individual_outputs: bool = False,
    command_name: Optional[str] = None,
) -> TaskConfig:
    """
    Create a multi-selector task configuration from N input URIs.

    Args:
        input_uris: List of input copick URI strings (N≥2)
        input_type: Type of input objects (all must be same type)
        output_uri: Output copick URI string
        output_type: Type of output object
        pairing_method: How to pair inputs ("n_way")
        individual_outputs: Whether to create individual outputs
        command_name: Name of command for smart defaults

    Returns:
        TaskConfig with multi_selector type

    Raises:
        ValueError: If fewer than 2 input URIs provided

    Example:
        config = create_multi_selector_config(
            input_uris=[
                "membrane:user1/manual-*",
                "vesicle:user2/auto-*",
                "ribosome:user3/pred-*"
            ],
            input_type="segmentation",
            output_uri="merged",
            output_type="segmentation",
            command_name="segop",
        )
    """
    from copick.util.uri import parse_copick_uri

    if len(input_uris) < 2:
        raise ValueError(f"At least 2 input URIs required, got {len(input_uris)}")

    # Create selector for first input (determines output config)
    first_selector = SelectorConfig.from_uris(
        input_uri=input_uris[0],
        input_type=input_type,
        output_uri=output_uri,
        output_type=output_type,
        individual_outputs=individual_outputs,
        command_name=command_name,
    )

    selectors = [first_selector]

    # Create selectors for remaining inputs (output fields unused)
    for input_uri in input_uris[1:]:
        params = parse_copick_uri(input_uri, input_type)
        object_name = params.get("object_name") or params.get("name")

        selector_dict = {
            "input_type": input_type,
            "output_type": output_type,
            "input_object_name": object_name,
            "input_user_id": params["user_id"],
            "input_session_id": params["session_id"],
            "output_object_name": object_name,  # Not used
            "output_user_id": params["user_id"],  # Not used
            "output_session_id": params["session_id"],  # Not used
            "individual_outputs": False,
        }

        # Add segmentation-specific fields
        if input_type == "segmentation":
            voxel_spacing = params.get("voxel_spacing")
            if isinstance(voxel_spacing, str) and voxel_spacing != "*":
                voxel_spacing = float(voxel_spacing)

            selector_dict["segmentation_name"] = object_name
            selector_dict["voxel_spacing"] = voxel_spacing

        selectors.append(SelectorConfig(**selector_dict))

    return TaskConfig(
        type="multi_selector",
        selectors=selectors,
        pairing_method=pairing_method,
    )


def create_reference_config(
    input_uri: str,
    input_type: Literal["picks", "mesh", "segmentation"],
    output_uri: str,
    output_type: Literal["picks", "mesh", "segmentation"],
    reference_uri: str,
    reference_type: Literal["mesh", "segmentation"],
    additional_params: Optional[Dict[str, Any]] = None,
    command_name: Optional[str] = None,
) -> TaskConfig:
    """Create a single-selector-with-reference task configuration from URIs.

    Args:
        input_uri: Input copick URI string.
        input_type: Type of input object.
        output_uri: Output copick URI string (supports smart defaults).
        output_type: Type of output object.
        reference_uri: Reference copick URI string.
        reference_type: Type of reference object.
        additional_params: Additional parameters for reference config.
        command_name: Name of the command (used for smart defaults in output_uri).

    Returns:
        TaskConfig instance ready for use.

    Example:
        config = create_reference_config(
            input_uri="ribosome:user1/all-001",
            input_type="picks",
            output_uri="ribosome",
            output_type="picks",
            reference_uri="boundary:user1/boundary-001",
            reference_type="mesh",
            command_name="picksin",
        )
    """
    selector_config = SelectorConfig.from_uris(
        input_uri=input_uri,
        input_type=input_type,
        output_uri=output_uri,
        output_type=output_type,
        command_name=command_name,
    )

    reference_config = ReferenceConfig.from_uri(
        uri=reference_uri,
        reference_type=reference_type,
        additional_params=additional_params,
    )

    return TaskConfig(
        type="single_selector_with_reference",
        selector=selector_config,
        reference=reference_config,
    )
