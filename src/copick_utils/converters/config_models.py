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


class TaskConfig(BaseModel):
    """Pydantic model for complete task configuration."""

    type: Literal["single_selector", "dual_selector", "single_selector_with_reference"]
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
        return v

    @field_validator("selectors")
    @classmethod
    def validate_dual_selectors(cls, v, info):
        """Validate dual selector configuration."""
        values = info.data
        config_type = values.get("type")
        if config_type == "dual_selector" and (v is None or len(v) != 2):
            raise ValueError("exactly 2 selectors required for dual_selector type")
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


# Convenience functions for creating configurations
def create_simple_config(
    input_type: Literal["picks", "mesh", "segmentation"],
    output_type: Literal["picks", "mesh", "segmentation"],
    input_object_name: str,
    input_user_id: str,
    input_session_id: str,
    output_object_name: str,
    output_user_id: str = "converter",
    output_session_id: str = "0",
    **kwargs,
) -> TaskConfig:
    """Create a simple single-selector task configuration."""
    selector_config = SelectorConfig(
        input_type=input_type,
        output_type=output_type,
        input_object_name=input_object_name,
        input_user_id=input_user_id,
        input_session_id=input_session_id,
        output_object_name=output_object_name,
        output_user_id=output_user_id,
        output_session_id=output_session_id,
        **kwargs,
    )

    return TaskConfig(type="single_selector", selector=selector_config)


def create_reference_config(
    selector_config: SelectorConfig,
    reference_type: Literal["mesh", "segmentation"],
    reference_object_name: str,
    reference_user_id: str,
    reference_session_id: str,
    reference_voxel_spacing: Optional[float] = None,
    **additional_params,
) -> TaskConfig:
    """Create a single-selector-with-reference task configuration."""
    ref_config = ReferenceConfig(
        reference_type=reference_type,
        object_name=reference_object_name,
        user_id=reference_user_id,
        session_id=reference_session_id,
        voxel_spacing=reference_voxel_spacing,
        additional_params=additional_params,
    )

    return TaskConfig(type="single_selector_with_reference", selector=selector_config, reference=ref_config)


def create_boolean_config(
    selector1_config: SelectorConfig,
    selector2_config: SelectorConfig,
    pairing_method: str = "index_order",
    **additional_params,
) -> TaskConfig:
    """Create a dual-selector task configuration for boolean operations."""
    return TaskConfig(
        type="dual_selector",
        selectors=[selector1_config, selector2_config],
        pairing_method=pairing_method,
        additional_params=additional_params,
    )
