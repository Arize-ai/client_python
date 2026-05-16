"""Public types for the annotation_configs subdomain."""

from pydantic import BaseModel, ConfigDict, field_validator

from arize._generated.api_client.models.annotation_config import (
    AnnotationConfig as _GenAnnotationConfig,
)
from arize._generated.api_client.models.annotation_config_type import (
    AnnotationConfigType,
)
from arize._generated.api_client.models.categorical_annotation_config import (
    CategoricalAnnotationConfig,
)
from arize._generated.api_client.models.categorical_annotation_value import (
    CategoricalAnnotationValue,
)
from arize._generated.api_client.models.continuous_annotation_config import (
    ContinuousAnnotationConfig,
)
from arize._generated.api_client.models.freeform_annotation_config import (
    FreeformAnnotationConfig,
)
from arize._generated.api_client.models.optimization_direction import (
    OptimizationDirection,
)
from arize._generated.api_client.models.pagination_metadata import (
    PaginationMetadata,
)


class AnnotationConfigsList200Response(BaseModel):
    """SDK view of the generated list response with each ``AnnotationConfig`` unwrapped.

    The ``annotation_configs`` field contains the concrete inner types
    (:class:`CategoricalAnnotationConfig`, :class:`ContinuousAnnotationConfig`,
    or :class:`FreeformAnnotationConfig`) instead of the oneOf wrapper.
    """

    annotation_configs: list[
        CategoricalAnnotationConfig
        | ContinuousAnnotationConfig
        | FreeformAnnotationConfig
    ]
    pagination: PaginationMetadata

    model_config = ConfigDict(from_attributes=True)

    @field_validator("annotation_configs", mode="before")
    @classmethod
    def _coerce_annotation_configs(
        cls, v: object
    ) -> list[
        CategoricalAnnotationConfig
        | ContinuousAnnotationConfig
        | FreeformAnnotationConfig
    ]:
        result = []
        for item in v:  # type: ignore[attr-defined]
            if isinstance(item, _GenAnnotationConfig):
                if item.actual_instance is None:
                    raise ValueError(
                        "AnnotationConfig wrapper has actual_instance=None"
                    )
                item = item.actual_instance
            result.append(item)
        return result


__all__ = [
    "AnnotationConfigType",
    "AnnotationConfigsList200Response",
    "CategoricalAnnotationConfig",
    "CategoricalAnnotationValue",
    "ContinuousAnnotationConfig",
    "FreeformAnnotationConfig",
    "OptimizationDirection",
    "PaginationMetadata",
]
