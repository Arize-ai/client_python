"""Public type re-exports for the evaluators subdomain."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator

from arize._generated.api_client.models.code_config import CodeConfig
from arize._generated.api_client.models.custom_code_config import (
    CustomCodeConfig,
)
from arize._generated.api_client.models.data_granularity import DataGranularity
from arize._generated.api_client.models.evaluator import Evaluator
from arize._generated.api_client.models.evaluator_list_response import (
    EvaluatorListResponse,
)
from arize._generated.api_client.models.evaluator_llm_config import (
    EvaluatorLlmConfig,
)
from arize._generated.api_client.models.evaluator_type import EvaluatorType

# Imported under private aliases so our SDK wrappers can claim the public names.
from arize._generated.api_client.models.evaluator_version import (
    EvaluatorVersion as _GenEvaluatorVersion,
)
from arize._generated.api_client.models.evaluator_version_code import (
    EvaluatorVersionCode as _GenEvaluatorVersionCode,
)
from arize._generated.api_client.models.evaluator_version_code_create import (
    EvaluatorVersionCodeCreate,
)
from arize._generated.api_client.models.evaluator_version_create import (
    EvaluatorVersionCreate,
)

# Harness and remote versions expose only common version metadata; their
# configurations are not yet accessible via the REST API, so the generated
# models are re-exported directly with no SDK-side unwrapping.
from arize._generated.api_client.models.evaluator_version_harness import (
    EvaluatorVersionHarness,
)
from arize._generated.api_client.models.evaluator_version_remote import (
    EvaluatorVersionRemote,
)
from arize._generated.api_client.models.evaluator_version_template import (
    EvaluatorVersionTemplate,
)
from arize._generated.api_client.models.evaluator_version_template_create import (
    EvaluatorVersionTemplateCreate,
)
from arize._generated.api_client.models.managed_code_config import (
    ManagedCodeConfig,
)
from arize._generated.api_client.models.managed_code_evaluator import (
    ManagedCodeEvaluator,
)
from arize._generated.api_client.models.optimization_direction import (
    OptimizationDirection,
)
from arize._generated.api_client.models.pagination_metadata import (
    PaginationMetadata,
)
from arize._generated.api_client.models.static_param import StaticParam
from arize._generated.api_client.models.static_param_default_value import (
    StaticParamDefaultValue,
)
from arize._generated.api_client.models.static_param_type import StaticParamType
from arize._generated.api_client.models.template_config import TemplateConfig


class EvaluatorVersionCode(BaseModel):
    """SDK view of the generated ``EvaluatorVersionCode`` with ``code_config`` unwrapped.

    The ``code_config`` field holds the concrete inner type
    (:class:`CustomCodeConfig` or :class:`ManagedCodeConfig`) instead of the
    oneOf wrapper :class:`CodeConfig`, so ``model_dump()`` and attribute access
    work as expected.
    """

    id: str
    evaluator_id: str
    commit_hash: str
    commit_message: str | None = None
    created_at: datetime
    created_by_user_id: str | None = None
    type: str
    code_config: CustomCodeConfig | ManagedCodeConfig

    model_config = ConfigDict(from_attributes=True)

    @field_validator("code_config", mode="before")
    @classmethod
    def _coerce_code_config(
        cls, v: object
    ) -> CustomCodeConfig | ManagedCodeConfig:
        if isinstance(v, CodeConfig):
            if v.actual_instance is None:
                raise ValueError("CodeConfig wrapper has actual_instance=None")
            return v.actual_instance
        return v  # type: ignore[return-value]


class EvaluatorWithVersion(BaseModel):
    """SDK view of the generated ``EvaluatorWithVersion`` with ``version`` unwrapped.

    The ``version`` field holds the concrete inner type
    (:class:`EvaluatorVersionCode` for code evaluators,
    :class:`EvaluatorVersionTemplate` for template evaluators,
    :class:`EvaluatorVersionHarness` for harness evaluators, or
    :class:`EvaluatorVersionRemote` for remote evaluators) instead of the
    oneOf wrapper.
    """

    id: str
    name: str
    description: str | None = None
    type: EvaluatorType
    space_id: str
    created_at: datetime
    updated_at: datetime
    created_by_user_id: str | None = None
    version: (
        EvaluatorVersionCode
        | EvaluatorVersionTemplate
        | EvaluatorVersionHarness
        | EvaluatorVersionRemote
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("version", mode="before")
    @classmethod
    def _coerce_version(
        cls, v: object
    ) -> (
        EvaluatorVersionCode
        | EvaluatorVersionTemplate
        | EvaluatorVersionHarness
        | EvaluatorVersionRemote
    ):
        if isinstance(v, _GenEvaluatorVersion):
            v = v.actual_instance
        if isinstance(v, _GenEvaluatorVersionCode):
            return EvaluatorVersionCode.model_validate(v, from_attributes=True)
        return v  # type: ignore[return-value]


class EvaluatorVersionListResponse(BaseModel):
    """SDK view of the generated ``EvaluatorVersionListResponse`` with each version unwrapped."""

    evaluator_versions: list[
        EvaluatorVersionCode
        | EvaluatorVersionTemplate
        | EvaluatorVersionHarness
        | EvaluatorVersionRemote
    ]
    pagination: PaginationMetadata

    model_config = ConfigDict(from_attributes=True)

    @field_validator("evaluator_versions", mode="before")
    @classmethod
    def _coerce_evaluator_versions(
        cls, v: object
    ) -> list[
        EvaluatorVersionCode
        | EvaluatorVersionTemplate
        | EvaluatorVersionHarness
        | EvaluatorVersionRemote
    ]:
        result = []
        for item in v:  # type: ignore[attr-defined]
            if isinstance(item, _GenEvaluatorVersion):
                item = item.actual_instance
            if isinstance(item, _GenEvaluatorVersionCode):
                item = EvaluatorVersionCode.model_validate(
                    item, from_attributes=True
                )
            result.append(item)
        return result


__all__ = [
    "CodeConfig",
    "CustomCodeConfig",
    "DataGranularity",
    "Evaluator",
    "EvaluatorListResponse",
    "EvaluatorLlmConfig",
    "EvaluatorType",
    "EvaluatorVersionCode",
    "EvaluatorVersionCodeCreate",
    "EvaluatorVersionCreate",
    "EvaluatorVersionHarness",
    "EvaluatorVersionListResponse",
    "EvaluatorVersionRemote",
    "EvaluatorVersionTemplate",
    "EvaluatorVersionTemplateCreate",
    "EvaluatorWithVersion",
    "ManagedCodeConfig",
    "ManagedCodeEvaluator",
    "OptimizationDirection",
    "StaticParam",
    "StaticParamDefaultValue",
    "StaticParamType",
    "TemplateConfig",
]
