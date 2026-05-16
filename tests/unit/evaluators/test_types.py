"""Tests for arize.evaluators.types public re-exports."""

from __future__ import annotations

from datetime import datetime

import pytest

import arize.evaluators.types as types_module
from arize._generated.api_client.models.code_config import CodeConfig
from arize._generated.api_client.models.custom_code_config import (
    CustomCodeConfig,
)
from arize._generated.api_client.models.evaluator_type import EvaluatorType
from arize._generated.api_client.models.evaluator_version import (
    EvaluatorVersion as _GenEvaluatorVersion,
)
from arize._generated.api_client.models.evaluator_version_code import (
    EvaluatorVersionCode as _GenEvaluatorVersionCode,
)
from arize._generated.api_client.models.evaluator_version_template import (
    EvaluatorVersionTemplate,
)
from arize._generated.api_client.models.managed_code_config import (
    ManagedCodeConfig,
)
from arize._generated.api_client.models.pagination_metadata import (
    PaginationMetadata,
)
from arize.evaluators.types import (
    Evaluator,
    EvaluatorLlmConfig,
    EvaluatorVersionCode,
    EvaluatorVersionsList200Response,
    EvaluatorWithVersion,
    TemplateConfig,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_custom_code_config() -> CustomCodeConfig:
    return CustomCodeConfig.model_construct(
        type="custom", name="test_eval", code="pass", variables=[]
    )


def _make_managed_code_config() -> ManagedCodeConfig:
    return ManagedCodeConfig.model_construct(
        type="managed", name="test_eval", managed_evaluator=None, variables=[]
    )


def _make_gen_evaluator_version_code(
    code_config: CodeConfig | None = None,
) -> _GenEvaluatorVersionCode:
    if code_config is None:
        code_config = CodeConfig.model_construct(
            actual_instance=_make_custom_code_config()
        )
    return _GenEvaluatorVersionCode.model_construct(
        id="v1",
        evaluator_id="e1",
        commit_hash="abc123",
        commit_message=None,
        created_at=datetime(2024, 1, 1),
        created_by_user_id=None,
        type="code",
        code_config=code_config,
    )


def _make_evaluator_version_template() -> EvaluatorVersionTemplate:
    return EvaluatorVersionTemplate.model_construct(
        id="v2",
        evaluator_id="e1",
        commit_hash="def456",
        commit_message=None,
        created_at=datetime(2024, 1, 1),
        created_by_user_id=None,
        type="template",
        template_config=None,
    )


def _make_sdk_evaluator_version_code(
    code_config: CustomCodeConfig | ManagedCodeConfig | None = None,
) -> EvaluatorVersionCode:
    if code_config is None:
        code_config = _make_custom_code_config()
    return EvaluatorVersionCode(
        id="v1",
        evaluator_id="e1",
        commit_hash="abc123",
        commit_message=None,
        created_at=datetime(2024, 1, 1),
        created_by_user_id=None,
        type="code",
        code_config=code_config,
    )


@pytest.mark.unit
class TestEvaluatorsTypes:
    """Tests for the evaluators types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        expected = {
            "Evaluator",
            "EvaluatorLlmConfig",
            "EvaluatorVersionsList200Response",
            "EvaluatorWithVersion",
            "EvaluatorsList200Response",
            "TemplateConfig",
        }
        assert expected.issubset(set(types_module.__all__))

    @pytest.mark.parametrize(
        "cls",
        [
            Evaluator,
            EvaluatorLlmConfig,
            EvaluatorVersionsList200Response,
            EvaluatorWithVersion,
            TemplateConfig,
        ],
    )
    def test_type_is_class(self, cls: type) -> None:
        assert isinstance(cls, type)


@pytest.mark.unit
class TestEvaluatorVersionCodeCoercion:
    """Tests for EvaluatorVersionCode._coerce_code_config validator."""

    def test_unwraps_custom_code_config_from_wrapper(self) -> None:
        """A CodeConfig wrapping CustomCodeConfig should be unwrapped."""
        custom = _make_custom_code_config()
        wrapper = CodeConfig.model_construct(actual_instance=custom)

        sdk_version = EvaluatorVersionCode(
            id="v1",
            evaluator_id="e1",
            commit_hash="abc",
            commit_message=None,
            created_at=datetime(2024, 1, 1),
            created_by_user_id=None,
            type="code",
            code_config=wrapper,
        )

        assert sdk_version.code_config is custom

    def test_unwraps_managed_code_config_from_wrapper(self) -> None:
        """A CodeConfig wrapping ManagedCodeConfig should be unwrapped."""
        managed = _make_managed_code_config()
        wrapper = CodeConfig.model_construct(actual_instance=managed)

        sdk_version = EvaluatorVersionCode(
            id="v1",
            evaluator_id="e1",
            commit_hash="abc",
            commit_message=None,
            created_at=datetime(2024, 1, 1),
            created_by_user_id=None,
            type="code",
            code_config=wrapper,
        )

        assert sdk_version.code_config is managed

    def test_raises_when_wrapper_has_none_actual_instance(self) -> None:
        """CodeConfig with actual_instance=None must raise ValueError."""
        null_wrapper = CodeConfig.model_construct(actual_instance=None)

        with pytest.raises(Exception, match="actual_instance=None"):
            EvaluatorVersionCode(
                id="v1",
                evaluator_id="e1",
                commit_hash="abc",
                commit_message=None,
                created_at=datetime(2024, 1, 1),
                created_by_user_id=None,
                type="code",
                code_config=null_wrapper,
            )

    def test_passes_through_custom_code_config_directly(self) -> None:
        """A CustomCodeConfig passed directly should not be transformed."""
        custom = _make_custom_code_config()

        sdk_version = EvaluatorVersionCode(
            id="v1",
            evaluator_id="e1",
            commit_hash="abc",
            commit_message=None,
            created_at=datetime(2024, 1, 1),
            created_by_user_id=None,
            type="code",
            code_config=custom,
        )

        assert sdk_version.code_config is custom

    def test_passes_through_managed_code_config_directly(self) -> None:
        """A ManagedCodeConfig passed directly should not be transformed."""
        managed = _make_managed_code_config()

        sdk_version = EvaluatorVersionCode(
            id="v1",
            evaluator_id="e1",
            commit_hash="abc",
            commit_message=None,
            created_at=datetime(2024, 1, 1),
            created_by_user_id=None,
            type="code",
            code_config=managed,
        )

        assert sdk_version.code_config is managed


@pytest.mark.unit
class TestEvaluatorWithVersionCoercion:
    """Tests for EvaluatorWithVersion._coerce_version validator."""

    def _make_evaluator_with_version(
        self,
        version: object,
    ) -> EvaluatorWithVersion:
        return EvaluatorWithVersion(
            id="e1",
            name="my-eval",
            description=None,
            type=EvaluatorType.CODE,
            space_id="s1",
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
            created_by_user_id=None,
            version=version,
        )

    def test_unwraps_gen_evaluator_version_code_from_wrapper(self) -> None:
        """EvaluatorVersion wrapping a _GenEvaluatorVersionCode should yield SDK EvaluatorVersionCode."""
        gen_code = _make_gen_evaluator_version_code()
        gen_version = _GenEvaluatorVersion.model_construct(
            actual_instance=gen_code
        )

        result = self._make_evaluator_with_version(gen_version)

        assert isinstance(result.version, EvaluatorVersionCode)

    def test_unwraps_evaluator_version_template_from_wrapper(self) -> None:
        """EvaluatorVersion wrapping an EvaluatorVersionTemplate should yield EvaluatorVersionTemplate."""
        tmpl = _make_evaluator_version_template()
        gen_version = _GenEvaluatorVersion.model_construct(actual_instance=tmpl)

        result = EvaluatorWithVersion(
            id="e1",
            name="my-eval",
            description=None,
            type=EvaluatorType.TEMPLATE,
            space_id="s1",
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
            created_by_user_id=None,
            version=gen_version,
        )

        assert isinstance(result.version, EvaluatorVersionTemplate)
        assert result.version.id == "v2"

    def test_converts_raw_gen_evaluator_version_code(self) -> None:
        """A _GenEvaluatorVersionCode passed directly should be converted to SDK EvaluatorVersionCode."""
        gen_code = _make_gen_evaluator_version_code()

        result = self._make_evaluator_with_version(gen_code)

        assert isinstance(result.version, EvaluatorVersionCode)
        assert result.version.id == "v1"

    def test_passes_through_evaluator_version_template(self) -> None:
        """An EvaluatorVersionTemplate passed directly should not be transformed."""
        tmpl = _make_evaluator_version_template()

        result = EvaluatorWithVersion(
            id="e1",
            name="my-eval",
            description=None,
            type=EvaluatorType.TEMPLATE,
            space_id="s1",
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
            created_by_user_id=None,
            version=tmpl,
        )

        assert isinstance(result.version, EvaluatorVersionTemplate)

    def test_code_config_is_unwrapped_inside_version(self) -> None:
        """The SDK EvaluatorVersionCode produced by coercion should have an unwrapped code_config."""
        custom = _make_custom_code_config()
        code_config_wrapper = CodeConfig.model_construct(actual_instance=custom)
        gen_code = _make_gen_evaluator_version_code(
            code_config=code_config_wrapper
        )

        result = self._make_evaluator_with_version(gen_code)

        assert isinstance(result.version, EvaluatorVersionCode)
        assert isinstance(result.version.code_config, CustomCodeConfig)


@pytest.mark.unit
class TestEvaluatorVersionsListCoercion:
    """Tests for EvaluatorVersionsList200Response._coerce_evaluator_versions validator."""

    def _make_pagination(self) -> PaginationMetadata:
        return PaginationMetadata.model_construct(
            total_count=0, has_next_page=False, cursor=None
        )

    def test_unwraps_gen_evaluator_version_code_in_list(self) -> None:
        """_GenEvaluatorVersionCode items in the list should be converted to SDK EvaluatorVersionCode."""
        gen_code = _make_gen_evaluator_version_code()

        response = EvaluatorVersionsList200Response(
            evaluator_versions=[gen_code],
            pagination=self._make_pagination(),
        )

        assert len(response.evaluator_versions) == 1
        assert isinstance(response.evaluator_versions[0], EvaluatorVersionCode)

    def test_unwraps_gen_evaluator_version_wrapper_in_list(self) -> None:
        """_GenEvaluatorVersion wrapper items should be unwrapped and converted."""
        gen_code = _make_gen_evaluator_version_code()
        gen_version = _GenEvaluatorVersion.model_construct(
            actual_instance=gen_code
        )

        response = EvaluatorVersionsList200Response(
            evaluator_versions=[gen_version],
            pagination=self._make_pagination(),
        )

        assert isinstance(response.evaluator_versions[0], EvaluatorVersionCode)

    def test_passes_through_evaluator_version_template_in_list(self) -> None:
        """EvaluatorVersionTemplate items should pass through unchanged."""
        tmpl = _make_evaluator_version_template()

        response = EvaluatorVersionsList200Response(
            evaluator_versions=[tmpl],
            pagination=self._make_pagination(),
        )

        assert isinstance(
            response.evaluator_versions[0], EvaluatorVersionTemplate
        )

    def test_handles_mixed_version_list(self) -> None:
        """A list with both code and template versions should coerce each correctly."""
        gen_code = _make_gen_evaluator_version_code()
        tmpl = _make_evaluator_version_template()

        response = EvaluatorVersionsList200Response(
            evaluator_versions=[gen_code, tmpl],
            pagination=self._make_pagination(),
        )

        assert isinstance(response.evaluator_versions[0], EvaluatorVersionCode)
        assert isinstance(
            response.evaluator_versions[1], EvaluatorVersionTemplate
        )

    def test_handles_empty_list(self) -> None:
        """An empty list should produce an empty evaluator_versions list."""
        response = EvaluatorVersionsList200Response(
            evaluator_versions=[],
            pagination=self._make_pagination(),
        )

        assert response.evaluator_versions == []
