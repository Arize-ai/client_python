"""Tests for arize.ai_integrations.types public re-exports."""

from __future__ import annotations

import pytest

import arize.ai_integrations.types as types_module
from arize.ai_integrations.types import (
    AiIntegration,
    AiIntegrationAuthType,
    AiIntegrationProvider,
    AiIntegrationScoping,
    AiIntegrationsList200Response,
)


@pytest.mark.unit
class TestAiIntegrationsTypes:
    """Tests for the ai_integrations types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        expected = {
            "AiIntegration",
            "AiIntegrationAuthType",
            "AiIntegrationProvider",
            "AiIntegrationScoping",
            "AiIntegrationsList200Response",
        }
        assert expected.issubset(set(types_module.__all__))

    def test_ai_integration_auth_type_is_enum(self) -> None:
        from enum import Enum

        assert issubclass(AiIntegrationAuthType, Enum)

    def test_ai_integration_provider_is_enum(self) -> None:
        from enum import Enum

        assert issubclass(AiIntegrationProvider, Enum)

    def test_ai_integration_scoping_is_class(self) -> None:
        assert isinstance(AiIntegrationScoping, type)

    def test_ai_integration_is_class(self) -> None:
        assert isinstance(AiIntegration, type)

    def test_ai_integrations_list_response_is_class(self) -> None:
        assert isinstance(AiIntegrationsList200Response, type)
