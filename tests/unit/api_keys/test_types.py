"""Tests for arize.api_keys.types public re-exports."""

from __future__ import annotations

import pytest

import arize.api_keys.types as types_module
from arize.api_keys.types import (
    ApiKeyCreated,
    ApiKeysList200Response,
    ApiKeyStatus,
)


@pytest.mark.unit
class TestApiKeysTypes:
    """Tests for the api_keys types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        assert "ApiKeyCreated" in types_module.__all__
        assert "ApiKeyStatus" in types_module.__all__
        assert "ApiKeysList200Response" in types_module.__all__

    def test_api_key_status_is_enum(self) -> None:
        from enum import Enum

        assert issubclass(ApiKeyStatus, Enum)

    def test_api_key_created_is_class(self) -> None:
        assert isinstance(ApiKeyCreated, type)

    def test_api_keys_list_response_is_class(self) -> None:
        assert isinstance(ApiKeysList200Response, type)
