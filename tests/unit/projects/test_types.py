"""Tests for arize.projects.types public re-exports."""

from __future__ import annotations

import pytest

import arize.projects.types as types_module
from arize.projects.types import Project, ProjectsList200Response


@pytest.mark.unit
class TestProjectsTypes:
    """Tests for the projects types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        assert "Project" in types_module.__all__
        assert "ProjectsList200Response" in types_module.__all__

    def test_project_is_class(self) -> None:
        assert isinstance(Project, type)

    def test_projects_list_response_is_class(self) -> None:
        assert isinstance(ProjectsList200Response, type)
