"""Tests for protocol buffer schema utilities."""

from unittest.mock import MagicMock, patch

import pytest

from arize._generated.protocol.rec import public_pb2 as pb2
from arize.utils.proto import get_pb_schema_tracing


@pytest.mark.unit
class TestGetPbSchemaTracing:
    """Test get_pb_schema_tracing function."""

    def test_creates_schema_object(self) -> None:
        """Should return a pb2.Schema instance."""
        result = get_pb_schema_tracing("test_project")
        assert isinstance(result, pb2.Schema)

    def test_sets_model_id_from_project_name(self) -> None:
        """Should set schema.constants.model_id to the project_name."""
        project_name = "test_project"
        schema = get_pb_schema_tracing(project_name)
        assert schema.constants.model_id == project_name

    def test_sets_environment_to_tracing(self) -> None:
        """Should set schema.constants.environment to TRACING."""
        schema = get_pb_schema_tracing("test_project")
        assert schema.constants.environment == pb2.Schema.Environment.TRACING

    def test_sets_model_type_to_generative_llm(self) -> None:
        """Should set schema.constants.model_type to GENERATIVE_LLM."""
        schema = get_pb_schema_tracing("test_project")
        assert (
            schema.constants.model_type == pb2.Schema.ModelType.GENERATIVE_LLM
        )

    def test_enables_arize_spans(self) -> None:
        """Should call schema.arize_spans.SetInParent()."""
        with patch.object(pb2.Schema, "__new__") as mock_schema_new:
            mock_schema = MagicMock(spec=pb2.Schema)
            mock_schema.constants = MagicMock()
            mock_schema.arize_spans = MagicMock()
            mock_schema_new.return_value = mock_schema

            get_pb_schema_tracing("test_project")
            mock_schema.arize_spans.SetInParent.assert_called_once()

    def test_empty_project_name(self) -> None:
        """Should handle empty string as project_name."""
        schema = get_pb_schema_tracing("")
        assert schema.constants.model_id == ""
        assert isinstance(schema, pb2.Schema)

    def test_long_project_name(self) -> None:
        """Should handle long project names."""
        long_name = "a" * 1000
        schema = get_pb_schema_tracing(long_name)
        assert schema.constants.model_id == long_name

    @pytest.mark.parametrize(
        "special_name",
        [
            "project-with-dashes",
            "project_with_underscores",
            "project.with.dots",
            "project with spaces",
            "project/with/slashes",
            "プロジェクト",  # Japanese characters
            "проект",  # Cyrillic characters
            "🚀 emoji project 🎯",
        ],
    )
    def test_special_characters_in_project_name(
        self, special_name: str
    ) -> None:
        """Should handle special characters and unicode in project_name."""
        schema = get_pb_schema_tracing(special_name)
        assert schema.constants.model_id == special_name
        assert isinstance(schema, pb2.Schema)
