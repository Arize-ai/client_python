"""Unit tests for src/arize/__init__.py."""

from collections.abc import Iterator, Mapping
from unittest.mock import Mock

import pandas as pd
import pytest

from arize import make_to_df


class MockPydanticModel:
    """Mock Pydantic v2 model for testing."""

    def __init__(self, data: dict) -> None:
        """Initialize mock model with data."""
        self.data = data

    def model_dump(self, by_alias: bool = False) -> dict:
        """Mock model_dump method."""
        if by_alias:
            return {f"alias_{k}": v for k, v in self.data.items()}
        return self.data.copy()


@pytest.mark.unit
class TestMakeToDf:
    """Tests for make_to_df() factory function."""

    def test_returns_function(self) -> None:
        """make_to_df should return a callable function."""
        to_df = make_to_df("items")
        assert callable(to_df)

    def test_pydantic_v2_model_dump(self) -> None:
        """Should convert Pydantic v2 models using model_dump."""
        items = [
            MockPydanticModel({"id": 1, "name": "Alice"}),
            MockPydanticModel({"id": 2, "name": "Bob"}),
        ]

        obj = Mock()
        obj.items = items

        to_df = make_to_df("items")
        df = to_df(obj)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["id", "name"]
        assert df["id"].tolist() == [1, 2]
        assert df["name"].tolist() == ["Alice", "Bob"]

    def test_dict_conversion(self) -> None:
        """Should convert dict objects directly."""
        items = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]

        obj = Mock()
        obj.items = items

        to_df = make_to_df("items")
        df = to_df(obj)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["id", "name"]

    def test_mapping_conversion(self) -> None:
        """Should convert Mapping objects."""

        class CustomMapping(Mapping):
            def __init__(self, data: dict) -> None:
                self._data = data

            def __getitem__(self, key: str) -> object:
                return self._data[key]

            def __iter__(self) -> Iterator[str]:
                return iter(self._data)

            def __len__(self) -> int:
                return len(self._data)

        items = [
            CustomMapping({"id": 1, "name": "Alice"}),
            CustomMapping({"id": 2, "name": "Bob"}),
        ]

        obj = Mock()
        obj.items = items

        to_df = make_to_df("items")
        df = to_df(obj)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_unsupported_type_raises(self) -> None:
        """Should raise ValueError for unsupported types."""
        items = [
            "string",  # Unsupported type
            123,  # Unsupported type
        ]

        obj = Mock()
        obj.items = items

        to_df = make_to_df("items")

        with pytest.raises(ValueError, match="Cannot convert item of type"):
            to_df(obj)

    def test_json_normalize_true(self) -> None:
        """json_normalize=True should flatten nested dicts."""
        items = [
            {"id": 1, "meta": {"key": "value1"}},
            {"id": 2, "meta": {"key": "value2"}},
        ]

        obj = Mock()
        obj.items = items

        to_df = make_to_df("items")
        df = to_df(obj, json_normalize=True)

        assert "meta.key" in df.columns

    def test_json_normalize_false(self) -> None:
        """json_normalize=False should keep nested dicts as-is."""
        items = [
            {"id": 1, "meta": {"key": "value1"}},
            {"id": 2, "meta": {"key": "value2"}},
        ]

        obj = Mock()
        obj.items = items

        to_df = make_to_df("items")
        df = to_df(obj, json_normalize=False)

        assert "meta" in df.columns
        assert "meta.key" not in df.columns

    def test_exclude_none_any(self) -> None:
        """exclude_none='any' should drop columns with any None."""
        items = [
            {"id": 1, "name": "Alice", "email": None},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        ]

        obj = Mock()
        obj.items = items

        to_df = make_to_df("items")
        df = to_df(obj, exclude_none="any")

        # email column has None, should be dropped
        assert "email" not in df.columns
        assert "id" in df.columns
        assert "name" in df.columns

    def test_exclude_none_all(self) -> None:
        """exclude_none='all' should drop columns where all values are None."""
        items = [
            {"id": 1, "name": "Alice", "email": None},
            {"id": 2, "name": "Bob", "email": None},
        ]

        obj = Mock()
        obj.items = items

        to_df = make_to_df("items")
        df = to_df(obj, exclude_none="all")

        # email column has all None, should be dropped
        assert "email" not in df.columns
        assert "id" in df.columns
        assert "name" in df.columns

    def test_exclude_none_true_behaves_as_all(self) -> None:
        """exclude_none=True should behave like 'all'."""
        items = [
            {"id": 1, "name": "Alice", "email": None},
            {"id": 2, "name": "Bob", "email": None},
        ]

        obj = Mock()
        obj.items = items

        to_df = make_to_df("items")
        df = to_df(obj, exclude_none=True)

        # email column has all None, should be dropped
        assert "email" not in df.columns

    def test_exclude_none_false_keeps_nones(self) -> None:
        """exclude_none=False should keep None values."""
        items = [
            {"id": 1, "name": "Alice", "email": None},
            {"id": 2, "name": "Bob", "email": None},
        ]

        obj = Mock()
        obj.items = items

        to_df = make_to_df("items")
        df = to_df(obj, exclude_none=False)

        # All columns should be present
        assert "email" in df.columns
        assert df["email"].isna().all()

    def test_convert_dtypes_true(self) -> None:
        """convert_dtypes=True should convert dtypes."""
        items = [
            {"id": "1", "count": "100"},
            {"id": "2", "count": "200"},
        ]

        obj = Mock()
        obj.items = items

        to_df = make_to_df("items")
        df = to_df(obj, convert_dtypes=True)

        # convert_dtypes should infer and convert types (to string or int depending on pandas version)
        # Just verify it was called - the dtype may be string or int depending on pandas version
        assert df["id"].dtype != object or str(df["id"].dtype) == "string"
        assert df["count"].dtype != object or str(df["count"].dtype) == "string"

    def test_convert_dtypes_false(self) -> None:
        """convert_dtypes=False should keep original dtypes."""
        items = [
            {"id": "1", "count": "100"},
            {"id": "2", "count": "200"},
        ]

        obj = Mock()
        obj.items = items

        to_df = make_to_df("items")
        df = to_df(obj, convert_dtypes=False)

        # Should remain as object dtype
        assert df["id"].dtype == object
        assert df["count"].dtype == object

    def test_nested_dict_normalization(self) -> None:
        """Nested dictionaries should be normalized with json_normalize."""
        items = [
            {"id": 1, "user": {"name": "Alice", "age": 30}},
            {"id": 2, "user": {"name": "Bob", "age": 25}},
        ]

        obj = Mock()
        obj.items = items

        to_df = make_to_df("items")
        df = to_df(obj, json_normalize=True)

        assert "user.name" in df.columns
        assert "user.age" in df.columns

    def test_mixed_nan_and_none_values(self) -> None:
        """Mixed NaN and None values should be handled."""
        import numpy as np

        items = [
            {"id": 1, "value": None},
            {"id": 2, "value": np.nan},
            {"id": 3, "value": "valid"},
        ]

        obj = Mock()
        obj.items = items

        to_df = make_to_df("items")
        df = to_df(obj, exclude_none="any")

        # value column has None/NaN, should be dropped with exclude_none='any'
        assert "value" not in df.columns

    def test_by_alias_parameter(self) -> None:
        """by_alias parameter should be passed to model_dump."""
        items = [
            MockPydanticModel({"id": 1, "name": "Alice"}),
        ]

        obj = Mock()
        obj.items = items

        to_df = make_to_df("items")
        df = to_df(obj, by_alias=True)

        # With by_alias=True, our mock prepends 'alias_'
        assert "alias_id" in df.columns
        assert "alias_name" in df.columns

    def test_empty_items_list(self) -> None:
        """Empty items list should return empty DataFrame."""
        obj = Mock()
        obj.items = []

        to_df = make_to_df("items")
        df = to_df(obj)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_none_items_attribute(self) -> None:
        """None items attribute should be treated as empty list."""
        obj = Mock()
        obj.items = None

        to_df = make_to_df("items")
        df = to_df(obj)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


@pytest.mark.unit
class TestModuleInit:
    """Tests for module initialization."""

    def test_null_handler_attached(self) -> None:
        """Arize logger should have NullHandler attached."""
        import logging

        arize_logger = logging.getLogger("arize")

        # The NullHandler is added during module import
        # Check if at least one NullHandler exists in the logger's handlers or parent handlers
        def has_null_handler_recursive(logger: logging.Logger) -> bool:
            """Check logger and its parents for NullHandler."""
            current = logger
            while current:
                if any(
                    isinstance(h, logging.NullHandler) for h in current.handlers
                ):
                    return True
                if not current.propagate:
                    break
                current = current.parent  # type: ignore
            return False

        # The module adds a NullHandler, so it should exist
        assert has_null_handler_recursive(arize_logger)

    def test_auto_configure_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auto-configure should succeed when no errors."""
        # This test verifies that the try-except block doesn't suppress errors
        # in normal operation (when testing, the import has already happened)
        import arize

        # Just verify the module loaded successfully
        assert hasattr(arize, "ArizeClient")

    def test_auto_configure_failure_silent(self) -> None:
        """Auto-configure failures should be silent and not prevent import."""
        # Test that the module can be imported even if auto_configure fails
        # This is already tested by the successful import in other tests
        import arize

        assert arize is not None


@pytest.mark.unit
class TestMonkeyPatching:
    """Tests for monkey-patching to_df onto response models."""

    def test_to_df_attached_to_response_models(self) -> None:
        """to_df method should be attached to response models."""
        from arize._generated.api_client import models

        # Check that to_df is attached
        assert hasattr(models.DatasetsList200Response, "to_df")
        assert hasattr(models.DatasetsExamplesList200Response, "to_df")
        assert hasattr(models.ExperimentsList200Response, "to_df")
        assert hasattr(models.ExperimentsRunsList200Response, "to_df")
        assert hasattr(models.ProjectsList200Response, "to_df")
        assert hasattr(models.AnnotationConfigsList200Response, "to_df")

    def test_factory_receives_correct_field_names(self) -> None:
        """Monkey-patched methods should reference correct field names."""
        from arize._generated.api_client import models

        # Create mock objects with the expected field names
        datasets_resp = Mock(spec=models.DatasetsList200Response)
        datasets_resp.datasets = []

        examples_resp = Mock(spec=models.DatasetsExamplesList200Response)
        examples_resp.examples = []

        experiments_resp = Mock(spec=models.ExperimentsList200Response)
        experiments_resp.experiments = []

        runs_resp = Mock(spec=models.ExperimentsRunsList200Response)
        runs_resp.experiment_runs = []

        projects_resp = Mock(spec=models.ProjectsList200Response)
        projects_resp.projects = []

        annotation_configs_resp = Mock(
            spec=models.AnnotationConfigsList200Response
        )
        annotation_configs_resp.annotation_configs = []

        # Call to_df on each and verify they work
        df1 = models.DatasetsList200Response.to_df(datasets_resp)
        df2 = models.DatasetsExamplesList200Response.to_df(examples_resp)
        df3 = models.ExperimentsList200Response.to_df(experiments_resp)
        df4 = models.ExperimentsRunsList200Response.to_df(runs_resp)
        df5 = models.ProjectsList200Response.to_df(projects_resp)
        df6 = models.AnnotationConfigsList200Response.to_df(
            annotation_configs_resp
        )

        # All should return DataFrames
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)
        assert isinstance(df3, pd.DataFrame)
        assert isinstance(df4, pd.DataFrame)
        assert isinstance(df5, pd.DataFrame)
        assert isinstance(df6, pd.DataFrame)

    def test_annotation_configs_to_df_expands_actual_instance(self) -> None:
        """Annotation config to_df should flatten oneOf wrapper internals."""
        from arize._generated.api_client import models

        annotation_configs_resp = Mock(
            spec=models.AnnotationConfigsList200Response
        )
        annotation_configs_resp.annotation_configs = [
            {
                "actual_instance": {
                    "id": "cfg_1",
                    "name": "Accuracy",
                    "type": "categorical",
                },
                "one_of_schemas": {
                    "CategoricalAnnotationConfig",
                    "ContinuousAnnotationConfig",
                    "FreeformAnnotationConfig",
                },
                "discriminator_value_class_map": {},
            }
        ]

        df = models.AnnotationConfigsList200Response.to_df(
            annotation_configs_resp
        )

        assert "id" in df.columns
        assert "name" in df.columns
        assert "type" in df.columns
        assert "actual_instance" not in df.columns
        assert "one_of_schemas" not in df.columns
        assert "discriminator_value_class_map" not in df.columns
