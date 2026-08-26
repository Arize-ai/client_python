"""Unit tests for the ML client."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pyarrow as pa
import pytest

from arize.config import SDKConfiguration
from arize.ml.client import MLModelsClient
from arize.ml.types import Environments, ModelTypes, Schema


@pytest.mark.unit
class TestCategoricalDtypeDetection:
    """Categorical columns are converted before Arrow serialization."""

    def test_log_converts_categorical_columns_to_strings(self) -> None:
        """log() should not pass pandas categorical columns to PyArrow."""
        df = pd.DataFrame(
            {
                "label": pd.Categorical(["cat", "dog", "cat"]),
                "score": [0.9, 0.1, 0.8],
            }
        )
        config = Mock(spec=SDKConfiguration)
        config.files_url = "https://api.arize.com/v1/pandas_arrow"
        config.headers = {}
        config.request_verify = True
        config.pyarrow_max_chunksize = 1000
        config.max_past_years = 2
        client = MLModelsClient(sdk_config=config)

        with patch(
            "arize.utils.arrow.post_arrow_table", return_value=Mock()
        ) as post:
            client.log(
                space_id="space-id",
                model_name="model",
                model_type=ModelTypes.BINARY_CLASSIFICATION,
                dataframe=df,
                schema=Schema(feature_column_names=["label", "score"]),
                environment=Environments.PRODUCTION,
                validate=False,
            )

        assert (
            post.call_args.kwargs["pa_table"].schema.field("label").type
            == pa.string()
        )

    def test_log_categorical_columns_no_deprecation_warning(
        self, recwarn: pytest.WarningsChecker
    ) -> None:
        """log() must not emit a pandas DeprecationWarning for categorical dtype detection."""
        df = pd.DataFrame({"label": pd.Categorical(["cat", "dog"])})
        config = Mock(spec=SDKConfiguration)
        config.files_url = "https://api.arize.com/v1/pandas_arrow"
        config.headers = {}
        config.request_verify = True
        config.pyarrow_max_chunksize = 1000
        config.max_past_years = 2
        client = MLModelsClient(sdk_config=config)

        with patch("arize.utils.arrow.post_arrow_table", return_value=Mock()):
            client.log(
                space_id="space-id",
                model_name="model",
                model_type=ModelTypes.BINARY_CLASSIFICATION,
                dataframe=df,
                schema=Schema(feature_column_names=["label"]),
                environment=Environments.PRODUCTION,
                validate=False,
            )

        pandas_deprecations = [
            w
            for w in recwarn.list
            if issubclass(w.category, DeprecationWarning)
            and "is_categorical_dtype" in str(w.message)
        ]
        assert pandas_deprecations == []
