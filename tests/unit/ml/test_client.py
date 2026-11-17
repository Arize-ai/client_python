"""Unit tests for the ML client."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow as pa
import pytest

from arize._generated.protocol.rec import public_pb2 as pb2
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


@pytest.mark.unit
class TestClientCertificate:
    """Batch uploads and streamed records present the client certificate."""

    CERT_PAIR = ("/certs/client.crt", "/certs/client.key")

    @staticmethod
    def _config() -> Mock:
        config = Mock(spec=SDKConfiguration)
        config.files_url = "https://api.arize.com/v1/pandas_arrow"
        config.records_url = "https://api.arize.com/v1/log"
        config.headers = {}
        config.request_verify = True
        config.pyarrow_max_chunksize = 1000
        config.max_past_years = 2
        config.client_cert_pair = TestClientCertificate.CERT_PAIR
        config.requests_verify = True
        return config

    def test_log_passes_client_cert(self) -> None:
        client = MLModelsClient(sdk_config=self._config())

        with patch(
            "arize.utils.arrow.post_arrow_table", return_value=Mock()
        ) as post:
            client.log(
                space_id="space-id",
                model_name="model",
                model_type=ModelTypes.BINARY_CLASSIFICATION,
                dataframe=pd.DataFrame({"score": [0.9, 0.1]}),
                schema=Schema(feature_column_names=["score"]),
                environment=Environments.PRODUCTION,
                validate=False,
            )

        assert post.call_args.kwargs["cert"] == self.CERT_PAIR

    def test_stream_post_passes_client_cert(self) -> None:
        client = MLModelsClient(sdk_config=self._config())
        session = Mock()

        with patch.object(
            MLModelsClient, "_ensure_session", return_value=session
        ):
            client._post(
                record=pb2.Record(),
                headers={},
                timeout=None,
                indexes=None,
            )

        assert session.post.call_args.kwargs["cert"] == self.CERT_PAIR
        assert session.post.call_args.kwargs["verify"] is True


@pytest.mark.unit
class TestRequestsVerify:
    """Batch uploads and streamed records verify against ssl_ca_cert."""

    def test_log_verify(
        self, requests_verify_case: tuple[dict[str, Any], bool | str]
    ) -> None:
        config_kwargs, expected = requests_verify_case
        client = MLModelsClient(
            sdk_config=SDKConfiguration(api_key="test_key", **config_kwargs)
        )

        with patch(
            "arize.utils.arrow.post_arrow_table", return_value=Mock()
        ) as post:
            client.log(
                space_id="space-id",
                model_name="model",
                model_type=ModelTypes.BINARY_CLASSIFICATION,
                dataframe=pd.DataFrame({"score": [0.9, 0.1]}),
                schema=Schema(feature_column_names=["score"]),
                environment=Environments.PRODUCTION,
                validate=False,
            )

        assert post.call_args.kwargs["verify"] == expected

    def test_stream_post_verify(
        self, requests_verify_case: tuple[dict[str, Any], bool | str]
    ) -> None:
        config_kwargs, expected = requests_verify_case
        client = MLModelsClient(
            sdk_config=SDKConfiguration(api_key="test_key", **config_kwargs)
        )
        session = Mock()

        with patch.object(
            MLModelsClient, "_ensure_session", return_value=session
        ):
            client._post(
                record=pb2.Record(), headers={}, timeout=None, indexes=None
            )

        assert session.post.call_args.kwargs["verify"] == expected
