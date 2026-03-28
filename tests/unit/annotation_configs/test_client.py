"""Unit tests for src/arize/annotation_configs/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from arize.annotation_configs.client import AnnotationConfigsClient
from arize.annotation_configs.types import AnnotationConfigType


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock AnnotationConfigsApi instance."""
    return Mock()


@pytest.fixture
def annotation_configs_client(
    mock_sdk_config: Mock, mock_api: Mock
) -> AnnotationConfigsClient:
    """Provide an AnnotationConfigsClient with mocked internals."""
    with patch(
        "arize._generated.api_client.AnnotationConfigsApi",
        return_value=mock_api,
    ):
        return AnnotationConfigsClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestAnnotationConfigsClientInit:
    """Tests for AnnotationConfigsClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.AnnotationConfigsApi",
            return_value=mock_api,
        ):
            client = AnnotationConfigsClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_annotation_configs_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to AnnotationConfigsApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.AnnotationConfigsApi"
        ) as mock_api_cls:
            AnnotationConfigsClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestAnnotationConfigsClientList:
    """Tests for AnnotationConfigsClient.list()."""

    def test_list_with_space_id(
        self,
        annotation_configs_client: AnnotationConfigsClient,
        mock_api: Mock,
    ) -> None:
        """list() should resolve a base64 resource ID space value to space_id."""
        annotation_configs_client.list(
            name="my-config",
            space="U3BhY2U6OTA1MDoxSmtS",
            limit=25,
            cursor="cursor-xyz",
        )

        mock_api.annotation_configs_list.assert_called_once_with(
            space_id="U3BhY2U6OTA1MDoxSmtS",
            space_name=None,
            name="my-config",
            limit=25,
            cursor="cursor-xyz",
        )

    def test_list_with_space_name(
        self,
        annotation_configs_client: AnnotationConfigsClient,
        mock_api: Mock,
    ) -> None:
        """list() should resolve a non-prefixed space value to space_name."""
        annotation_configs_client.list(
            name="my-config",
            space="my-space",
            limit=25,
            cursor="cursor-xyz",
        )

        mock_api.annotation_configs_list.assert_called_once_with(
            space_id=None,
            space_name="my-space",
            name="my-config",
            limit=25,
            cursor="cursor-xyz",
        )

    def test_list_defaults(
        self,
        annotation_configs_client: AnnotationConfigsClient,
        mock_api: Mock,
    ) -> None:
        """list() should default space/name/cursor to None and limit to 100."""
        annotation_configs_client.list()

        mock_api.annotation_configs_list.assert_called_once_with(
            space_id=None,
            space_name=None,
            name=None,
            limit=100,
            cursor=None,
        )

    def test_list_returns_api_response(
        self,
        annotation_configs_client: AnnotationConfigsClient,
        mock_api: Mock,
    ) -> None:
        """list() should propagate the return value from annotation_configs_list."""
        expected = Mock()
        mock_api.annotation_configs_list.return_value = expected

        result = annotation_configs_client.list()

        assert result is expected

    def test_list_emits_beta_prerelease_warning(
        self,
        annotation_configs_client: AnnotationConfigsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        annotation_configs_client.list()

        assert any(
            "BETA" in record.message
            and "annotation_configs.list" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestAnnotationConfigsClientCreate:
    """Tests for AnnotationConfigsClient.create() — covers the enum comparison bug fix."""

    def test_continuous_creates_correct_type(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """--type continuous must create a ContinuousAnnotationConfigCreate, not freeform."""
        with (
            patch(
                "arize._generated.api_client.ContinuousAnnotationConfigCreate"
            ) as mock_continuous_cls,
            patch(
                "arize._generated.api_client.CategoricalAnnotationConfigCreate"
            ) as mock_categorical_cls,
            patch(
                "arize._generated.api_client.FreeformAnnotationConfigCreate"
            ) as mock_freeform_cls,
            patch(
                "arize._generated.api_client.CreateAnnotationConfigRequestBody"
            ),
        ):
            annotation_configs_client.create(
                name="my-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                config_type=AnnotationConfigType.CONTINUOUS,
                minimum_score=0.0,
                maximum_score=1.0,
            )

        mock_continuous_cls.assert_called_once()
        mock_categorical_cls.assert_not_called()
        mock_freeform_cls.assert_not_called()

    def test_categorical_creates_correct_type(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """--type categorical must create a CategoricalAnnotationConfigCreate, not freeform."""
        mock_values = [Mock(), Mock()]
        with (
            patch(
                "arize._generated.api_client.ContinuousAnnotationConfigCreate"
            ) as mock_continuous_cls,
            patch(
                "arize._generated.api_client.CategoricalAnnotationConfigCreate"
            ) as mock_categorical_cls,
            patch(
                "arize._generated.api_client.FreeformAnnotationConfigCreate"
            ) as mock_freeform_cls,
            patch(
                "arize._generated.api_client.CreateAnnotationConfigRequestBody"
            ),
        ):
            annotation_configs_client.create(
                name="my-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                config_type=AnnotationConfigType.CATEGORICAL,
                values=mock_values,
            )

        mock_categorical_cls.assert_called_once()
        mock_continuous_cls.assert_not_called()
        mock_freeform_cls.assert_not_called()

    def test_freeform_creates_correct_type(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """--type freeform must create a FreeformAnnotationConfigCreate."""
        with (
            patch(
                "arize._generated.api_client.ContinuousAnnotationConfigCreate"
            ) as mock_continuous_cls,
            patch(
                "arize._generated.api_client.CategoricalAnnotationConfigCreate"
            ) as mock_categorical_cls,
            patch(
                "arize._generated.api_client.FreeformAnnotationConfigCreate"
            ) as mock_freeform_cls,
            patch(
                "arize._generated.api_client.CreateAnnotationConfigRequestBody"
            ),
        ):
            annotation_configs_client.create(
                name="my-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                config_type=AnnotationConfigType.FREEFORM,
            )

        mock_freeform_cls.assert_called_once()
        mock_continuous_cls.assert_not_called()
        mock_categorical_cls.assert_not_called()

    def test_continuous_passes_scores_to_api(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """Continuous create must forward minimum_score and maximum_score."""
        with (
            patch(
                "arize._generated.api_client.ContinuousAnnotationConfigCreate"
            ) as mock_continuous_cls,
            patch(
                "arize._generated.api_client.CreateAnnotationConfigRequestBody"
            ),
        ):
            annotation_configs_client.create(
                name="score-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                config_type=AnnotationConfigType.CONTINUOUS,
                minimum_score=0.5,
                maximum_score=10.0,
            )

        mock_continuous_cls.assert_called_once_with(
            name="score-config",
            space_id="U3BhY2U6OTA1MDoxSmtS",
            annotation_config_type=AnnotationConfigType.CONTINUOUS.value,
            minimum_score=0.5,
            maximum_score=10.0,
            optimization_direction=None,
        )

    def test_categorical_passes_values_to_api(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """Categorical create must forward values."""
        mock_values = [Mock(), Mock()]
        with (
            patch(
                "arize._generated.api_client.CategoricalAnnotationConfigCreate"
            ) as mock_categorical_cls,
            patch(
                "arize._generated.api_client.CreateAnnotationConfigRequestBody"
            ),
        ):
            annotation_configs_client.create(
                name="cat-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                config_type=AnnotationConfigType.CATEGORICAL,
                values=mock_values,
            )

        mock_categorical_cls.assert_called_once_with(
            name="cat-config",
            space_id="U3BhY2U6OTA1MDoxSmtS",
            annotation_config_type=AnnotationConfigType.CATEGORICAL.value,
            values=mock_values,
            optimization_direction=None,
        )

    def test_continuous_raises_if_minimum_score_missing(
        self, annotation_configs_client: AnnotationConfigsClient
    ) -> None:
        """Continuous create must raise ValueError when minimum_score is None."""
        with pytest.raises(
            ValueError, match="minimum_score and maximum_score are required"
        ):
            annotation_configs_client.create(
                name="my-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                config_type=AnnotationConfigType.CONTINUOUS,
                maximum_score=1.0,
            )

    def test_continuous_raises_if_maximum_score_missing(
        self, annotation_configs_client: AnnotationConfigsClient
    ) -> None:
        """Continuous create must raise ValueError when maximum_score is None."""
        with pytest.raises(
            ValueError, match="minimum_score and maximum_score are required"
        ):
            annotation_configs_client.create(
                name="my-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                config_type=AnnotationConfigType.CONTINUOUS,
                minimum_score=0.0,
            )

    def test_continuous_raises_if_both_scores_missing(
        self, annotation_configs_client: AnnotationConfigsClient
    ) -> None:
        """Continuous create must raise ValueError when both scores are None."""
        with pytest.raises(
            ValueError, match="minimum_score and maximum_score are required"
        ):
            annotation_configs_client.create(
                name="my-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                config_type=AnnotationConfigType.CONTINUOUS,
            )

    def test_categorical_raises_if_values_missing(
        self, annotation_configs_client: AnnotationConfigsClient
    ) -> None:
        """Categorical create must raise ValueError when values is None."""
        with pytest.raises(
            ValueError, match="values are required for categorical configs"
        ):
            annotation_configs_client.create(
                name="my-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                config_type=AnnotationConfigType.CATEGORICAL,
            )

    def test_create_calls_api(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """create() must call annotation_configs_create with the built request body."""
        with (
            patch("arize._generated.api_client.FreeformAnnotationConfigCreate"),
            patch(
                "arize._generated.api_client.CreateAnnotationConfigRequestBody"
            ) as mock_body_cls,
        ):
            mock_body = Mock()
            mock_body_cls.return_value = mock_body

            annotation_configs_client.create(
                name="my-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                config_type=AnnotationConfigType.FREEFORM,
            )

        mock_api.annotation_configs_create.assert_called_once_with(
            create_annotation_config_request_body=mock_body
        )

    def test_create_returns_api_response(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """create() must return the response from the API."""
        expected = Mock()
        mock_api.annotation_configs_create.return_value = expected

        with (
            patch("arize._generated.api_client.FreeformAnnotationConfigCreate"),
            patch(
                "arize._generated.api_client.CreateAnnotationConfigRequestBody"
            ),
        ):
            result = annotation_configs_client.create(
                name="my-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                config_type=AnnotationConfigType.FREEFORM,
            )

        assert result is expected

    def test_create_emits_beta_prerelease_warning(
        self,
        annotation_configs_client: AnnotationConfigsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to create() should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with (
            patch("arize._generated.api_client.FreeformAnnotationConfigCreate"),
            patch(
                "arize._generated.api_client.CreateAnnotationConfigRequestBody"
            ),
        ):
            annotation_configs_client.create(
                name="my-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                config_type=AnnotationConfigType.FREEFORM,
            )

        assert any(
            "BETA" in record.message
            and "annotation_configs.create" in record.message
            for record in caplog.records
        )
