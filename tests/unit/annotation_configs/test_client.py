"""Unit tests for src/arize/annotation_configs/client.py."""

from __future__ import annotations

from unittest.mock import Mock, create_autospec, patch

import pytest

from arize._generated.api_client import (
    AnnotationConfigsApi,
    CategoricalAnnotationValueRequest,
)
from arize.annotation_configs.client import AnnotationConfigsClient
from arize.annotation_configs.types import (
    AnnotationConfigType,
    ListAnnotationConfigsResponse,
)


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock AnnotationConfigsApi instance."""
    return create_autospec(AnnotationConfigsApi, instance=True)


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

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            ListAnnotationConfigsResponse,
            "model_validate",
            side_effect=lambda v, **kw: v,
        ):
            yield

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

        mock_api.list_annotation_configs.assert_called_once_with(
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

        mock_api.list_annotation_configs.assert_called_once_with(
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
        """list() should default space/name/cursor to None and limit to 50."""
        annotation_configs_client.list()

        mock_api.list_annotation_configs.assert_called_once_with(
            space_id=None,
            space_name=None,
            name=None,
            limit=50,
            cursor=None,
        )

    def test_list_returns_api_response(
        self,
        annotation_configs_client: AnnotationConfigsClient,
        mock_api: Mock,
    ) -> None:
        """list() should propagate the return value from annotation_configs_list."""
        expected = Mock()
        mock_api.list_annotation_configs.return_value = expected

        result = annotation_configs_client.list()

        assert result is expected


@pytest.mark.unit
class TestAnnotationConfigsClientCreateContinuous:
    """Tests for AnnotationConfigsClient.create_continuous()."""

    def test_passes_scores_to_api(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """create_continuous() must forward minimum_score and maximum_score."""
        with (
            patch(
                "arize._generated.api_client.CreateContinuousAnnotationConfigRequest"
            ) as mock_continuous_cls,
            patch("arize._generated.api_client.CreateAnnotationConfigRequest"),
        ):
            annotation_configs_client.create_continuous(
                name="score-config",
                space="U3BhY2U6OTA1MDoxSmtS",
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

    def test_passes_optimization_direction_to_api(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """create_continuous() must forward optimization_direction when given."""
        with (
            patch(
                "arize._generated.api_client.CreateContinuousAnnotationConfigRequest"
            ) as mock_continuous_cls,
            patch("arize._generated.api_client.CreateAnnotationConfigRequest"),
        ):
            annotation_configs_client.create_continuous(
                name="score-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                minimum_score=0.0,
                maximum_score=1.0,
                optimization_direction="maximize",
            )

        mock_continuous_cls.assert_called_once_with(
            name="score-config",
            space_id="U3BhY2U6OTA1MDoxSmtS",
            annotation_config_type=AnnotationConfigType.CONTINUOUS.value,
            minimum_score=0.0,
            maximum_score=1.0,
            optimization_direction="maximize",
        )

    def test_calls_api_and_returns_response(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """create_continuous() must call the API and unwrap the response."""
        expected = Mock()
        mock_api.create_annotation_config.return_value.actual_instance = (
            expected
        )

        with (
            patch(
                "arize._generated.api_client.CreateContinuousAnnotationConfigRequest"
            ),
            patch(
                "arize._generated.api_client.CreateAnnotationConfigRequest"
            ) as mock_body_cls,
        ):
            mock_body = Mock()
            mock_body_cls.return_value = mock_body

            result = annotation_configs_client.create_continuous(
                name="score-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                minimum_score=0.0,
                maximum_score=1.0,
            )

        mock_api.create_annotation_config.assert_called_once_with(
            create_annotation_config_request=mock_body
        )
        assert result is expected


@pytest.mark.unit
class TestAnnotationConfigsClientCreateCategorical:
    """Tests for AnnotationConfigsClient.create_categorical()."""

    def test_passes_values_to_api(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """create_categorical() must forward space_id and values."""
        mock_values = [
            Mock(spec=CategoricalAnnotationValueRequest),
            Mock(spec=CategoricalAnnotationValueRequest),
        ]
        with (
            patch(
                "arize._generated.api_client.CreateCategoricalAnnotationConfigRequest"
            ) as mock_categorical_cls,
            patch("arize._generated.api_client.CreateAnnotationConfigRequest"),
        ):
            annotation_configs_client.create_categorical(
                name="cat-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                values=mock_values,
            )

        mock_categorical_cls.assert_called_once_with(
            name="cat-config",
            space_id="U3BhY2U6OTA1MDoxSmtS",
            annotation_config_type=AnnotationConfigType.CATEGORICAL.value,
            values=mock_values,
            optimization_direction=None,
        )

    def test_passes_optimization_direction_to_api(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """create_categorical() must forward optimization_direction when given."""
        mock_values = [Mock(spec=CategoricalAnnotationValueRequest)]
        with (
            patch(
                "arize._generated.api_client.CreateCategoricalAnnotationConfigRequest"
            ) as mock_categorical_cls,
            patch("arize._generated.api_client.CreateAnnotationConfigRequest"),
        ):
            annotation_configs_client.create_categorical(
                name="cat-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                values=mock_values,
                optimization_direction="maximize",
            )

        mock_categorical_cls.assert_called_once_with(
            name="cat-config",
            space_id="U3BhY2U6OTA1MDoxSmtS",
            annotation_config_type=AnnotationConfigType.CATEGORICAL.value,
            values=mock_values,
            optimization_direction="maximize",
        )

    def test_calls_api_and_returns_response(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """create_categorical() must call the API and unwrap the response."""
        expected = Mock()
        mock_api.create_annotation_config.return_value.actual_instance = (
            expected
        )

        with (
            patch(
                "arize._generated.api_client.CreateCategoricalAnnotationConfigRequest"
            ),
            patch(
                "arize._generated.api_client.CreateAnnotationConfigRequest"
            ) as mock_body_cls,
        ):
            mock_body = Mock()
            mock_body_cls.return_value = mock_body

            result = annotation_configs_client.create_categorical(
                name="cat-config",
                space="U3BhY2U6OTA1MDoxSmtS",
                values=[Mock(spec=CategoricalAnnotationValueRequest)],
            )

        mock_api.create_annotation_config.assert_called_once_with(
            create_annotation_config_request=mock_body
        )
        assert result is expected


# base64("AnnotationConfig:1234:xYz") — passes is_resource_id()
_ANNOTATION_CONFIG_ID = "QW5ub3RhdGlvbkNvbmZpZzoxMjM0OnhZeg=="


@pytest.mark.unit
class TestAnnotationConfigsClientGet:
    """Tests for AnnotationConfigsClient.get()."""

    def test_get_calls_api_and_returns_response(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """get() should resolve the config ID and call get_annotation_config."""
        expected = Mock()
        mock_api.get_annotation_config.return_value.actual_instance = expected

        result = annotation_configs_client.get(
            annotation_config=_ANNOTATION_CONFIG_ID,
        )

        mock_api.get_annotation_config.assert_called_once_with(
            annotation_config_id=_ANNOTATION_CONFIG_ID
        )
        assert result is expected


@pytest.mark.unit
class TestAnnotationConfigsClientUpdateContinuous:
    """Tests for AnnotationConfigsClient.update_continuous()."""

    def test_update_continuous_builds_request_and_calls_api(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """update_continuous() should build the correct request and call the API."""
        expected = Mock()
        mock_api.update_annotation_config.return_value = Mock(
            actual_instance=expected
        )

        with (
            patch(
                "arize._generated.api_client.UpdateContinuousAnnotationConfigRequest"
            ) as mock_update_cls,
            patch(
                "arize._generated.api_client.UpdateAnnotationConfigRequest"
            ) as mock_body_cls,
        ):
            mock_inner = Mock()
            mock_update_cls.return_value = mock_inner
            mock_body = Mock()
            mock_body_cls.return_value = mock_body

            result = annotation_configs_client.update_continuous(
                annotation_config=_ANNOTATION_CONFIG_ID,
                minimum_score=0.0,
                maximum_score=1.0,
            )

        mock_body_cls.assert_called_once_with(actual_instance=mock_inner)
        mock_api.update_annotation_config.assert_called_once_with(
            annotation_config_id=_ANNOTATION_CONFIG_ID,
            update_annotation_config_request=mock_body,
        )
        assert result is expected


@pytest.mark.unit
class TestAnnotationConfigsClientUpdateCategorical:
    """Tests for AnnotationConfigsClient.update_categorical()."""

    def test_update_categorical_builds_request_and_calls_api(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """update_categorical() should coerce values and call the API."""
        expected = Mock()
        mock_api.update_annotation_config.return_value = Mock(
            actual_instance=expected
        )

        mock_val = Mock(spec=CategoricalAnnotationValueRequest)
        with (
            patch(
                "arize._generated.api_client.UpdateCategoricalAnnotationConfigRequest"
            ) as mock_update_cls,
            patch(
                "arize._generated.api_client.UpdateAnnotationConfigRequest"
            ) as mock_body_cls,
        ):
            mock_inner = Mock()
            mock_update_cls.return_value = mock_inner
            mock_body = Mock()
            mock_body_cls.return_value = mock_body

            result = annotation_configs_client.update_categorical(
                annotation_config=_ANNOTATION_CONFIG_ID,
                values=[mock_val],
            )

        mock_body_cls.assert_called_once()
        mock_api.update_annotation_config.assert_called_once_with(
            annotation_config_id=_ANNOTATION_CONFIG_ID,
            update_annotation_config_request=mock_body,
        )
        assert result is expected


@pytest.mark.unit
class TestAnnotationConfigsClientUpdateFreeform:
    """Tests for AnnotationConfigsClient.update_freeform()."""

    def test_update_freeform_builds_request_and_calls_api(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """update_freeform() should build the correct request and call the API."""
        expected = Mock()
        mock_api.update_annotation_config.return_value = Mock(
            actual_instance=expected
        )

        with (
            patch(
                "arize._generated.api_client.UpdateFreeformAnnotationConfigRequest"
            ) as mock_update_cls,
            patch(
                "arize._generated.api_client.UpdateAnnotationConfigRequest"
            ) as mock_body_cls,
        ):
            mock_inner = Mock()
            mock_update_cls.return_value = mock_inner
            mock_body = Mock()
            mock_body_cls.return_value = mock_body

            result = annotation_configs_client.update_freeform(
                annotation_config=_ANNOTATION_CONFIG_ID,
                name="new-name",
            )

        mock_body_cls.assert_called_once_with(actual_instance=mock_inner)
        mock_api.update_annotation_config.assert_called_once_with(
            annotation_config_id=_ANNOTATION_CONFIG_ID,
            update_annotation_config_request=mock_body,
        )
        assert result is expected


@pytest.mark.unit
class TestAnnotationConfigsClientDelete:
    """Tests for AnnotationConfigsClient.delete()."""

    def test_delete_calls_api(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """delete() should resolve config ID and call delete_annotation_config."""
        annotation_configs_client.delete(
            annotation_config=_ANNOTATION_CONFIG_ID,
        )

        mock_api.delete_annotation_config.assert_called_once_with(
            annotation_config_id=_ANNOTATION_CONFIG_ID
        )


@pytest.mark.unit
class TestAnnotationConfigsClientCreateFreeform:
    """Tests for AnnotationConfigsClient.create_freeform()."""

    def test_passes_name_and_space_id_to_api(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """create_freeform() must forward name and the resolved space_id."""
        with (
            patch(
                "arize._generated.api_client.CreateFreeformAnnotationConfigRequest"
            ) as mock_freeform_cls,
            patch("arize._generated.api_client.CreateAnnotationConfigRequest"),
        ):
            annotation_configs_client.create_freeform(
                name="feedback",
                space="U3BhY2U6OTA1MDoxSmtS",
            )

        mock_freeform_cls.assert_called_once_with(
            name="feedback",
            space_id="U3BhY2U6OTA1MDoxSmtS",
            annotation_config_type=AnnotationConfigType.FREEFORM.value,
        )

    def test_calls_api_and_returns_response(
        self, annotation_configs_client: AnnotationConfigsClient, mock_api: Mock
    ) -> None:
        """create_freeform() must call the API and unwrap the response."""
        expected = Mock()
        mock_api.create_annotation_config.return_value.actual_instance = (
            expected
        )

        with (
            patch(
                "arize._generated.api_client.CreateFreeformAnnotationConfigRequest"
            ),
            patch(
                "arize._generated.api_client.CreateAnnotationConfigRequest"
            ) as mock_body_cls,
        ):
            mock_body = Mock()
            mock_body_cls.return_value = mock_body

            result = annotation_configs_client.create_freeform(
                name="feedback",
                space="U3BhY2U6OTA1MDoxSmtS",
            )

        mock_api.create_annotation_config.assert_called_once_with(
            create_annotation_config_request=mock_body
        )
        assert result is expected
