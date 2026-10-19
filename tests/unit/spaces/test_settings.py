"""Tests for public space settings SDK methods."""

from unittest.mock import Mock, patch

from arize.spaces.client import SpacesClient


def test_update_forwards_space_settings(mock_sdk_config: Mock) -> None:
    with patch("arize._generated.api_client.SpacesApi") as api_cls:
        client = SpacesClient(
            sdk_config=mock_sdk_config, generated_client=Mock()
        )
        client.update(
            space="U3BhY2U6MQ==",
            default_llm_integration={
                "integration_id": "i-1",
                "model_name": "gpt-4.1",
            },
            ml_models_enabled=False,
            gradient_start_color="#3366FF",
            default_agent_runtime=None,
        )

    body = api_cls.return_value.update_space.call_args.kwargs[
        "update_space_request"
    ]
    assert body.to_dict() == {
        "default_llm_integration": {
            "integration_id": "i-1",
            "model_name": "gpt-4.1",
        },
        "ml_models_enabled": False,
        "gradient_start_color": "#3366FF",
        "default_agent_runtime": None,
    }
