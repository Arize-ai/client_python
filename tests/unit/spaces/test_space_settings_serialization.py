"""Tests for space settings serialization in the generated API client."""

import unittest

from arize._generated.api_client.models.default_agent_runtime import (
    DefaultAgentRuntime,
)
from arize._generated.api_client.models.default_llm_integration import (
    DefaultLLMIntegration,
)
from arize._generated.api_client.models.space import Space
from arize._generated.api_client.models.update_space_request import (
    UpdateSpaceRequest,
)


class SpaceSettingsSerializationTest(unittest.TestCase):
    def test_space_serialization_keeps_settings_in_response(self) -> None:
        space = Space.from_dict(
            {
                "id": "space-1",
                "name": "space",
                "description": "",
                "created_at": "2026-10-12T00:00:00Z",
                "is_private": False,
                "default_llm_integration": {
                    "integration_id": "integration-1",
                    "model_name": "claude-sonnet-4",
                },
                "ml_models_enabled": True,
                "gradient_start_color": "#123456",
                "gradient_end_color": "#abcdef",
                "default_agent_runtime": {"integration_id": "runtime-1"},
            }
        )

        self.assertEqual(space.to_dict()["ml_models_enabled"], True)
        self.assertEqual(space.to_dict()["gradient_start_color"], "#123456")
        self.assertEqual(space.to_dict()["gradient_end_color"], "#abcdef")

    def test_space_accepts_unknown_default_llm_integration_response_fields(
        self,
    ) -> None:
        space = Space.from_dict(
            {
                "id": "space-1",
                "name": "space",
                "description": "",
                "created_at": "2026-10-12T00:00:00Z",
                "is_private": False,
                "default_llm_integration": {
                    "integration_id": "integration-1",
                    "model_name": "claude-sonnet-4",
                    "future_field": "future-value",
                },
            }
        )

        self.assertIsNotNone(space.default_llm_integration)
        self.assertEqual(
            space.default_llm_integration.model_name,
            "claude-sonnet-4",
        )

    def test_patch_serialization_distinguishes_omitted_null_and_value(
        self,
    ) -> None:
        omitted = UpdateSpaceRequest().to_dict()
        cleared = UpdateSpaceRequest(
            default_llm_integration=None,
            default_agent_runtime=None,
        ).to_dict()
        configured = UpdateSpaceRequest(
            default_llm_integration=DefaultLLMIntegration(
                integration_id="integration-1",
                model_name="claude-sonnet-4",
            ),
            default_agent_runtime=DefaultAgentRuntime(
                integration_id="runtime-1"
            ),
        ).to_dict()

        self.assertNotIn("default_llm_integration", omitted)
        self.assertNotIn("default_agent_runtime", omitted)
        self.assertIsNone(cleared["default_llm_integration"])
        self.assertIsNone(cleared["default_agent_runtime"])
        self.assertEqual(
            configured["default_llm_integration"],
            {
                "integration_id": "integration-1",
                "model_name": "claude-sonnet-4",
            },
        )
        self.assertEqual(
            configured["default_agent_runtime"], {"integration_id": "runtime-1"}
        )

    def test_patch_preserves_omitted_fields(self) -> None:
        request = UpdateSpaceRequest(name="renamed")
        clear_request = UpdateSpaceRequest(default_llm_integration=None)

        self.assertEqual(request.to_dict(), {"name": "renamed"})
        self.assertEqual(
            clear_request.to_dict(), {"default_llm_integration": None}
        )


if __name__ == "__main__":
    unittest.main()
