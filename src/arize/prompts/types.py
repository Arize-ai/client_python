"""Public type re-exports and SDK-facing prompt types for the prompts subdomain."""

from __future__ import annotations

from enum import Enum

from arize._generated.api_client.models.input_variable_format import (
    InputVariableFormat,
)
from arize._generated.api_client.models.invocation_params import (
    InvocationParams as _GenInvocationParams,
)
from arize._generated.api_client.models.llm_message import LLMMessage
from arize._generated.api_client.models.llm_provider import LlmProvider
from arize._generated.api_client.models.message_role import MessageRole
from arize._generated.api_client.models.prompt import Prompt
from arize._generated.api_client.models.prompt_list_response import (
    PromptListResponse,
)
from arize._generated.api_client.models.prompt_version import (
    PromptVersion as _GenPromptVersion,
)
from arize._generated.api_client.models.prompt_version_labels_response import (
    PromptVersionLabelsResponse,
)
from arize._generated.api_client.models.prompt_version_list_response import (
    PromptVersionListResponse,
)
from arize._generated.api_client.models.prompt_with_version import (
    PromptWithVersion as _GenPromptWithVersion,
)
from arize._generated.api_client.models.provider_params import ProviderParams
from arize._generated.api_client.models.tool_call_type import ToolCallType


class InvocationParams(_GenInvocationParams):
    """Invocation parameters with a clean string representation.

    Omits fields whose value is ``None`` so that an all-None object
    displays as an empty string rather than leaking internal repr.
    """

    def __str__(self) -> str:
        """Return non-None fields as ``key=value`` pairs, or empty string."""
        parts = [
            f"{name}={getattr(self, name)}"
            for name in type(self).model_fields
            if name != "additional_properties"
            and getattr(self, name) is not None
        ]
        return ", ".join(parts)


class PromptVersion(_GenPromptVersion):
    """SDK view of a prompt version with a clean string representation.

    Used when the version is displayed as a nested field inside
    :class:`PromptWithVersion` — avoids leaking internal SDK repr strings
    for enum values, messages, and invocation params.
    """

    def __str__(self) -> str:
        """Return a human-readable summary of the version."""
        provider_str = (
            self.provider.value
            if isinstance(self.provider, Enum)
            else str(self.provider)
        )
        ivf_str = (
            self.input_variable_format.value
            if isinstance(self.input_variable_format, Enum)
            else str(self.input_variable_format)
        )

        msg_lines = []
        for msg in self.messages or []:
            role_str = (
                msg.role.value if isinstance(msg.role, Enum) else str(msg.role)
            )
            if msg.content is not None:
                msg_lines.append(f"{role_str}: {msg.content}")
            elif msg.tool_calls:
                for tc in msg.tool_calls:
                    fn = getattr(tc, "function", None)
                    fn_name = getattr(fn, "name", "?") if fn else "?"
                    msg_lines.append(f"{role_str}: \u2192 {fn_name}()")
            elif msg.tool_call_id:
                msg_lines.append(
                    f"{role_str}: (tool response for {msg.tool_call_id})"
                )
            else:
                msg_lines.append(f"{role_str}:")

        labels_str = (
            f"  labels: {', '.join(self.labels)}\n" if self.labels else ""
        )
        params_str = ""
        if self.invocation_params is not None and str(self.invocation_params):
            params_str = f"\n  params: {self.invocation_params}"

        msgs_block = "\n  ".join(msg_lines) if msg_lines else "(no messages)"
        return (
            f"commit: {self.commit_message}\n"
            f"{labels_str}"
            f"  {self.model} ({provider_str}) | {ivf_str}{params_str}\n"
            f"  {msgs_block}"
        )


class PromptWithVersion(_GenPromptWithVersion):
    """SDK view of a prompt with its resolved version.

    Overrides the ``version`` field type to :class:`PromptVersion` so that
    the nested version object renders cleanly via ``__str__`` when displayed
    by the CLI's generic formatter.
    """

    version: PromptVersion  # type: ignore[assignment]


__all__ = [
    "InputVariableFormat",
    "InvocationParams",
    "LLMMessage",
    "LlmProvider",
    "MessageRole",
    "Prompt",
    "PromptListResponse",
    "PromptVersion",
    "PromptVersionLabelsResponse",
    "PromptVersionListResponse",
    "PromptWithVersion",
    "ProviderParams",
    "ToolCallType",
]
