"""
Vertex AI-specific prompt implementations.
"""

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence

from vertexai.generative_models import Content, Part

from arize.experimental.prompt_hub.prompts.base import FormattedPrompt, Prompt


@dataclass(frozen=True)
class VertexAIPrompt(FormattedPrompt):
    """
    Vertex AI-specific formatted prompt.

    Contains fully formatted messages and additional parameters (e.g. model, temperature, etc.)
    required by the Vertex AI API.
    """

    messages: Sequence[Content]
    model_name: str


def to_vertexai_prompt(
    prompt: Prompt, variables: Mapping[str, Any]
) -> VertexAIPrompt:
    """
    Convert a Prompt to a Vertex AI-specific formatted prompt.

    Args:
        prompt: The prompt to format.
        variables: A mapping of variable names to values.

    Returns:
        A Vertex AI-specific formatted prompt.

    Raises:
        ImportError: If Vertex AI dependencies are not available.
    """
    return VertexAIPrompt(
        messages=format_vertexai_prompt(prompt, variables),
        model_name=prompt.model_name,
    )


def format_vertexai_prompt(
    prompt: Prompt, variables: Mapping[str, Any]
) -> List[Content]:
    """
    Format a Prompt's messages for the Vertex AI API.

    Args:
        prompt: The prompt to format.
        variables: A mapping of variable names to values.

    Returns:
        A list of formatted Content objects.

    Raises:
        ImportError: If Vertex AI dependencies are not available.
    """
    formatted_messages = []
    for message in prompt.messages:
        formatted_message = Content(
            role=message["role"],
            parts=[Part.from_text(message["content"].format(**variables))],
        )
        formatted_messages.append(formatted_message)
    return formatted_messages
