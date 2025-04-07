"""
OpenAI-specific prompt implementations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from arize.experimental.prompt_hub.prompts.base import FormattedPrompt, Prompt


@dataclass(frozen=True)
class OpenAIPrompt(FormattedPrompt):
    """
    OpenAI-specific formatted prompt.

    Contains fully formatted messages and additional parameters (e.g. model, temperature, etc.)
    required by the OpenAI API.
    """

    messages: Sequence[Dict[str, str]]
    kwargs: Dict[str, Any]


def to_openai_prompt(
    prompt: Prompt, variables: Mapping[str, Any]
) -> OpenAIPrompt:
    """
    Convert a Prompt to an OpenAI-specific formatted prompt.

    Args:
        prompt: The prompt to format.
        variables: A mapping of variable names to values.

    Returns:
        An OpenAI-specific formatted prompt.
    """
    return OpenAIPrompt(
        messages=format_openai_prompt(prompt, variables),
        kwargs=openai_kwargs(prompt),
    )


def format_openai_prompt(
    prompt: Prompt, variables: Mapping[str, Any]
) -> List[Dict[str, str]]:
    """
    Format a Prompt's messages for the OpenAI API.

    Args:
        prompt: The prompt to format.
        variables: A mapping of variable names to values.

    Returns:
        A list of formatted message dictionaries.
    """
    formatted_messages = []
    for message in prompt.messages:
        formatted_message = message.copy()
        formatted_message["content"] = message["content"].format(**variables)
        formatted_messages.append(formatted_message)
    return formatted_messages


def openai_kwargs(prompt: Prompt) -> Dict[str, Any]:
    """
    Generate kwargs for the OpenAI API based on the prompt.

    Args:
        prompt: The prompt to generate kwargs for.

    Returns:
        A dictionary of kwargs for the OpenAI API.
    """
    return {"model": prompt.model_name}
