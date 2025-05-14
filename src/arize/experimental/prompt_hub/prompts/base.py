"""
Base classes for formatted prompts.
"""

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field


class FormattedPrompt(ABC, Mapping[str, Any]):
    """
    Base class for formatted prompts that can be unpacked as kwargs
    to plug into LLM provider client libraries.

    This abstraction allows provider-specific formatted prompts to be
    used directly with their respective client libraries.
    """

    messages: Sequence[Any]
    kwargs: Mapping[str, Any]

    def __len__(self) -> int:
        return 1 + len(self.kwargs)

    def __iter__(self):
        yield "messages"
        yield from self.kwargs

    def __getitem__(self, key: str) -> Any:
        if key == "messages":
            return self.messages
        return self.kwargs[key]


class PromptInputVariableFormat(str, Enum):
    """Enum for specifying how input variables are formatted in prompts."""

    F_STRING = (
        "F_STRING"  # Represented by single curly braces ({variable_name})
    )
    MUSTACHE = (
        "MUSTACHE"  # Represented by double curly braces ({{variable_name}})
    )


class LLMProvider(str, Enum):
    """Enum for supported LLM providers."""

    OPENAI = "openAI"
    AZURE_OPENAI = "azureOpenAI"
    AWS_BEDROCK = "awsBedrock"
    VERTEX_AI = "vertexAI"
    CUSTOM = "custom"


@dataclass
class Prompt:
    """
    Represents a prompt template with associated metadata and formatting options.

    This class stores the prompt structure and provides methods to format it
    for specific providers and variables.
    """

    name: str
    messages: List[Dict[str, Any]]
    provider: LLMProvider
    model_name: str
    id: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    commit_message: Optional[str] = None
    input_variable_format: PromptInputVariableFormat = (
        PromptInputVariableFormat.F_STRING
    )
    tool_choice: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    llm_parameters: Dict[str, Any] = field(default_factory=dict)

    def format(self, variables: Mapping[str, Any]) -> FormattedPrompt:
        """
        Format the prompt with the given variables.

        Args:
            variables: A mapping of variable names to values.

        Returns:
            A provider-specific formatted prompt that can be unpacked directly into API calls.
            For example:
            ```python
            prompt_vars = {"topic": "Sports", "article": "..."}
            formatted_prompt = prompt.format(variables=prompt_vars)

            oai_client = OpenAI()
            resp = oai_client.chat.completions.create(**formatted_prompt)
            ```

        Raises:
            NotImplementedError: If formatting for the provider is not implemented.
        """
        if self.provider == LLMProvider.OPENAI:
            from arize.experimental.prompt_hub.prompts.open_ai import (
                to_openai_prompt,
            )

            return to_openai_prompt(self, variables)
        elif self.provider == LLMProvider.VERTEX_AI:
            try:
                from .vertex_ai import to_vertexai_prompt

                return to_vertexai_prompt(self, variables)
            except ImportError as err:
                raise ImportError(
                    "VertexAI support requires additional dependencies. "
                    "Install them with: pip install arize[PromptHub_VertexAI]"
                ) from err
        else:
            raise NotImplementedError(
                f"Formatting for provider {self.provider} is not implemented"
            )


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class GeneratedToolInput(BaseModel):
    id: str
    type: str
    function: Dict[str, Any]


class LLMMessageInput(BaseModel):
    role: MessageRole
    content: Optional[str] = None
    imageUrls: Optional[List[str]] = None
    toolCalls: Optional[List[GeneratedToolInput]] = None
    toolCallId: Optional[str] = None

    class Config:
        exclude_none = True


class ToolConfig(BaseModel):
    tools: Optional[List[Dict[str, Any]]] = None
    toolChoice: Optional[str] = None


class InvocationParamsInput(BaseModel):
    temperature: Optional[float] = None
    topP: Optional[float] = None
    maxTokens: Optional[int] = None
    maxCompletionTokens: Optional[int] = None
    stop: Optional[List[str]] = None
    presencePenalty: Optional[float] = None
    frequencyPenalty: Optional[float] = None
    topK: Optional[int] = None
    toolConfig: Optional[ToolConfig] = None

    class Config:
        exclude_none = True


class CustomProviderParams(BaseModel):
    customModelEndpoint: Dict[str, Any]
    customProviderAPIVersion: Optional[str] = None


class AzureOpenAIParams(BaseModel):
    azureDeploymentName: str
    azureOpenAIEndpoint: str
    azureOpenAIVersion: Optional[str] = None


class AnthropicHeaders(BaseModel):
    anthropicBeta: Optional[List[str]] = None


class BedrockOptions(BaseModel):
    useConverseEndpoint: Optional[bool] = True


class ProviderParamsInput(BaseModel):
    azureParams: Optional[AzureOpenAIParams] = None
    anthropicHeaders: Optional[AnthropicHeaders] = None
    customProviderParams: Optional[CustomProviderParams] = None
    anthropicVersion: Optional[str] = None
    region: Optional[str] = None
    bedrockOptions: Optional[BedrockOptions] = None

    class Config:
        exclude_none = True


class LLMInput(BaseModel):
    invocationParams: InvocationParamsInput = Field(
        default_factory=InvocationParamsInput
    )
    messages: List[LLMMessageInput] = Field(default_factory=list)
    model: Optional[str] = None
    provider: LLMProvider = LLMProvider.OPENAI
    providerParams: ProviderParamsInput = Field(
        default_factory=ProviderParamsInput
    )

    class Config:
        exclude_none = True


def prompt_to_llm_input(prompt: Prompt) -> LLMInput:
    """
    Convert a Prompt object to the LLMInput format required by the API.
    """
    messages = []
    for msg in prompt.messages:
        message = LLMMessageInput(
            role=MessageRole(msg["role"]),
            content=msg.get("content"),
            imageUrls=msg.get("imageUrls"),
            toolCalls=msg.get("toolCalls"),
            toolCallId=msg.get("toolCallId"),
        )
        messages.append(message)

    # Extract LLM parameters
    invocation_params = InvocationParamsInput(
        temperature=prompt.llm_parameters.get("temperature"),
        topP=prompt.llm_parameters.get("topP"),
        maxTokens=prompt.llm_parameters.get("maxTokens"),
        maxCompletionTokens=prompt.llm_parameters.get("maxCompletionTokens"),
        stop=prompt.llm_parameters.get("stop"),
        presencePenalty=prompt.llm_parameters.get("presencePenalty"),
        frequencyPenalty=prompt.llm_parameters.get("frequencyPenalty"),
        topK=prompt.llm_parameters.get("topK"),
        toolConfig=(
            ToolConfig(
                tools=prompt.llm_parameters.get("tools"),
                toolChoice=prompt.llm_parameters.get("toolChoice"),
            )
            if prompt.llm_parameters.get("tools")
            or prompt.llm_parameters.get("toolChoice")
            else None
        ),
    )

    provider_params = ProviderParamsInput(
        customProviderParams=(
            CustomProviderParams(
                customModelEndpoint=prompt.llm_parameters.get(
                    "customModelEndpoint", {}
                ),
                customProviderAPIVersion=prompt.llm_parameters.get(
                    "customProviderAPIVersion"
                ),
            )
            if prompt.llm_parameters.get("customModelEndpoint")
            else None
        ),
        anthropicVersion=prompt.llm_parameters.get("anthropicVersion"),
        region=prompt.llm_parameters.get("region"),
    )
    result = LLMInput.model_validate(
        {
            "invocationParams": invocation_params,
            "messages": messages,
            "model": prompt.model_name,
            "provider": LLMProvider(prompt.provider.value),
            "providerParams": provider_params,
        }
    )
    return result
