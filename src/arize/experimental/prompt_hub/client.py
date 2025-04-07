from typing import Any, Dict, List, Optional

from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

from arize.experimental.prompt_hub.constants import (
    ARIZE_EXTERNAL_MODEL_MAPPING,
    ARIZE_INTERNAL_MODEL_MAPPING,
)
from arize.experimental.prompt_hub.prompts import (
    LLMProvider,
    Prompt,
    PromptInputVariableFormat,
)


def _convert_to_external_name(model: str) -> str:
    return ARIZE_INTERNAL_MODEL_MAPPING.get(model, model)


def _convert_to_internal_name(model: str) -> str:
    return ARIZE_EXTERNAL_MODEL_MAPPING.get(model, model)


class ArizePromptClient:
    """
    Client for interacting with the Arize Prompt Hub.

    This client provides methods to create, retrieve, and manage prompts in
    the Arize AI platform.

    Args:
        developer_key: The API key for authentication.
        space_id: The ID of the space to interact with.
        base_url: The base URL of the Arize API. Defaults to "https://app.arize.com".
    """

    def __init__(
        self,
        developer_key: str,
        space_id: str,
        base_url: str = "https://app.arize.com",
    ):
        self.api_key = developer_key
        self.space_id = space_id
        self.base_url = base_url
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        transport = RequestsHTTPTransport(
            url=f"{self.base_url}/graphql",
            headers=self.headers,
            use_json=True,
        )
        self.client = Client(
            transport=transport, fetch_schema_from_transport=True
        )

    def _make_request(
        self, query: str, variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make a GraphQL request to the Arize API.

        Args:
            query: The GraphQL query string.
            variables: Variables to use in the query.

        Returns:
            The response from the API.
        """
        query_obj = gql(query)
        result = self.client.execute(query_obj, variables)
        return result

    def pull_prompts(self) -> List[Prompt]:
        """
        Pull all prompts from the space.

        Returns:
            A list of prompts.
        """
        query = """
            query GetPrompts($spaceId: ID!) {
                node(id: $spaceId) {
                  ... on Space {
                    prompts(first: 50) {
                        edges {
                            node {
                                id
                                name
                                description
                                tags
                                commitMessage
                                messages
                                inputVariableFormat
                                toolChoice
                                toolCalls
                                llmParameters
                                provider
                                modelName
                            }
                        }
                    }
                  }
                }
            }
        """
        variables = {"spaceId": self.space_id}
        result = self._make_request(query, variables)
        prompts = []
        for edge in result["node"]["prompts"]["edges"]:
            node = edge["node"]
            prompts.append(
                Prompt(
                    id=node["id"],
                    name=node["name"],
                    description=node["description"],
                    tags=node["tags"],
                    commit_message=node["commitMessage"],
                    messages=node["messages"],
                    input_variable_format=PromptInputVariableFormat(
                        node["inputVariableFormat"]
                    ),
                    tool_choice=node["toolChoice"],
                    tool_calls=node["toolCalls"],
                    llm_parameters=node["llmParameters"],
                    provider=LLMProvider(node["provider"]),
                    model_name=_convert_to_external_name(node["modelName"]),
                )
            )
        return prompts

    def pull_prompt(self, prompt_name: str) -> Prompt:
        """
        Pull a prompt by name.

        Args:
            prompt_name: The name of the prompt to pull.

        Returns:
            The prompt with the given name.

        Raises:
            ValueError: If no prompt with the given name is found.
        """
        # Client-side filtering since the GraphQL endpoint doesn't support strict filtering yet
        prompts = self.pull_prompts()
        for prompt in prompts:
            if prompt.name == prompt_name:
                return prompt
        raise ValueError(f"Prompt with name {prompt_name} not found")

    def push_prompt(
        self, prompt: Prompt, commit_message: Optional[str] = None
    ) -> None:
        """
        Push a prompt to the Arize Prompt Hub.
        """
        if prompt.id is None:
            return self._create_prompt(prompt)
        return self._edit_prompt(prompt, commit_message)

    def _edit_prompt(
        self, prompt: Prompt, commit_message: Optional[str] = None
    ) -> None:
        """
        Push a prompt to the Arize Prompt Hub.

        Args:
            prompt: The prompt to push.
            commit_message: Optional commit message to override the one in the prompt.
        """
        query = """
            mutation CreatePromptVersion(
                $spaceId: ID!,
                $promptId: ID!,
                $commitMessage: String!,
                $messages: [JSON!]!,
                $inputVariableFormat: PromptVersionInputVariableFormatEnum!,
                $toolChoice: String,
                $toolCalls: [JSON!],
                $llmParameters: JSON!,
                $provider: LLMIntegrationProvider!,
                $modelName: ExternalLLMProviderModel
            ) {
                createPromptVersion(input: {
                    spaceId: $spaceId,
                    promptId: $promptId,
                    commitMessage: $commitMessage,
                    messages: $messages,
                    inputVariableFormat: $inputVariableFormat,
                    toolChoice: $toolChoice,
                    toolCalls: $toolCalls,
                    llmParameters: $llmParameters,
                    provider: $provider,
                    modelName: $modelName
                }) {
                    promptVersion {
                        id
                        commitMessage
                        messages
                        inputVariableFormat
                        toolChoice
                        toolCalls
                        llmParameters
                        provider
                        modelName
                        createdAt
                    }
                }
            }
        """
        variables = {
            "spaceId": self.space_id,
            "promptId": prompt.id,
            "commitMessage": commit_message
            if commit_message is not None
            else "pushed via ArizePromptClient",
            "messages": prompt.messages,
            "inputVariableFormat": prompt.input_variable_format.value,
            "toolChoice": prompt.tool_choice,
            "toolCalls": prompt.tool_calls,
            "llmParameters": prompt.llm_parameters,
            "provider": prompt.provider.value,
            "modelName": _convert_to_internal_name(prompt.model_name),
        }
        self._make_request(query, variables)
        return prompt

    def _create_prompt(self, prompt: Prompt) -> Prompt:
        query = """
            mutation CreatePrompt(
                $spaceId: ID!,
                $name: String!,
                $description: String!,
                $tags: [String!]!,
                $commitMessage: String!,
                $messages: [JSON!]!,
                $inputVariableFormat: PromptVersionInputVariableFormatEnum!,
                $llmParameters: JSON!,
                $provider: LLMIntegrationProvider!,
                $modelName: ExternalLLMProviderModel
            ) {
                createPrompt(input: {
                    spaceId: $spaceId,
                    name: $name,
                    description: $description,
                    tags: $tags,
                    commitMessage: $commitMessage,
                    messages: $messages,
                    inputVariableFormat: $inputVariableFormat,
                    llmParameters: $llmParameters,
                    provider: $provider,
                    modelName: $modelName
                }) {
                    prompt {
                        id
                        name
                        description
                        tags
                        commitMessage
                        messages
                        inputVariableFormat
                        toolChoice
                        toolCalls
                        llmParameters
                        provider
                        modelName
                    }
                }
            }
        """
        normalized_model_name = _convert_to_internal_name(prompt.model_name)
        variables = {
            "spaceId": self.space_id,
            "name": prompt.name,
            "description": prompt.description or "",
            "tags": prompt.tags or [],
            "commitMessage": prompt.commit_message or "Initial prompt creation",
            "messages": prompt.messages,
            "inputVariableFormat": prompt.input_variable_format.value,
            "llmParameters": prompt.llm_parameters,
            "provider": prompt.provider.value,
            "modelName": normalized_model_name,
        }
        result = self._make_request(query, variables)
        prompt_data = result["createPrompt"]["prompt"]
        prompt.id = prompt_data["id"]
        return prompt
