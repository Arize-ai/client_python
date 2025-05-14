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
from arize.experimental.prompt_hub.prompts.base import prompt_to_llm_input
from arize.utils.logging import logger


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
        space_id: The ID of the space to interact with.
        api_key: The API key for authentication.
        developer_key: (Deprecated) Use api_key instead.
        base_url: The base URL of the Arize API. Defaults to "https://app.arize.com".
    """

    def __init__(
        self,
        space_id: str,
        api_key: Optional[str] = None,
        developer_key: Optional[str] = None,
        base_url: str = "https://app.arize.com",
    ):
        if api_key is not None:
            self.api_key = api_key
        elif developer_key is not None:
            logger.warning(
                "The 'developer_key' parameter is deprecated and will be removed in a future release. "
                "Please use 'api_key' instead."
            )
            self.api_key = developer_key
        else:
            raise ValueError(
                "You must provide 'api_key'(preferred) or 'developer_key'(deprecated)."
            )

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
    ) -> Prompt:
        """
        Push a prompt to the Arize Prompt Hub.

        Returns:
            The updated prompt.
        """
        if prompt.id is None:
            return self._create_prompt(prompt)
        return self._edit_prompt(prompt, commit_message)

    def _edit_prompt(
        self, prompt: Prompt, commit_message: Optional[str] = None
    ) -> Prompt:
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
                $inputVariableFormat: PromptVersionInputVariableFormatEnum!,
                $provider: ExternalLLMProvider!,
                $model: String,
                $messages: [LLMMessageInput!]!,
                $invocationParams: InvocationParamsInput!,
                $providerParams: ProviderParamsInput!,
            ) {
                createPromptVersion(input: {
                    spaceId: $spaceId,
                    promptId: $promptId,
                    commitMessage: $commitMessage,
                    inputVariableFormat: $inputVariableFormat,
                    provider: $provider,
                    model: $model,
                    messages: $messages,
                    invocationParams: $invocationParams,
                    providerParams: $providerParams,
                }) {
                    promptVersion {
                        id
                    }
                }
            }
        """

        llm_input = prompt_to_llm_input(prompt)
        variables = {
            "spaceId": self.space_id,
            "promptId": prompt.id,
            "commitMessage": (
                commit_message
                if commit_message is not None
                else "pushed via ArizePromptClient"
            ),
            "inputVariableFormat": prompt.input_variable_format.value,
            "provider": llm_input.provider.value,
            "model": _convert_to_internal_name(prompt.model_name),
            "messages": [
                msg.model_dump(exclude_none=True) for msg in llm_input.messages
            ],
            "invocationParams": llm_input.invocationParams.model_dump(
                exclude_none=True
            ),
            "providerParams": llm_input.providerParams.model_dump(
                exclude_none=True
            ),
        }
        self._make_request(query, variables)
        return prompt

    def _create_prompt(self, prompt: Prompt) -> Prompt:
        query = """
            mutation CreatePrompt(
                $spaceId: ID!,
                $name: String!,
                $description: String,
                $tags: [String!],
                $commitMessage: String!,
                $inputVariableFormat: PromptVersionInputVariableFormatEnum!,
                $provider: ExternalLLMProvider!,
                $model: String,
                $messages: [LLMMessageInput!]!,
                $invocationParams: InvocationParamsInput!,
                $providerParams: ProviderParamsInput!,
            ) {
                createPrompt(input: {
                    spaceId: $spaceId,
                    name: $name,
                    description: $description,
                    tags: $tags,
                    commitMessage: $commitMessage,
                    inputVariableFormat: $inputVariableFormat,
                    provider: $provider,
                    model: $model,
                    messages: $messages,
                    invocationParams: $invocationParams,
                    providerParams: $providerParams,
                }) {
                    prompt {
                        id
                        name
                    }
                }
            }
        """
        normalized_model_name = _convert_to_internal_name(prompt.model_name)
        llm_input = prompt_to_llm_input(prompt)

        variables = {
            "spaceId": self.space_id,
            "name": prompt.name,
            "description": prompt.description or "",
            "tags": prompt.tags or [],
            "commitMessage": prompt.commit_message or "Initial prompt creation",
            "inputVariableFormat": prompt.input_variable_format.value,
            "provider": llm_input.provider.value,
            "model": normalized_model_name,
            "messages": [
                msg.model_dump(exclude_none=True) for msg in llm_input.messages
            ],
            "invocationParams": llm_input.invocationParams.model_dump(
                exclude_none=True
            ),
            "providerParams": llm_input.providerParams.model_dump(
                exclude_none=True
            ),
        }

        result = self._make_request(query, variables)
        prompt_data = result["createPrompt"]["prompt"]
        prompt.id = prompt_data["id"]
        return prompt
