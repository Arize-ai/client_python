# CreateLiteLlmConfig

Create config for a LiteLLM integration. `base_url` is required and points at the LiteLLM endpoint (validated server-side); LiteLLM is self-hosted, so there is no default endpoint. `api_key` is required: the virtual key scopes the models Arize can resolve and call. `api_key` and `headers` are write-only (never returned; headers surface as `header_names` on read).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Enable function/tool calling. Defaults to true. | [optional] 
**provider** | **str** |  | 
**base_url** | **str** | LiteLLM endpoint URL requests are sent to (HTTPS). | 
**api_key** | **str** | LiteLLM virtual key (write-only, never returned). | 
**headers** | **Dict[str, str]** | Custom request headers sent to the endpoint, as a name-to-value map. Write-only: values are never returned; names are exposed as &#x60;header_names&#x60; on read. Defaults to no headers. The serialized header map must not exceed 8,175 bytes. | [optional] 
**model_names** | **List[str]** | Custom model names to make available. Defaults to an empty list. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_lite_llm_config import CreateLiteLlmConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CreateLiteLlmConfig from a JSON string
create_lite_llm_config_instance = CreateLiteLlmConfig.from_json(json)
# print the JSON string representation of the object
print(CreateLiteLlmConfig.to_json())

# convert the object into a dict
create_lite_llm_config_dict = create_lite_llm_config_instance.to_dict()
# create an instance of CreateLiteLlmConfig from a dict
create_lite_llm_config_from_dict = CreateLiteLlmConfig.from_dict(create_lite_llm_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


