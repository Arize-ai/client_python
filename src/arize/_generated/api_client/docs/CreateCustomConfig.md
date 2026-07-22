# CreateCustomConfig

Create config for a custom OpenAI-compatible endpoint integration. `base_url` is required and must implement the OpenAI API shape (it is validated server-side and must resolve to a public address). `api_key` and `headers` are write-only (never returned; headers surface as `header_names` on read). The integration must have at least one model source: enable `is_default_models_enabled` or provide at least one entry in `model_names`, otherwise the request is rejected with 422.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Enable function/tool calling. Defaults to true. | [optional] 
**provider** | **str** |  | 
**base_url** | **str** | Endpoint URL requests are sent to (HTTPS). | 
**api_key** | **str** | API key for the endpoint (write-only, never returned). | [optional] 
**headers** | **Dict[str, str]** | Custom request headers sent to the endpoint, as a name-to-value map. Write-only: values are never returned; names are exposed as &#x60;header_names&#x60; on read. Defaults to no headers. | [optional] 
**is_default_models_enabled** | **bool** | Enable Arize&#39;s default model catalog. Defaults to false. | [optional] 
**model_names** | **List[str]** | Custom model names to make available. Defaults to none. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_custom_config import CreateCustomConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CreateCustomConfig from a JSON string
create_custom_config_instance = CreateCustomConfig.from_json(json)
# print the JSON string representation of the object
print(CreateCustomConfig.to_json())

# convert the object into a dict
create_custom_config_dict = create_custom_config_instance.to_dict()
# create an instance of CreateCustomConfig from a dict
create_custom_config_from_dict = CreateCustomConfig.from_dict(create_custom_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


