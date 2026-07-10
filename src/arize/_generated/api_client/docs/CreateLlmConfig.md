# CreateLlmConfig


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Enable function/tool calling. Defaults to true. | [optional] 
**provider** | **str** |  | 
**api_key** | **str** | API key for the provider (write-only, never returned). | 

## Example

```python
from arize._generated.api_client.models.create_llm_config import CreateLlmConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CreateLlmConfig from a JSON string
create_llm_config_instance = CreateLlmConfig.from_json(json)
# print the JSON string representation of the object
print(CreateLlmConfig.to_json())

# convert the object into a dict
create_llm_config_dict = create_llm_config_instance.to_dict()
# create an instance of CreateLlmConfig from a dict
create_llm_config_from_dict = CreateLlmConfig.from_dict(create_llm_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


