# CreateLlmConfigBase


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Enable function/tool calling. Defaults to true. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_llm_config_base import CreateLlmConfigBase

# TODO update the JSON string below
json = "{}"
# create an instance of CreateLlmConfigBase from a JSON string
create_llm_config_base_instance = CreateLlmConfigBase.from_json(json)
# print the JSON string representation of the object
print(CreateLlmConfigBase.to_json())

# convert the object into a dict
create_llm_config_base_dict = create_llm_config_base_instance.to_dict()
# create an instance of CreateLlmConfigBase from a dict
create_llm_config_base_from_dict = CreateLlmConfigBase.from_dict(create_llm_config_base_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


