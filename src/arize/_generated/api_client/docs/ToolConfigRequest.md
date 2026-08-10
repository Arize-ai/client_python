# ToolConfigRequest

Tool configuration in a write request (strict form of ToolConfig)

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**tools** | **List[Dict[str, object]]** | List of tool definitions available to the model | [optional] 
**tool_choice** | **object** | Tool choice configuration | [optional] 

## Example

```python
from arize._generated.api_client.models.tool_config_request import ToolConfigRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ToolConfigRequest from a JSON string
tool_config_request_instance = ToolConfigRequest.from_json(json)
# print the JSON string representation of the object
print(ToolConfigRequest.to_json())

# convert the object into a dict
tool_config_request_dict = tool_config_request_instance.to_dict()
# create an instance of ToolConfigRequest from a dict
tool_config_request_from_dict = ToolConfigRequest.from_dict(tool_config_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


