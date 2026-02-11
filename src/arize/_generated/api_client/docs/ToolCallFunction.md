# ToolCallFunction

The function to call

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | The name of the function | 
**arguments** | **str** | The arguments to the function as a JSON string | 

## Example

```python
from arize._generated.api_client.models.tool_call_function import ToolCallFunction

# TODO update the JSON string below
json = "{}"
# create an instance of ToolCallFunction from a JSON string
tool_call_function_instance = ToolCallFunction.from_json(json)
# print the JSON string representation of the object
print(ToolCallFunction.to_json())

# convert the object into a dict
tool_call_function_dict = tool_call_function_instance.to_dict()
# create an instance of ToolCallFunction from a dict
tool_call_function_from_dict = ToolCallFunction.from_dict(tool_call_function_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


