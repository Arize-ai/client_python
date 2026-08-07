# ToolCallFunctionRequest

The function to call (strict request form of ToolCallFunction)

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | The name of the function | 
**arguments** | **str** | The arguments to the function as a JSON string | 

## Example

```python
from arize._generated.api_client.models.tool_call_function_request import ToolCallFunctionRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ToolCallFunctionRequest from a JSON string
tool_call_function_request_instance = ToolCallFunctionRequest.from_json(json)
# print the JSON string representation of the object
print(ToolCallFunctionRequest.to_json())

# convert the object into a dict
tool_call_function_request_dict = tool_call_function_request_instance.to_dict()
# create an instance of ToolCallFunctionRequest from a dict
tool_call_function_request_from_dict = ToolCallFunctionRequest.from_dict(tool_call_function_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


