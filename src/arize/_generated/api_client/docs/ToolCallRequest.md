# ToolCallRequest

A tool call in a prompt write request (strict request form of ToolCall)

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The ID of the tool call | [optional] 
**type** | [**ToolCallType**](ToolCallType.md) |  | 
**function** | [**ToolCallFunctionRequest**](ToolCallFunctionRequest.md) |  | 

## Example

```python
from arize._generated.api_client.models.tool_call_request import ToolCallRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ToolCallRequest from a JSON string
tool_call_request_instance = ToolCallRequest.from_json(json)
# print the JSON string representation of the object
print(ToolCallRequest.to_json())

# convert the object into a dict
tool_call_request_dict = tool_call_request_instance.to_dict()
# create an instance of ToolCallRequest from a dict
tool_call_request_from_dict = ToolCallRequest.from_dict(tool_call_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


