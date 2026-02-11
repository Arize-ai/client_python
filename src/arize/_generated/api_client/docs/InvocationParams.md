# InvocationParams

Parameters for the LLM invocation

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**temperature** | **float** | Sampling temperature (higher &#x3D; more random) | [optional] 
**max_tokens** | **int** | Maximum number of tokens to generate | [optional] 
**max_completion_tokens** | **int** | Maximum number of completion tokens to generate | [optional] 
**top_p** | **float** | Nucleus sampling parameter | [optional] 
**frequency_penalty** | **float** | Frequency penalty (-2.0 to 2.0) | [optional] 
**presence_penalty** | **float** | Presence penalty (-2.0 to 2.0) | [optional] 
**stop** | **List[str]** | Stop sequences | [optional] 
**response_format** | [**ResponseFormat**](ResponseFormat.md) |  | [optional] 
**tool_config** | [**ToolConfig**](ToolConfig.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.invocation_params import InvocationParams

# TODO update the JSON string below
json = "{}"
# create an instance of InvocationParams from a JSON string
invocation_params_instance = InvocationParams.from_json(json)
# print the JSON string representation of the object
print(InvocationParams.to_json())

# convert the object into a dict
invocation_params_dict = invocation_params_instance.to_dict()
# create an instance of InvocationParams from a dict
invocation_params_from_dict = InvocationParams.from_dict(invocation_params_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


