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
**response_format** | [**ResponseFormat**](ResponseFormat.md) | Response format configuration. Optional. When omitted, no structured output constraint is applied (the provider&#39;s default plain-text behavior is used). | [optional] 
**tool_config** | [**ToolConfig**](ToolConfig.md) | Tool configuration for the LLM invocation. Optional. When omitted, no tools are made available to the model. | [optional] 
**top_k** | **int** | Top-K sampling parameter. A top-K of 1 means the next selected token is the most probable (greedy decoding). | [optional] 
**thinking_level** | **str** | Controls how much reasoning the model performs before responding. Supported by Gemini 3.x models. Accepted values: &#39;low&#39;, &#39;high&#39;. | [optional] 
**thinking_budget** | **int** | Maximum tokens the model may use for internal reasoning. Supported by Gemini 2.5 models. Range: 0-24576 (Flash/Flash-Lite) or 128-32768 (Pro). Set 0 to disable thinking on Flash models. | [optional] 
**reasoning_effort** | **str** | Controls how much reasoning the model performs before responding. Supported by OpenAI o-series and GPT-5 models. o-series: &#39;low&#39; | &#39;medium&#39; | &#39;high&#39;. GPT-5: &#39;none&#39; | &#39;low&#39; | &#39;medium&#39; | &#39;high&#39; | &#39;xhigh&#39;. | [optional] 
**verbosity** | **str** | Controls the verbosity of model output. Supported by OpenAI GPT-5 series. Accepted values: &#39;low&#39; | &#39;medium&#39; | &#39;high&#39;. | [optional] 

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


