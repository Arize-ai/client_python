# EvaluatorLlmConfigRequest

LLM configuration for an evaluator in a write request (strict form of EvaluatorLlmConfig)

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**ai_integration_id** | **str** | AI integration identifier (base64) | 
**model_name** | **str** | Model name (e.g. gpt-4o) | 
**invocation_parameters** | [**InvocationParamsRequest**](InvocationParamsRequest.md) |  | 
**provider_parameters** | [**ProviderParamsRequest**](ProviderParamsRequest.md) |  | 

## Example

```python
from arize._generated.api_client.models.evaluator_llm_config_request import EvaluatorLlmConfigRequest

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorLlmConfigRequest from a JSON string
evaluator_llm_config_request_instance = EvaluatorLlmConfigRequest.from_json(json)
# print the JSON string representation of the object
print(EvaluatorLlmConfigRequest.to_json())

# convert the object into a dict
evaluator_llm_config_request_dict = evaluator_llm_config_request_instance.to_dict()
# create an instance of EvaluatorLlmConfigRequest from a dict
evaluator_llm_config_request_from_dict = EvaluatorLlmConfigRequest.from_dict(evaluator_llm_config_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


