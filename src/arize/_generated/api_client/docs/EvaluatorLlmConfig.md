# EvaluatorLlmConfig


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**ai_integration_id** | **str** | AI integration global ID (base64) | 
**model_name** | **str** | Model name (e.g. gpt-4o) | 
**invocation_parameters** | **Dict[str, object]** | Parameters for the LLM call (e.g. temperature, max_tokens) | 
**provider_parameters** | **Dict[str, object]** | Provider-specific parameters | 

## Example

```python
from arize._generated.api_client.models.evaluator_llm_config import EvaluatorLlmConfig

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorLlmConfig from a JSON string
evaluator_llm_config_instance = EvaluatorLlmConfig.from_json(json)
# print the JSON string representation of the object
print(EvaluatorLlmConfig.to_json())

# convert the object into a dict
evaluator_llm_config_dict = evaluator_llm_config_instance.to_dict()
# create an instance of EvaluatorLlmConfig from a dict
evaluator_llm_config_from_dict = EvaluatorLlmConfig.from_dict(evaluator_llm_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


