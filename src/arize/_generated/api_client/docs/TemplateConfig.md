# TemplateConfig


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Eval column name. Must match ^[a-zA-Z0-9_\\s\\-&amp;()]+$ | 
**template** | **str** | The prompt template with variable placeholders | 
**include_explanations** | **bool** | Whether to include explanations in the evaluation output | 
**use_function_calling_if_available** | **bool** | Whether to use function calling if the model supports it | 
**classification_choices** | **Dict[str, float]** | Map of choice label to numeric score (e.g. {\&quot;relevant\&quot;: 1, \&quot;irrelevant\&quot;: 0}). When omitted, the evaluator produces freeform (non-classification) output. | [optional] 
**direction** | [**OptimizationDirection**](OptimizationDirection.md) |  | [optional] [default to OptimizationDirection.NONE]
**data_granularity** | **str** | Data granularity level. Defaults to null when omitted. | [optional] 
**llm_config** | [**EvaluatorLlmConfig**](EvaluatorLlmConfig.md) |  | 

## Example

```python
from arize._generated.api_client.models.template_config import TemplateConfig

# TODO update the JSON string below
json = "{}"
# create an instance of TemplateConfig from a JSON string
template_config_instance = TemplateConfig.from_json(json)
# print the JSON string representation of the object
print(TemplateConfig.to_json())

# convert the object into a dict
template_config_dict = template_config_instance.to_dict()
# create an instance of TemplateConfig from a dict
template_config_from_dict = TemplateConfig.from_dict(template_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


