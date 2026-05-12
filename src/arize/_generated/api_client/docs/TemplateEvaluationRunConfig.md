# TemplateEvaluationRunConfig

Configuration for running a template-based LLM evaluator against each dataset example.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**experiment_type** | **str** | Discriminator. Must be &#x60;\&quot;template_evaluation\&quot;&#x60;. | 
**ai_integration_id** | **str** | AI integration global ID (base64). The LLM that judges each example. | 
**model_name** | **str** | Model name (e.g. &#x60;gpt-4o&#x60;). Falls back to the integration&#39;s default if omitted. | [optional] 
**template** | **str** | The evaluation prompt template. Use &#x60;{{variable}}&#x60; placeholders that map to dataset column paths via &#x60;column_mapping&#x60;.  | 
**provide_explanation** | **bool** | Whether to ask the LLM to include a written explanation alongside the score/label. | 
**classification_choices** | **Dict[str, float]** | Map of choice label to numeric score (e.g. &#x60;{\&quot;relevant\&quot;: 1, \&quot;irrelevant\&quot;: 0}&#x60;). | [optional] 
**column_mapping** | **Dict[str, str]** | Maps template variable names to dataset column paths. | [optional] 
**evaluator_version_id** | **str** | EvaluatorVersion global ID (base64). Links this run to an Eval Hub evaluator version. | [optional] 
**invocation_parameters** | [**InvocationParams**](InvocationParams.md) |  | [optional] 
**provider_parameters** | **object** | Provider-specific parameters. Defaults to &#x60;{}&#x60; (no overrides) if omitted. | [optional] 

## Example

```python
from arize._generated.api_client.models.template_evaluation_run_config import TemplateEvaluationRunConfig

# TODO update the JSON string below
json = "{}"
# create an instance of TemplateEvaluationRunConfig from a JSON string
template_evaluation_run_config_instance = TemplateEvaluationRunConfig.from_json(json)
# print the JSON string representation of the object
print(TemplateEvaluationRunConfig.to_json())

# convert the object into a dict
template_evaluation_run_config_dict = template_evaluation_run_config_instance.to_dict()
# create an instance of TemplateEvaluationRunConfig from a dict
template_evaluation_run_config_from_dict = TemplateEvaluationRunConfig.from_dict(template_evaluation_run_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


