# RunConfiguration

Experiment execution configuration for a `run_experiment` task. Exactly one variant must be supplied, identified by `experiment_type`. All fields sit at the top level alongside `experiment_type` (flat — no wrapper sub-object). 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**experiment_type** | **str** | Discriminator. Must be &#x60;\&quot;template_evaluation\&quot;&#x60;. | 
**ai_integration_id** | **str** | AI integration global ID (base64). The LLM that judges each example. | 
**model_name** | **str** | Model name (e.g. &#x60;gpt-4o&#x60;). Falls back to the integration&#39;s default if omitted. | [optional] 
**messages** | [**List[LLMMessage]**](LLMMessage.md) | Array of message objects (at least one). | 
**input_variable_format** | [**InputVariableFormat**](InputVariableFormat.md) |  | 
**invocation_parameters** | [**InvocationParams**](InvocationParams.md) |  | [optional] 
**provider_parameters** | **object** | Provider-specific parameters. Defaults to &#x60;{}&#x60; (no overrides) if omitted. | [optional] 
**tool_config** | [**ToolConfig**](ToolConfig.md) |  | [optional] 
**prompt_version_id** | **str** | Prompt version global ID (base64). Links to a Prompt Hub version for traceability. | [optional] 
**template** | **str** | The evaluation prompt template. Use &#x60;{{variable}}&#x60; placeholders that map to dataset column paths via &#x60;column_mapping&#x60;.  | 
**provide_explanation** | **bool** | Whether to ask the LLM to include a written explanation alongside the score/label. | 
**classification_choices** | **Dict[str, float]** | Map of choice label to numeric score (e.g. &#x60;{\&quot;relevant\&quot;: 1, \&quot;irrelevant\&quot;: 0}&#x60;). | [optional] 
**column_mapping** | **Dict[str, str]** | Maps template variable names to dataset column paths. | [optional] 
**evaluator_version_id** | **str** | EvaluatorVersion global ID (base64). Links this run to an Eval Hub evaluator version. | [optional] 

## Example

```python
from arize._generated.api_client.models.run_configuration import RunConfiguration

# TODO update the JSON string below
json = "{}"
# create an instance of RunConfiguration from a JSON string
run_configuration_instance = RunConfiguration.from_json(json)
# print the JSON string representation of the object
print(RunConfiguration.to_json())

# convert the object into a dict
run_configuration_dict = run_configuration_instance.to_dict()
# create an instance of RunConfiguration from a dict
run_configuration_from_dict = RunConfiguration.from_dict(run_configuration_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


