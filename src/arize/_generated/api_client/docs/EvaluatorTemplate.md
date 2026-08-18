# EvaluatorTemplate

A built-in LLM-as-a-judge evaluator template. Templates are the same catalog offered in the product's create-evaluator flow, and are identical for every caller. They carry no space, account, or user data.  A template is a starting point for an evaluator. To create one from it, map its fields onto `POST /v2/evaluators`. See the field-by-field mapping and a complete example on `GET /v2/evaluator-templates`. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**column_name** | **str** | Stable identifier for the template, and the eval column name it writes to by default (e.g. &#x60;hallucination&#x60;). Unique across all templates.  | 
**display_name** | **str** | Human-readable name shown in the product. | 
**template** | **str** | The judge prompt. Variables are single-brace, f-string style (e.g. &#x60;{input}&#x60;, &#x60;{output}&#x60;, &#x60;{context}&#x60;) and are bound to real data by a task&#39;s column mappings when the evaluator runs.  This is the only prompt you need. To have the judge explain its label, set &#x60;include_explanations&#x60; on &#x60;POST /v2/evaluators&#x60;. The explanation request is added at run time, not by editing this prompt.  | 
**rails** | **List[str]** | The labels the judge is allowed to return, in the order the product displays them.  | 
**classification_choices** | **Dict[str, float]** | Maps each label to its numeric score. Pass this through unchanged when creating an evaluator, since the labels must match those named in the template.  | 
**direction** | [**OptimizationDirection**](OptimizationDirection.md) | Whether a higher score is better (&#x60;MAXIMIZE&#x60;), worse (&#x60;MINIMIZE&#x60;), or neither (&#x60;NONE&#x60;). Controls how trends are rendered. Pass it through unchanged, since it must agree with &#x60;classification_choices&#x60;. If the two disagree, the product renders the trend backwards.  | 
**data_granularity** | [**DataGranularity**](DataGranularity.md) | The unit this template evaluates. &#x60;null&#x60; means span level, which is the default for most response-quality, RAG, and security templates. &#x60;SESSION&#x60; templates score a whole conversation and require spans that carry a session identifier.  | 

## Example

```python
from arize._generated.api_client.models.evaluator_template import EvaluatorTemplate

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorTemplate from a JSON string
evaluator_template_instance = EvaluatorTemplate.from_json(json)
# print the JSON string representation of the object
print(EvaluatorTemplate.to_json())

# convert the object into a dict
evaluator_template_dict = evaluator_template_instance.to_dict()
# create an instance of EvaluatorTemplate from a dict
evaluator_template_from_dict = EvaluatorTemplate.from_dict(evaluator_template_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


