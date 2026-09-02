# SpanEvaluatorInput

Span-granularity evaluator input. Uses `query_filter` and `column_mappings`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**evaluator_id** | **str** | Evaluator identifier (base64). Duplicates are not allowed. | 
**evaluator_version_id** | **str** | Pin this evaluator to a specific version (base64). Defaults to null, which always runs the evaluator&#39;s latest version; omitting the field and sending null are equivalent. Must be a version of the evaluator named by &#x60;evaluator_id&#x60;, otherwise the request returns 422.  | [optional] 
**query_filter** | **str** | Per-evaluator query filter (span shape). Combined with the task-level filter (AND). | [optional] 
**column_mappings** | **Dict[str, str]** | Maps evaluator template variable names to data source column names (span shape). | [optional] 

## Example

```python
from arize._generated.api_client.models.span_evaluator_input import SpanEvaluatorInput

# TODO update the JSON string below
json = "{}"
# create an instance of SpanEvaluatorInput from a JSON string
span_evaluator_input_instance = SpanEvaluatorInput.from_json(json)
# print the JSON string representation of the object
print(SpanEvaluatorInput.to_json())

# convert the object into a dict
span_evaluator_input_dict = span_evaluator_input_instance.to_dict()
# create an instance of SpanEvaluatorInput from a dict
span_evaluator_input_from_dict = SpanEvaluatorInput.from_dict(span_evaluator_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


