# TraceOrSessionEvaluatorInput

Trace/session-granularity evaluator input. Uses `query_mappings`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**evaluator_id** | **str** | Evaluator identifier (base64). Duplicates are not allowed. | 
**evaluator_version_id** | **str** | Pin this evaluator to a specific version (base64). Defaults to null, which always runs the evaluator&#39;s latest version; omitting the field and sending null are equivalent. Must be a version of the evaluator named by &#x60;evaluator_id&#x60;, otherwise the request returns 422.  | [optional] 
**query_mappings** | [**List[TaskQueryMappingInput]**](TaskQueryMappingInput.md) | Per-evaluator variable-to-query mappings (trace/session shape). | 

## Example

```python
from arize._generated.api_client.models.trace_or_session_evaluator_input import TraceOrSessionEvaluatorInput

# TODO update the JSON string below
json = "{}"
# create an instance of TraceOrSessionEvaluatorInput from a JSON string
trace_or_session_evaluator_input_instance = TraceOrSessionEvaluatorInput.from_json(json)
# print the JSON string representation of the object
print(TraceOrSessionEvaluatorInput.to_json())

# convert the object into a dict
trace_or_session_evaluator_input_dict = trace_or_session_evaluator_input_instance.to_dict()
# create an instance of TraceOrSessionEvaluatorInput from a dict
trace_or_session_evaluator_input_from_dict = TraceOrSessionEvaluatorInput.from_dict(trace_or_session_evaluator_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


