# BaseEvaluationTaskRequestEvaluatorsInner


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**evaluator_id** | **str** | Evaluator global ID (base64). Duplicates are not allowed. | 
**query_filter** | **str** | Per-evaluator query filter. Combined with the task-level filter (AND). | [optional] 
**column_mappings** | **Dict[str, str]** | Maps evaluator template variable names to data source column names. | [optional] 

## Example

```python
from arize._generated.api_client.models.base_evaluation_task_request_evaluators_inner import BaseEvaluationTaskRequestEvaluatorsInner

# TODO update the JSON string below
json = "{}"
# create an instance of BaseEvaluationTaskRequestEvaluatorsInner from a JSON string
base_evaluation_task_request_evaluators_inner_instance = BaseEvaluationTaskRequestEvaluatorsInner.from_json(json)
# print the JSON string representation of the object
print(BaseEvaluationTaskRequestEvaluatorsInner.to_json())

# convert the object into a dict
base_evaluation_task_request_evaluators_inner_dict = base_evaluation_task_request_evaluators_inner_instance.to_dict()
# create an instance of BaseEvaluationTaskRequestEvaluatorsInner from a dict
base_evaluation_task_request_evaluators_inner_from_dict = BaseEvaluationTaskRequestEvaluatorsInner.from_dict(base_evaluation_task_request_evaluators_inner_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


