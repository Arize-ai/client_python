# TaskEvaluatorInput

An evaluator attachment supplied when creating or updating a task. At least one entry is required on evaluation-task requests. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**evaluator_id** | **str** | Evaluator identifier (base64). Duplicates are not allowed. | 
**query_filter** | **str** | Per-evaluator query filter. Combined with the task-level filter (AND). | [optional] 
**column_mappings** | **Dict[str, str]** | Maps evaluator template variable names to data source column names. | [optional] 

## Example

```python
from arize._generated.api_client.models.task_evaluator_input import TaskEvaluatorInput

# TODO update the JSON string below
json = "{}"
# create an instance of TaskEvaluatorInput from a JSON string
task_evaluator_input_instance = TaskEvaluatorInput.from_json(json)
# print the JSON string representation of the object
print(TaskEvaluatorInput.to_json())

# convert the object into a dict
task_evaluator_input_dict = task_evaluator_input_instance.to_dict()
# create an instance of TaskEvaluatorInput from a dict
task_evaluator_input_from_dict = TaskEvaluatorInput.from_dict(task_evaluator_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


