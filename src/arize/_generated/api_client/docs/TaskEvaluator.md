# TaskEvaluator


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**evaluator_id** | **str** | Evaluator global ID (base64). | 
**evaluator_name** | **str** | The name of the attached evaluator. | 
**query_filter** | **str** | Per-evaluator query filter, combined with the task-level filter (AND). | 
**column_mappings** | **Dict[str, str]** | Maps evaluator template variable names to data source column names. | 

## Example

```python
from arize._generated.api_client.models.task_evaluator import TaskEvaluator

# TODO update the JSON string below
json = "{}"
# create an instance of TaskEvaluator from a JSON string
task_evaluator_instance = TaskEvaluator.from_json(json)
# print the JSON string representation of the object
print(TaskEvaluator.to_json())

# convert the object into a dict
task_evaluator_dict = task_evaluator_instance.to_dict()
# create an instance of TaskEvaluator from a dict
task_evaluator_from_dict = TaskEvaluator.from_dict(task_evaluator_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


