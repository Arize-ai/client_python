# Task

A task is a typed, configurable unit of work that ties one or more evaluators to a data source (project or dataset). 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the task | 
**name** | **str** | The name of the task | 
**type** | **str** | The task type: template_evaluation or code_evaluation | 
**project_id** | **str** | The project global ID (base64). Present for project-based tasks. | [optional] 
**dataset_id** | **str** | The dataset global ID (base64). Present for dataset-based tasks. | [optional] 
**sampling_rate** | **float** | Sampling rate between 0 and 1. Only applicable for project-based tasks. | [optional] 
**is_continuous** | **bool** | Whether the task runs continuously on incoming data. | 
**query_filter** | **str** | Task-level query filter applied to all data. | 
**evaluators** | [**List[TaskEvaluator]**](TaskEvaluator.md) | The evaluators attached to this task. | 
**experiment_ids** | **List[str]** | Experiment global IDs (base64) for dataset-based tasks. | 
**last_run_at** | **datetime** | When the task was last run. | 
**created_at** | **datetime** | When the task was created. | 
**updated_at** | **datetime** | When the task was last updated. | 
**created_by_user_id** | **str** | The unique identifier for the user who created the task. | 

## Example

```python
from arize._generated.api_client.models.task import Task

# TODO update the JSON string below
json = "{}"
# create an instance of Task from a JSON string
task_instance = Task.from_json(json)
# print the JSON string representation of the object
print(Task.to_json())

# convert the object into a dict
task_dict = task_instance.to_dict()
# create an instance of Task from a dict
task_from_dict = Task.from_dict(task_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


