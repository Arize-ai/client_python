# TaskRun

A task run is an async job that executes the work defined on a task. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the task run. | 
**task_id** | **str** | The parent task global ID (base64). | 
**status** | **str** | The current status of the run. | 
**run_started_at** | **datetime** | When the run started processing. | 
**run_finished_at** | **datetime** | When the run finished processing. | 
**data_start_time** | **datetime** | Start of the data window evaluated. | 
**data_end_time** | **datetime** | End of the data window evaluated. | 
**num_successes** | **int** | Number of successfully evaluated items. | 
**num_errors** | **int** | Number of items that errored during evaluation. | 
**num_skipped** | **int** | Number of items that were skipped. | 
**created_at** | **datetime** | When the run was created. | 
**created_by_user_id** | **str** | The unique identifier for the user who triggered the run. | 

## Example

```python
from arize._generated.api_client.models.task_run import TaskRun

# TODO update the JSON string below
json = "{}"
# create an instance of TaskRun from a JSON string
task_run_instance = TaskRun.from_json(json)
# print the JSON string representation of the object
print(TaskRun.to_json())

# convert the object into a dict
task_run_dict = task_run_instance.to_dict()
# create an instance of TaskRun from a dict
task_run_from_dict = TaskRun.from_dict(task_run_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


