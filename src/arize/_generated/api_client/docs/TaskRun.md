# TaskRun

A task run is an async job that executes the work defined on a task. Runs are created by triggering an existing task (`POST /v2/tasks/{task_id}/trigger`). For `run_experiment` tasks, `experiment_id` is populated after the experiment is provisioned; poll `GET /v2/task-runs/{run_id}` until `status` reaches a terminal state. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the task run. | 
**task_id** | **str** | The parent task global ID (base64). | 
**experiment_id** | **str** | Created experiment global ID (base64). Present only for &#x60;run_experiment&#x60; task runs; null for all other task types.  | [optional] 
**status** | **str** | The current status of the run. | 
**run_started_at** | **datetime** | When the run started processing. | 
**run_finished_at** | **datetime** | When the run finished processing. | 
**data_start_time** | **datetime** | Start of the data window evaluated. Null for run_experiment runs. | 
**data_end_time** | **datetime** | End of the data window evaluated. Null for run_experiment runs. | 
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


