# TasksListRuns200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**task_runs** | [**List[TaskRun]**](TaskRun.md) | A list of task runs | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.tasks_list_runs200_response import TasksListRuns200Response

# TODO update the JSON string below
json = "{}"
# create an instance of TasksListRuns200Response from a JSON string
tasks_list_runs200_response_instance = TasksListRuns200Response.from_json(json)
# print the JSON string representation of the object
print(TasksListRuns200Response.to_json())

# convert the object into a dict
tasks_list_runs200_response_dict = tasks_list_runs200_response_instance.to_dict()
# create an instance of TasksListRuns200Response from a dict
tasks_list_runs200_response_from_dict = TasksListRuns200Response.from_dict(tasks_list_runs200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


