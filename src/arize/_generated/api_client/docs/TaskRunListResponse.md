# TaskRunListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**task_runs** | [**List[TaskRun]**](TaskRun.md) | A list of task runs | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.task_run_list_response import TaskRunListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of TaskRunListResponse from a JSON string
task_run_list_response_instance = TaskRunListResponse.from_json(json)
# print the JSON string representation of the object
print(TaskRunListResponse.to_json())

# convert the object into a dict
task_run_list_response_dict = task_run_list_response_instance.to_dict()
# create an instance of TaskRunListResponse from a dict
task_run_list_response_from_dict = TaskRunListResponse.from_dict(task_run_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


