# TaskListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**tasks** | [**List[Task]**](Task.md) | A list of tasks | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.task_list_response import TaskListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of TaskListResponse from a JSON string
task_list_response_instance = TaskListResponse.from_json(json)
# print the JSON string representation of the object
print(TaskListResponse.to_json())

# convert the object into a dict
task_list_response_dict = task_list_response_instance.to_dict()
# create an instance of TaskListResponse from a dict
task_list_response_from_dict = TaskListResponse.from_dict(task_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


