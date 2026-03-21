# TasksList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**tasks** | [**List[Task]**](Task.md) | A list of tasks | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.tasks_list200_response import TasksList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of TasksList200Response from a JSON string
tasks_list200_response_instance = TasksList200Response.from_json(json)
# print the JSON string representation of the object
print(TasksList200Response.to_json())

# convert the object into a dict
tasks_list200_response_dict = tasks_list200_response_instance.to_dict()
# create an instance of TasksList200Response from a dict
tasks_list200_response_from_dict = TasksList200Response.from_dict(tasks_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


