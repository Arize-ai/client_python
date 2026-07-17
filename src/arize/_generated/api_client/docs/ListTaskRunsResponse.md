# ListTaskRunsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**task_runs** | [**List[TaskRun]**](TaskRun.md) | A list of task runs | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_task_runs_response import ListTaskRunsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListTaskRunsResponse from a JSON string
list_task_runs_response_instance = ListTaskRunsResponse.from_json(json)
# print the JSON string representation of the object
print(ListTaskRunsResponse.to_json())

# convert the object into a dict
list_task_runs_response_dict = list_task_runs_response_instance.to_dict()
# create an instance of ListTaskRunsResponse from a dict
list_task_runs_response_from_dict = ListTaskRunsResponse.from_dict(list_task_runs_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


