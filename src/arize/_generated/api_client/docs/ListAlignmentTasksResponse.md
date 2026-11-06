# ListAlignmentTasksResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**alignment_tasks** | [**List[AlignmentTask]**](AlignmentTask.md) | A list of alignment tasks, most recently created first | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_alignment_tasks_response import ListAlignmentTasksResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListAlignmentTasksResponse from a JSON string
list_alignment_tasks_response_instance = ListAlignmentTasksResponse.from_json(json)
# print the JSON string representation of the object
print(ListAlignmentTasksResponse.to_json())

# convert the object into a dict
list_alignment_tasks_response_dict = list_alignment_tasks_response_instance.to_dict()
# create an instance of ListAlignmentTasksResponse from a dict
list_alignment_tasks_response_from_dict = ListAlignmentTasksResponse.from_dict(list_alignment_tasks_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


