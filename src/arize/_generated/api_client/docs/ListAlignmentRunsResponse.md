# ListAlignmentRunsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**alignment_runs** | [**List[AlignmentRun]**](AlignmentRun.md) | A list of alignment runs, most recently created first | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_alignment_runs_response import ListAlignmentRunsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListAlignmentRunsResponse from a JSON string
list_alignment_runs_response_instance = ListAlignmentRunsResponse.from_json(json)
# print the JSON string representation of the object
print(ListAlignmentRunsResponse.to_json())

# convert the object into a dict
list_alignment_runs_response_dict = list_alignment_runs_response_instance.to_dict()
# create an instance of ListAlignmentRunsResponse from a dict
list_alignment_runs_response_from_dict = ListAlignmentRunsResponse.from_dict(list_alignment_runs_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


