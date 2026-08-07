# DeleteSpansRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**project_id** | **str** | The project ID containing the spans to delete | 
**span_ids** | **List[str]** | List of span IDs to delete (maximum 5000) | 
**start_time** | **datetime** | Scope the delete to spans starting at or after this timestamp (inclusive). ISO 8601 format (e.g., &#x60;2024-01-01T00:00:00Z&#x60;). Each bound is independent: omitting &#x60;start_time&#x60; defaults to two years ago; omitting &#x60;end_time&#x60; defaults to now. You may provide either or both.  | [optional] 
**end_time** | **datetime** | Scope the delete to spans starting before this timestamp (exclusive). ISO 8601 format (e.g., &#x60;2024-01-02T00:00:00Z&#x60;). Each bound is independent: omitting &#x60;start_time&#x60; defaults to two years ago; omitting &#x60;end_time&#x60; defaults to now. You may provide either or both.  | [optional] 

## Example

```python
from arize._generated.api_client.models.delete_spans_request import DeleteSpansRequest

# TODO update the JSON string below
json = "{}"
# create an instance of DeleteSpansRequest from a JSON string
delete_spans_request_instance = DeleteSpansRequest.from_json(json)
# print the JSON string representation of the object
print(DeleteSpansRequest.to_json())

# convert the object into a dict
delete_spans_request_dict = delete_spans_request_instance.to_dict()
# create an instance of DeleteSpansRequest from a dict
delete_spans_request_from_dict = DeleteSpansRequest.from_dict(delete_spans_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


