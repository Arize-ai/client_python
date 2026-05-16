# ListSpansRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**project_id** | **str** | The project ID to list spans for | 
**start_time** | **datetime** | Filter to spans starting at or after this timestamp (inclusive). ISO 8601 format (e.g., &#x60;2024-01-01T00:00:00Z&#x60;). Defaults to 1 week ago.  | [optional] 
**end_time** | **datetime** | Filter to spans starting before this timestamp (exclusive). ISO 8601 format (e.g., &#x60;2024-01-02T00:00:00Z&#x60;). Defaults to the current time.  | [optional] 
**filter** | **str** | Filter expression to apply to the query. Supports SQL-like syntax for filtering spans by attributes (e.g., &#x60;status_code &#x3D; &#39;ERROR&#39;&#x60;).  | [optional] 

## Example

```python
from arize._generated.api_client.models.list_spans_request import ListSpansRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ListSpansRequest from a JSON string
list_spans_request_instance = ListSpansRequest.from_json(json)
# print the JSON string representation of the object
print(ListSpansRequest.to_json())

# convert the object into a dict
list_spans_request_dict = list_spans_request_instance.to_dict()
# create an instance of ListSpansRequest from a dict
list_spans_request_from_dict = ListSpansRequest.from_dict(list_spans_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


