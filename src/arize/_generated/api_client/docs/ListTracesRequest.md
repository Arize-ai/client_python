# ListTracesRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**project_id** | **str** | The project ID to list traces for | 
**start_time** | **datetime** | Return traces whose spans start at or after this timestamp (inclusive). ISO 8601 format (e.g., &#x60;2024-01-01T00:00:00Z&#x60;). Defaults to 1 week ago.  | [optional] 
**end_time** | **datetime** | Return traces whose spans start before this timestamp (exclusive). ISO 8601 format (e.g., &#x60;2024-01-02T00:00:00Z&#x60;). Defaults to the current time.  | [optional] 
**filter** | **str** | Filter expression to apply to the query. Supports SQL-like syntax for filtering spans by attributes (e.g., &#x60;status_code &#x3D; &#39;ERROR&#39;&#x60; or &#x60;span_kind &#x3D; &#39;LLM&#39;&#x60;). A trace is returned when **any** of its spans matches the filter — the matching span is usually a child, not the root. Optional; omit it to apply no filter. If provided, it must not be empty or whitespace-only.  | [optional] 

## Example

```python
from arize._generated.api_client.models.list_traces_request import ListTracesRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ListTracesRequest from a JSON string
list_traces_request_instance = ListTracesRequest.from_json(json)
# print the JSON string representation of the object
print(ListTracesRequest.to_json())

# convert the object into a dict
list_traces_request_dict = list_traces_request_instance.to_dict()
# create an instance of ListTracesRequest from a dict
list_traces_request_from_dict = ListTracesRequest.from_dict(list_traces_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


