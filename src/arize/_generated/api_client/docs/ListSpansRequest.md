# ListSpansRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**project_id** | **str** | The project ID to list spans for | 
**start_time** | **datetime** | Filter to spans starting at or after this timestamp (inclusive). ISO 8601 format (e.g., &#x60;2024-01-01T00:00:00Z&#x60;). Defaults to 1 week ago.  | [optional] 
**end_time** | **datetime** | Filter to spans starting before this timestamp (exclusive). ISO 8601 format (e.g., &#x60;2024-01-02T00:00:00Z&#x60;). Defaults to the current time.  | [optional] 
**filter** | **str** | Filter expression to apply to the query. Supports SQL-like syntax for filtering spans by attributes (e.g., &#x60;status_code &#x3D; &#39;ERROR&#39;&#x60;). Optional; omit it to apply no filter. If provided, it must not be empty or whitespace-only.  | [optional] 
**included_columns** | **List[str]** | Columns to include in each span. When set, only these columns (plus fixed span fields) are returned. Mutually exclusive with &#x60;excluded_columns&#x60; — providing both returns 422.  Values must be full dotted column paths (e.g., &#x60;attributes.llm.model_name&#x60;, &#x60;eval.hallucination.score&#x60;). Unknown column names are silently ignored.  Fixed span fields — name, context (trace_id, span_id), kind, parent_id, start_time, end_time, status_code, status_message, latency_ms, and events — are always returned regardless of this parameter.  | [optional] 
**excluded_columns** | **List[str]** | Columns to exclude from each span. When set, all columns except these are returned. Mutually exclusive with &#x60;included_columns&#x60; — providing both returns 422.  Values must be full dotted column paths (e.g., &#x60;attributes.embedding.vectors&#x60;, &#x60;eval.toxicity.score&#x60;). Unknown column names are silently ignored. Attempts to exclude fixed span fields (name, context, kind, parent_id, start_time, end_time, status_code, status_message, latency_ms, events) are silently ignored.  | [optional] 

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


