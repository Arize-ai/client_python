# Trace

A Trace is the collection of spans sharing a `trace_id`, anchored on a root span (a span with no parent). It captures a single end-to-end request through an LLM application, with lightweight roll-up metadata plus the full flat list of its spans. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**trace_id** | **str** | Unique identifier for the trace. | 
**root_span_id** | **str** | Span ID of the root span (the span with no parent) that anchors this trace entry. A trace with more than one root span is returned as multiple entries sharing the same &#x60;trace_id&#x60;, distinguished by &#x60;root_span_id&#x60;.  | 
**start_time** | **datetime** | Earliest span start time across the returned spans. | [optional] 
**end_time** | **datetime** | Latest span end time across the returned spans. | [optional] 
**spans_truncated** | **bool** | &#x60;true&#x60; when this trace contained more spans than the per-trace limit and its returned span list is incomplete. &#x60;false&#x60; otherwise.  Note: each page also has an overall cap on the total number of spans returned across all of its traces. On pages that include unusually large traces, an individual trace may return fewer spans than it actually has even when &#x60;spans_truncated&#x60; is &#x60;false&#x60;. To retrieve a trace&#39;s spans in full, narrow the time window or fetch them directly with &#x60;POST /v2/spans&#x60; filtered to that &#x60;trace_id&#x60;.  | 
**spans** | [**List[Span]**](Span.md) | Flat list of spans belonging to this trace. Each span has the same shape and enrichment as spans returned by &#x60;POST /v2/spans&#x60;. Reconstruct the trace tree client-side using each span&#39;s &#x60;parent_id&#x60;.  | 

## Example

```python
from arize._generated.api_client.models.trace import Trace

# TODO update the JSON string below
json = "{}"
# create an instance of Trace from a JSON string
trace_instance = Trace.from_json(json)
# print the JSON string representation of the object
print(Trace.to_json())

# convert the object into a dict
trace_dict = trace_instance.to_dict()
# create an instance of Trace from a dict
trace_from_dict = Trace.from_dict(trace_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


