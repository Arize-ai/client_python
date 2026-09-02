# AnnotateSpansRequest

Batch annotation request for project spans.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**project_id** | **str** | The project (model) ID whose spans are being annotated. | 
**start_time** | **datetime** | Start of the time range for span lookup. Optional; defaults to 31 days before end_time, or 7 days before end_time when granularity is SESSION. | [optional] 
**end_time** | **datetime** | End of the time range for span lookup. Optional; defaults to now. | [optional] 
**granularity** | [**RecordGranularity**](RecordGranularity.md) | Whether the record is a span, a trace, or a session, which affects whether annotations are written as span, trace, or session annotations. For TRACE, each &#x60;record_id&#x60; must be a trace&#39;s root span; attempts to write trace annotations on non-root spans will be rejected. For SESSION, each &#x60;record_id&#x60; is a session ID; the annotation is written to the root span of the session&#39;s earliest trace within the lookup window. Optional; defaults to &#39;SPAN&#39;. | [optional] 
**annotations** | [**List[AnnotateRecordInput]**](AnnotateRecordInput.md) | Batch of annotations to write. Up to 1000 records per request for SPAN or TRACE granularity; up to 100 records per request for SESSION granularity. | 

## Example

```python
from arize._generated.api_client.models.annotate_spans_request import AnnotateSpansRequest

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotateSpansRequest from a JSON string
annotate_spans_request_instance = AnnotateSpansRequest.from_json(json)
# print the JSON string representation of the object
print(AnnotateSpansRequest.to_json())

# convert the object into a dict
annotate_spans_request_dict = annotate_spans_request_instance.to_dict()
# create an instance of AnnotateSpansRequest from a dict
annotate_spans_request_from_dict = AnnotateSpansRequest.from_dict(annotate_spans_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


