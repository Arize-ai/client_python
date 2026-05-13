# AnnotateSpansRequestBody

Batch annotation request for project spans.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**project_id** | **str** | The project (model) ID whose spans are being annotated. | 
**start_time** | **datetime** | Start of the time range for span lookup. Optional; defaults to 31 days ago. | [optional] 
**end_time** | **datetime** | End of the time range for span lookup. Optional; defaults to now. | [optional] 
**annotations** | [**List[AnnotateRecordInput]**](AnnotateRecordInput.md) | Batch of span annotations to write. Up to 1000 spans per request. | 

## Example

```python
from arize._generated.api_client.models.annotate_spans_request_body import AnnotateSpansRequestBody

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotateSpansRequestBody from a JSON string
annotate_spans_request_body_instance = AnnotateSpansRequestBody.from_json(json)
# print the JSON string representation of the object
print(AnnotateSpansRequestBody.to_json())

# convert the object into a dict
annotate_spans_request_body_dict = annotate_spans_request_body_instance.to_dict()
# create an instance of AnnotateSpansRequestBody from a dict
annotate_spans_request_body_from_dict = AnnotateSpansRequestBody.from_dict(annotate_spans_request_body_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


