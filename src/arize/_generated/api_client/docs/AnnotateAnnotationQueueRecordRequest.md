# AnnotateAnnotationQueueRecordRequest

Annotations to submit for an annotation queue record. Annotations are upserted by annotation config name; omitted configs are left unchanged.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**annotations** | [**List[AnnotationInput]**](AnnotationInput.md) | Annotations to upsert on this record, keyed by annotation config name. At most 500 annotations may be submitted per request — one per annotation config associated with the queue. | 

## Example

```python
from arize._generated.api_client.models.annotate_annotation_queue_record_request import AnnotateAnnotationQueueRecordRequest

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotateAnnotationQueueRecordRequest from a JSON string
annotate_annotation_queue_record_request_instance = AnnotateAnnotationQueueRecordRequest.from_json(json)
# print the JSON string representation of the object
print(AnnotateAnnotationQueueRecordRequest.to_json())

# convert the object into a dict
annotate_annotation_queue_record_request_dict = annotate_annotation_queue_record_request_instance.to_dict()
# create an instance of AnnotateAnnotationQueueRecordRequest from a dict
annotate_annotation_queue_record_request_from_dict = AnnotateAnnotationQueueRecordRequest.from_dict(annotate_annotation_queue_record_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


