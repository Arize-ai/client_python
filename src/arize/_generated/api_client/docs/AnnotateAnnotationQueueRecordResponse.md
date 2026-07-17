# AnnotateAnnotationQueueRecordResponse

A snapshot of the annotation queue record fields that were modified by an annotate operation. Only the record identity fields and the submitted annotations are returned. Evaluations and user assignments are not fetched and are not included in this response for performance reasons; use the list records endpoint to retrieve the full record state.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the record | 
**annotation_queue_id** | **str** | The annotation queue this record belongs to | 
**source_type** | [**AnnotationQueueSourceType**](AnnotationQueueSourceType.md) |  | 
**granularity** | [**RecordGranularity**](RecordGranularity.md) | The granularity of the record, if applicable. | [optional] 
**annotations** | [**List[Annotation]**](Annotation.md) | The annotations that were submitted in this request | 

## Example

```python
from arize._generated.api_client.models.annotate_annotation_queue_record_response import AnnotateAnnotationQueueRecordResponse

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotateAnnotationQueueRecordResponse from a JSON string
annotate_annotation_queue_record_response_instance = AnnotateAnnotationQueueRecordResponse.from_json(json)
# print the JSON string representation of the object
print(AnnotateAnnotationQueueRecordResponse.to_json())

# convert the object into a dict
annotate_annotation_queue_record_response_dict = annotate_annotation_queue_record_response_instance.to_dict()
# create an instance of AnnotateAnnotationQueueRecordResponse from a dict
annotate_annotation_queue_record_response_from_dict = AnnotateAnnotationQueueRecordResponse.from_dict(annotate_annotation_queue_record_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


