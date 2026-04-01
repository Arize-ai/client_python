# AnnotationQueueRecordAnnotateResult

A snapshot of the annotation queue record fields that were modified by an annotate operation. Only the record identity fields and the submitted annotations are returned. Evaluations and user assignments are not fetched and are not included in this response for performance reasons; use the list records endpoint to retrieve the full record state.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the record | 
**annotation_queue_id** | **str** | The annotation queue this record belongs to | 
**source_type** | **str** | The source type of the record (spans or dataset) | 
**annotations** | [**List[Annotation]**](Annotation.md) | The annotations that were submitted in this request | 

## Example

```python
from arize._generated.api_client.models.annotation_queue_record_annotate_result import AnnotationQueueRecordAnnotateResult

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationQueueRecordAnnotateResult from a JSON string
annotation_queue_record_annotate_result_instance = AnnotationQueueRecordAnnotateResult.from_json(json)
# print the JSON string representation of the object
print(AnnotationQueueRecordAnnotateResult.to_json())

# convert the object into a dict
annotation_queue_record_annotate_result_dict = annotation_queue_record_annotate_result_instance.to_dict()
# create an instance of AnnotationQueueRecordAnnotateResult from a dict
annotation_queue_record_annotate_result_from_dict = AnnotationQueueRecordAnnotateResult.from_dict(annotation_queue_record_annotate_result_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


