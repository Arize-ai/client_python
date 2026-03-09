# AnnotationQueueRecord

A record in an annotation queue with its data

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the record | 
**annotation_queue_id** | **str** | The annotation queue this record belongs to | 
**source_type** | **str** | The source type of the record (spans or dataset) | 
**data** | **Dict[str, object]** | Record data as flat key-value pairs containing span or dataset fields. Does not include annotation or evaluation columns. | 
**annotations** | [**List[Annotation]**](Annotation.md) | Human annotations on this record | 
**evaluations** | [**List[Evaluation]**](Evaluation.md) | Evaluation results on this record | 
**assigned_users** | [**List[AnnotationQueueAssignedUser]**](AnnotationQueueAssignedUser.md) | Users assigned to this record | 

## Example

```python
from arize._generated.api_client.models.annotation_queue_record import AnnotationQueueRecord

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationQueueRecord from a JSON string
annotation_queue_record_instance = AnnotationQueueRecord.from_json(json)
# print the JSON string representation of the object
print(AnnotationQueueRecord.to_json())

# convert the object into a dict
annotation_queue_record_dict = annotation_queue_record_instance.to_dict()
# create an instance of AnnotationQueueRecord from a dict
annotation_queue_record_from_dict = AnnotationQueueRecord.from_dict(annotation_queue_record_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


