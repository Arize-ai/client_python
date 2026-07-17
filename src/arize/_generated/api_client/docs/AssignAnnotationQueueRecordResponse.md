# AssignAnnotationQueueRecordResponse

A snapshot of the annotation queue record fields that were modified by an assign operation. Only the record identity fields and the resulting user assignments are returned. Annotations and evaluations are not fetched and are not included in this response for performance reasons; use the list records endpoint to retrieve the full record state.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the record | 
**annotation_queue_id** | **str** | The annotation queue this record belongs to | 
**source_type** | [**AnnotationQueueSourceType**](AnnotationQueueSourceType.md) |  | 
**granularity** | [**RecordGranularity**](RecordGranularity.md) | The granularity of the record, if applicable. | [optional] 
**assigned_users** | [**List[AnnotationQueueAssignedUser]**](AnnotationQueueAssignedUser.md) | The users now assigned to this record after this operation | 

## Example

```python
from arize._generated.api_client.models.assign_annotation_queue_record_response import AssignAnnotationQueueRecordResponse

# TODO update the JSON string below
json = "{}"
# create an instance of AssignAnnotationQueueRecordResponse from a JSON string
assign_annotation_queue_record_response_instance = AssignAnnotationQueueRecordResponse.from_json(json)
# print the JSON string representation of the object
print(AssignAnnotationQueueRecordResponse.to_json())

# convert the object into a dict
assign_annotation_queue_record_response_dict = assign_annotation_queue_record_response_instance.to_dict()
# create an instance of AssignAnnotationQueueRecordResponse from a dict
assign_annotation_queue_record_response_from_dict = AssignAnnotationQueueRecordResponse.from_dict(assign_annotation_queue_record_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


