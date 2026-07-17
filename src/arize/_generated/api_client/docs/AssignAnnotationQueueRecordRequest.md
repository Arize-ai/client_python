# AssignAnnotationQueueRecordRequest

User assignment for an annotation queue record. Fully replaces the current record-level user assignment. Pass an empty array to remove all assignments.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**assigned_user_emails** | **List[str]** | Emails of users to assign to this record. Replaces the current record-level user assignment entirely. At most 100 emails may be provided per request. | 

## Example

```python
from arize._generated.api_client.models.assign_annotation_queue_record_request import AssignAnnotationQueueRecordRequest

# TODO update the JSON string below
json = "{}"
# create an instance of AssignAnnotationQueueRecordRequest from a JSON string
assign_annotation_queue_record_request_instance = AssignAnnotationQueueRecordRequest.from_json(json)
# print the JSON string representation of the object
print(AssignAnnotationQueueRecordRequest.to_json())

# convert the object into a dict
assign_annotation_queue_record_request_dict = assign_annotation_queue_record_request_instance.to_dict()
# create an instance of AssignAnnotationQueueRecordRequest from a dict
assign_annotation_queue_record_request_from_dict = AssignAnnotationQueueRecordRequest.from_dict(assign_annotation_queue_record_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


