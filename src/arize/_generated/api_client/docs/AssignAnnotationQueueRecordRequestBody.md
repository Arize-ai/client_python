# AssignAnnotationQueueRecordRequestBody

User assignment for an annotation queue record. Fully replaces the current record-level user assignment. Pass an empty array to remove all assignments.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**assigned_user_emails** | **List[str]** | Emails of users to assign to this record. Replaces the current record-level user assignment entirely. At most 100 emails may be provided per request. | 

## Example

```python
from arize._generated.api_client.models.assign_annotation_queue_record_request_body import AssignAnnotationQueueRecordRequestBody

# TODO update the JSON string below
json = "{}"
# create an instance of AssignAnnotationQueueRecordRequestBody from a JSON string
assign_annotation_queue_record_request_body_instance = AssignAnnotationQueueRecordRequestBody.from_json(json)
# print the JSON string representation of the object
print(AssignAnnotationQueueRecordRequestBody.to_json())

# convert the object into a dict
assign_annotation_queue_record_request_body_dict = assign_annotation_queue_record_request_body_instance.to_dict()
# create an instance of AssignAnnotationQueueRecordRequestBody from a dict
assign_annotation_queue_record_request_body_from_dict = AssignAnnotationQueueRecordRequestBody.from_dict(assign_annotation_queue_record_request_body_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


