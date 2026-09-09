# AnnotationQueueSessionRecordInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**record_type** | **str** | Discriminator identifying this record as a session record. | 
**project_id** | **str** | The project ID these sessions belong to. | 
**start_time** | **datetime** | Start of the time range used to resolve each session&#39;s first trace. The range (end_time - start_time) must not exceed 7 days.  | 
**end_time** | **datetime** | End of the time range. Must be after start_time.  | 
**session_ids** | **List[str]** | List of session IDs to add to the queue. A request may contain at most 100 session IDs in total across all record sources.  | 

## Example

```python
from arize._generated.api_client.models.annotation_queue_session_record_input import AnnotationQueueSessionRecordInput

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationQueueSessionRecordInput from a JSON string
annotation_queue_session_record_input_instance = AnnotationQueueSessionRecordInput.from_json(json)
# print the JSON string representation of the object
print(AnnotationQueueSessionRecordInput.to_json())

# convert the object into a dict
annotation_queue_session_record_input_dict = annotation_queue_session_record_input_instance.to_dict()
# create an instance of AnnotationQueueSessionRecordInput from a dict
annotation_queue_session_record_input_from_dict = AnnotationQueueSessionRecordInput.from_dict(annotation_queue_session_record_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


