# AnnotationQueueTraceRecordInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**record_type** | **str** | Discriminator identifying this record as a trace record. | 
**project_id** | **str** | The project ID these traces belong to. | 
**start_time** | **datetime** | Start of the time range used to resolve each trace&#39;s root span. The range (end_time - start_time) must not exceed 7 days.  | 
**end_time** | **datetime** | End of the time range. Must be after start_time.  | 
**trace_ids** | **List[str]** | List of trace IDs to add to the queue.  | 

## Example

```python
from arize._generated.api_client.models.annotation_queue_trace_record_input import AnnotationQueueTraceRecordInput

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationQueueTraceRecordInput from a JSON string
annotation_queue_trace_record_input_instance = AnnotationQueueTraceRecordInput.from_json(json)
# print the JSON string representation of the object
print(AnnotationQueueTraceRecordInput.to_json())

# convert the object into a dict
annotation_queue_trace_record_input_dict = annotation_queue_trace_record_input_instance.to_dict()
# create an instance of AnnotationQueueTraceRecordInput from a dict
annotation_queue_trace_record_input_from_dict = AnnotationQueueTraceRecordInput.from_dict(annotation_queue_trace_record_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


