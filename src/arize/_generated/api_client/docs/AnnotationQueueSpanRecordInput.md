# AnnotationQueueSpanRecordInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**record_type** | **str** | The type of record | 
**project_id** | **str** | The project ID these spans belong to | 
**start_time** | **datetime** | Start of the time range to search for spans in Druid. The range (end_time - start_time) must not exceed 7 days.  | 
**end_time** | **datetime** | End of the time range. Must be after start_time.  | 
**span_ids** | **List[str]** | List of span IDs to add to the queue | 

## Example

```python
from arize._generated.api_client.models.annotation_queue_span_record_input import AnnotationQueueSpanRecordInput

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationQueueSpanRecordInput from a JSON string
annotation_queue_span_record_input_instance = AnnotationQueueSpanRecordInput.from_json(json)
# print the JSON string representation of the object
print(AnnotationQueueSpanRecordInput.to_json())

# convert the object into a dict
annotation_queue_span_record_input_dict = annotation_queue_span_record_input_instance.to_dict()
# create an instance of AnnotationQueueSpanRecordInput from a dict
annotation_queue_span_record_input_from_dict = AnnotationQueueSpanRecordInput.from_dict(annotation_queue_span_record_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


