# AnnotationQueueRecordInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**record_type** | **str** | The type of record | 
**dataset_id** | **str** | The dataset ID these examples belong to | 
**dataset_version_id** | **str** | Optional. The specific dataset version to use. If omitted, the latest version is used.  | [optional] 
**example_ids** | **List[str]** | List of example IDs within the dataset to add to the queue | 
**project_id** | **str** | The project ID these spans belong to | 
**start_time** | **datetime** | Start of the time range to search for spans in Druid. The range (end_time - start_time) must not exceed 7 days.  | 
**end_time** | **datetime** | End of the time range. Must be after start_time.  | 
**span_ids** | **List[str]** | List of span IDs to add to the queue | 

## Example

```python
from arize._generated.api_client.models.annotation_queue_record_input import AnnotationQueueRecordInput

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationQueueRecordInput from a JSON string
annotation_queue_record_input_instance = AnnotationQueueRecordInput.from_json(json)
# print the JSON string representation of the object
print(AnnotationQueueRecordInput.to_json())

# convert the object into a dict
annotation_queue_record_input_dict = annotation_queue_record_input_instance.to_dict()
# create an instance of AnnotationQueueRecordInput from a dict
annotation_queue_record_input_from_dict = AnnotationQueueRecordInput.from_dict(annotation_queue_record_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


