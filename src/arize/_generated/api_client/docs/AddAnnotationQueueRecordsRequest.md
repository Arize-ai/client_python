# AddAnnotationQueueRecordsRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**record_sources** | [**List[AnnotationQueueRecordInput]**](AnnotationQueueRecordInput.md) | Record sources to add to the annotation queue. At most 2 record sources (projects or datasets) may be provided in a single request. | 

## Example

```python
from arize._generated.api_client.models.add_annotation_queue_records_request import AddAnnotationQueueRecordsRequest

# TODO update the JSON string below
json = "{}"
# create an instance of AddAnnotationQueueRecordsRequest from a JSON string
add_annotation_queue_records_request_instance = AddAnnotationQueueRecordsRequest.from_json(json)
# print the JSON string representation of the object
print(AddAnnotationQueueRecordsRequest.to_json())

# convert the object into a dict
add_annotation_queue_records_request_dict = add_annotation_queue_records_request_instance.to_dict()
# create an instance of AddAnnotationQueueRecordsRequest from a dict
add_annotation_queue_records_request_from_dict = AddAnnotationQueueRecordsRequest.from_dict(add_annotation_queue_records_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


