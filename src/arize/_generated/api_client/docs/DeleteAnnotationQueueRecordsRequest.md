# DeleteAnnotationQueueRecordsRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**record_ids** | **List[str]** | The IDs of the annotation queue records to delete. | 

## Example

```python
from arize._generated.api_client.models.delete_annotation_queue_records_request import DeleteAnnotationQueueRecordsRequest

# TODO update the JSON string below
json = "{}"
# create an instance of DeleteAnnotationQueueRecordsRequest from a JSON string
delete_annotation_queue_records_request_instance = DeleteAnnotationQueueRecordsRequest.from_json(json)
# print the JSON string representation of the object
print(DeleteAnnotationQueueRecordsRequest.to_json())

# convert the object into a dict
delete_annotation_queue_records_request_dict = delete_annotation_queue_records_request_instance.to_dict()
# create an instance of DeleteAnnotationQueueRecordsRequest from a dict
delete_annotation_queue_records_request_from_dict = DeleteAnnotationQueueRecordsRequest.from_dict(delete_annotation_queue_records_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


