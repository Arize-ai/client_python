# DeleteAnnotationQueueRecordsRequestBody


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**record_ids** | **List[str]** | The IDs of the annotation queue records to delete. | 

## Example

```python
from arize._generated.api_client.models.delete_annotation_queue_records_request_body import DeleteAnnotationQueueRecordsRequestBody

# TODO update the JSON string below
json = "{}"
# create an instance of DeleteAnnotationQueueRecordsRequestBody from a JSON string
delete_annotation_queue_records_request_body_instance = DeleteAnnotationQueueRecordsRequestBody.from_json(json)
# print the JSON string representation of the object
print(DeleteAnnotationQueueRecordsRequestBody.to_json())

# convert the object into a dict
delete_annotation_queue_records_request_body_dict = delete_annotation_queue_records_request_body_instance.to_dict()
# create an instance of DeleteAnnotationQueueRecordsRequestBody from a dict
delete_annotation_queue_records_request_body_from_dict = DeleteAnnotationQueueRecordsRequestBody.from_dict(delete_annotation_queue_records_request_body_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


