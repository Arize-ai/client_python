# AnnotationQueueRecordListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**records** | [**List[AnnotationQueueRecord]**](AnnotationQueueRecord.md) | A list of annotation queue records | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.annotation_queue_record_list_response import AnnotationQueueRecordListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationQueueRecordListResponse from a JSON string
annotation_queue_record_list_response_instance = AnnotationQueueRecordListResponse.from_json(json)
# print the JSON string representation of the object
print(AnnotationQueueRecordListResponse.to_json())

# convert the object into a dict
annotation_queue_record_list_response_dict = annotation_queue_record_list_response_instance.to_dict()
# create an instance of AnnotationQueueRecordListResponse from a dict
annotation_queue_record_list_response_from_dict = AnnotationQueueRecordListResponse.from_dict(annotation_queue_record_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


