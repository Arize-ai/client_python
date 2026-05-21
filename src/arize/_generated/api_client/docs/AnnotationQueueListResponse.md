# AnnotationQueueListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**annotation_queues** | [**List[AnnotationQueue]**](AnnotationQueue.md) | A list of annotation queues | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.annotation_queue_list_response import AnnotationQueueListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationQueueListResponse from a JSON string
annotation_queue_list_response_instance = AnnotationQueueListResponse.from_json(json)
# print the JSON string representation of the object
print(AnnotationQueueListResponse.to_json())

# convert the object into a dict
annotation_queue_list_response_dict = annotation_queue_list_response_instance.to_dict()
# create an instance of AnnotationQueueListResponse from a dict
annotation_queue_list_response_from_dict = AnnotationQueueListResponse.from_dict(annotation_queue_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


