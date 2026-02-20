# AnnotationQueuesList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**annotation_queues** | [**List[AnnotationQueue]**](AnnotationQueue.md) | A list of annotation queues | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.annotation_queues_list200_response import AnnotationQueuesList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationQueuesList200Response from a JSON string
annotation_queues_list200_response_instance = AnnotationQueuesList200Response.from_json(json)
# print the JSON string representation of the object
print(AnnotationQueuesList200Response.to_json())

# convert the object into a dict
annotation_queues_list200_response_dict = annotation_queues_list200_response_instance.to_dict()
# create an instance of AnnotationQueuesList200Response from a dict
annotation_queues_list200_response_from_dict = AnnotationQueuesList200Response.from_dict(annotation_queues_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


