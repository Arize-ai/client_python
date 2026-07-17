# ListAnnotationQueuesResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**annotation_queues** | [**List[AnnotationQueue]**](AnnotationQueue.md) | A list of annotation queues | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_annotation_queues_response import ListAnnotationQueuesResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListAnnotationQueuesResponse from a JSON string
list_annotation_queues_response_instance = ListAnnotationQueuesResponse.from_json(json)
# print the JSON string representation of the object
print(ListAnnotationQueuesResponse.to_json())

# convert the object into a dict
list_annotation_queues_response_dict = list_annotation_queues_response_instance.to_dict()
# create an instance of ListAnnotationQueuesResponse from a dict
list_annotation_queues_response_from_dict = ListAnnotationQueuesResponse.from_dict(list_annotation_queues_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


