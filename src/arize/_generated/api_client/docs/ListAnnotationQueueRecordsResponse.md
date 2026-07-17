# ListAnnotationQueueRecordsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**records** | [**List[AnnotationQueueRecord]**](AnnotationQueueRecord.md) | A list of annotation queue records | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_annotation_queue_records_response import ListAnnotationQueueRecordsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListAnnotationQueueRecordsResponse from a JSON string
list_annotation_queue_records_response_instance = ListAnnotationQueueRecordsResponse.from_json(json)
# print the JSON string representation of the object
print(ListAnnotationQueueRecordsResponse.to_json())

# convert the object into a dict
list_annotation_queue_records_response_dict = list_annotation_queue_records_response_instance.to_dict()
# create an instance of ListAnnotationQueueRecordsResponse from a dict
list_annotation_queue_records_response_from_dict = ListAnnotationQueueRecordsResponse.from_dict(list_annotation_queue_records_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


