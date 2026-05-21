# AnnotationQueueRecordCreateResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**record_sources** | [**List[AnnotationQueueRecord]**](AnnotationQueueRecord.md) | The created annotation queue records | 

## Example

```python
from arize._generated.api_client.models.annotation_queue_record_create_response import AnnotationQueueRecordCreateResponse

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationQueueRecordCreateResponse from a JSON string
annotation_queue_record_create_response_instance = AnnotationQueueRecordCreateResponse.from_json(json)
# print the JSON string representation of the object
print(AnnotationQueueRecordCreateResponse.to_json())

# convert the object into a dict
annotation_queue_record_create_response_dict = annotation_queue_record_create_response_instance.to_dict()
# create an instance of AnnotationQueueRecordCreateResponse from a dict
annotation_queue_record_create_response_from_dict = AnnotationQueueRecordCreateResponse.from_dict(annotation_queue_record_create_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


