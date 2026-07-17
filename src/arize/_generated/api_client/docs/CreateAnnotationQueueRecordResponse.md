# CreateAnnotationQueueRecordResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**record_sources** | [**List[AnnotationQueueRecord]**](AnnotationQueueRecord.md) | The created annotation queue records | 

## Example

```python
from arize._generated.api_client.models.create_annotation_queue_record_response import CreateAnnotationQueueRecordResponse

# TODO update the JSON string below
json = "{}"
# create an instance of CreateAnnotationQueueRecordResponse from a JSON string
create_annotation_queue_record_response_instance = CreateAnnotationQueueRecordResponse.from_json(json)
# print the JSON string representation of the object
print(CreateAnnotationQueueRecordResponse.to_json())

# convert the object into a dict
create_annotation_queue_record_response_dict = create_annotation_queue_record_response_instance.to_dict()
# create an instance of CreateAnnotationQueueRecordResponse from a dict
create_annotation_queue_record_response_from_dict = CreateAnnotationQueueRecordResponse.from_dict(create_annotation_queue_record_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


