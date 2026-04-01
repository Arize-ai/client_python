# AnnotationQueuesRecordsCreate200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**record_sources** | [**List[AnnotationQueueRecord]**](AnnotationQueueRecord.md) | The created annotation queue records | 

## Example

```python
from arize._generated.api_client.models.annotation_queues_records_create200_response import AnnotationQueuesRecordsCreate200Response

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationQueuesRecordsCreate200Response from a JSON string
annotation_queues_records_create200_response_instance = AnnotationQueuesRecordsCreate200Response.from_json(json)
# print the JSON string representation of the object
print(AnnotationQueuesRecordsCreate200Response.to_json())

# convert the object into a dict
annotation_queues_records_create200_response_dict = annotation_queues_records_create200_response_instance.to_dict()
# create an instance of AnnotationQueuesRecordsCreate200Response from a dict
annotation_queues_records_create200_response_from_dict = AnnotationQueuesRecordsCreate200Response.from_dict(annotation_queues_records_create200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


