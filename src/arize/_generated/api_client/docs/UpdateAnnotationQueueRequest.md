# UpdateAnnotationQueueRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | The name of the annotation queue. Must be unique within the space.  | [optional] 
**instructions** | **str** | The instructions for annotators working on this queue. Send an empty string to clear the instructions.  | [optional] 
**annotation_config_ids** | **List[str]** | The full list of annotation config IDs to associate with this queue. This replaces all existing annotation config associations. All annotation configs must belong to the same space as the queue.  | [optional] 
**annotator_emails** | **List[str]** | The full list of user emails to assign to this queue. This replaces all existing user assignments. All users must have an active account and access to the queue&#39;s space.  | [optional] 

## Example

```python
from arize._generated.api_client.models.update_annotation_queue_request import UpdateAnnotationQueueRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateAnnotationQueueRequest from a JSON string
update_annotation_queue_request_instance = UpdateAnnotationQueueRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateAnnotationQueueRequest.to_json())

# convert the object into a dict
update_annotation_queue_request_dict = update_annotation_queue_request_instance.to_dict()
# create an instance of UpdateAnnotationQueueRequest from a dict
update_annotation_queue_request_from_dict = UpdateAnnotationQueueRequest.from_dict(update_annotation_queue_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


