# AnnotationQueueAssignedUser

A user assigned to a record with their completion status

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**user** | [**AnnotatorUser**](AnnotatorUser.md) |  | 
**completion_status** | **str** | The completion status for this user on this record | 

## Example

```python
from arize._generated.api_client.models.annotation_queue_assigned_user import AnnotationQueueAssignedUser

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationQueueAssignedUser from a JSON string
annotation_queue_assigned_user_instance = AnnotationQueueAssignedUser.from_json(json)
# print the JSON string representation of the object
print(AnnotationQueueAssignedUser.to_json())

# convert the object into a dict
annotation_queue_assigned_user_dict = annotation_queue_assigned_user_instance.to_dict()
# create an instance of AnnotationQueueAssignedUser from a dict
annotation_queue_assigned_user_from_dict = AnnotationQueueAssignedUser.from_dict(annotation_queue_assigned_user_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


