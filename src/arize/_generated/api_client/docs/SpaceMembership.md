# SpaceMembership

A space membership record. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the membership record | 
**user_id** | **str** | The unique identifier of the user | 
**space_id** | **str** | The unique identifier of the space | 
**role** | [**SpaceRoleAssignment**](SpaceRoleAssignment.md) |  | 

## Example

```python
from arize._generated.api_client.models.space_membership import SpaceMembership

# TODO update the JSON string below
json = "{}"
# create an instance of SpaceMembership from a JSON string
space_membership_instance = SpaceMembership.from_json(json)
# print the JSON string representation of the object
print(SpaceMembership.to_json())

# convert the object into a dict
space_membership_dict = space_membership_instance.to_dict()
# create an instance of SpaceMembership from a dict
space_membership_from_dict = SpaceMembership.from_dict(space_membership_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


