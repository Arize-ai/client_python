# SpaceMembershipInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**user_id** | **str** | The unique identifier of the user to add | 
**role** | [**SpaceRoleAssignment**](SpaceRoleAssignment.md) |  | 

## Example

```python
from arize._generated.api_client.models.space_membership_input import SpaceMembershipInput

# TODO update the JSON string below
json = "{}"
# create an instance of SpaceMembershipInput from a JSON string
space_membership_input_instance = SpaceMembershipInput.from_json(json)
# print the JSON string representation of the object
print(SpaceMembershipInput.to_json())

# convert the object into a dict
space_membership_input_dict = space_membership_input_instance.to_dict()
# create an instance of SpaceMembershipInput from a dict
space_membership_input_from_dict = SpaceMembershipInput.from_dict(space_membership_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


