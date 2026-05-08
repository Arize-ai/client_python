# CustomUserRoleAssignment

A custom RBAC role assignment.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**UserRoleAssignmentType**](UserRoleAssignmentType.md) | Discriminator identifying this as a custom role assignment. Must be &#x60;custom&#x60;. | 
**id** | **str** | The unique identifier of the custom RBAC role. | 
**name** | **str** | Human-readable name of the custom role. Returned in responses only; ignored on input.  | [optional] [readonly] 

## Example

```python
from arize._generated.api_client.models.custom_user_role_assignment import CustomUserRoleAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of CustomUserRoleAssignment from a JSON string
custom_user_role_assignment_instance = CustomUserRoleAssignment.from_json(json)
# print the JSON string representation of the object
print(CustomUserRoleAssignment.to_json())

# convert the object into a dict
custom_user_role_assignment_dict = custom_user_role_assignment_instance.to_dict()
# create an instance of CustomUserRoleAssignment from a dict
custom_user_role_assignment_from_dict = CustomUserRoleAssignment.from_dict(custom_user_role_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


