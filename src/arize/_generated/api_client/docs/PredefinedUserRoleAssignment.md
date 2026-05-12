# PredefinedUserRoleAssignment

A predefined account-level role assignment.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**UserRoleAssignmentType**](UserRoleAssignmentType.md) | Discriminator identifying this as a predefined role assignment. Must be &#x60;predefined&#x60;. | 
**name** | [**UserRole**](UserRole.md) |  | 

## Example

```python
from arize._generated.api_client.models.predefined_user_role_assignment import PredefinedUserRoleAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of PredefinedUserRoleAssignment from a JSON string
predefined_user_role_assignment_instance = PredefinedUserRoleAssignment.from_json(json)
# print the JSON string representation of the object
print(PredefinedUserRoleAssignment.to_json())

# convert the object into a dict
predefined_user_role_assignment_dict = predefined_user_role_assignment_instance.to_dict()
# create an instance of PredefinedUserRoleAssignment from a dict
predefined_user_role_assignment_from_dict = PredefinedUserRoleAssignment.from_dict(predefined_user_role_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


