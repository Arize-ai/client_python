# PredefinedUserRoleAssignmentRequest

A predefined account-level role assignment in a write request (strict form of PredefinedUserRoleAssignment).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**UserRoleAssignmentType**](UserRoleAssignmentType.md) | Discriminator identifying this as a predefined role assignment. Must be &#x60;PREDEFINED&#x60;. | 
**name** | [**UserRole**](UserRole.md) |  | 

## Example

```python
from arize._generated.api_client.models.predefined_user_role_assignment_request import PredefinedUserRoleAssignmentRequest

# TODO update the JSON string below
json = "{}"
# create an instance of PredefinedUserRoleAssignmentRequest from a JSON string
predefined_user_role_assignment_request_instance = PredefinedUserRoleAssignmentRequest.from_json(json)
# print the JSON string representation of the object
print(PredefinedUserRoleAssignmentRequest.to_json())

# convert the object into a dict
predefined_user_role_assignment_request_dict = predefined_user_role_assignment_request_instance.to_dict()
# create an instance of PredefinedUserRoleAssignmentRequest from a dict
predefined_user_role_assignment_request_from_dict = PredefinedUserRoleAssignmentRequest.from_dict(predefined_user_role_assignment_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


