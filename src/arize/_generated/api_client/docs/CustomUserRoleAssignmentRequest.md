# CustomUserRoleAssignmentRequest

A custom RBAC role assignment in a write request (strict form of CustomUserRoleAssignment).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**UserRoleAssignmentType**](UserRoleAssignmentType.md) | Discriminator identifying this as a custom role assignment. Must be &#x60;CUSTOM&#x60;. | 
**id** | **str** | The unique identifier of the custom RBAC role. | 

## Example

```python
from arize._generated.api_client.models.custom_user_role_assignment_request import CustomUserRoleAssignmentRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CustomUserRoleAssignmentRequest from a JSON string
custom_user_role_assignment_request_instance = CustomUserRoleAssignmentRequest.from_json(json)
# print the JSON string representation of the object
print(CustomUserRoleAssignmentRequest.to_json())

# convert the object into a dict
custom_user_role_assignment_request_dict = custom_user_role_assignment_request_instance.to_dict()
# create an instance of CustomUserRoleAssignmentRequest from a dict
custom_user_role_assignment_request_from_dict = CustomUserRoleAssignmentRequest.from_dict(custom_user_role_assignment_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


