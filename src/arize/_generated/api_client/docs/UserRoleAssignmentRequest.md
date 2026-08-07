# UserRoleAssignmentRequest

Strict request form of UserRoleAssignment. Used in write request bodies. - `PREDEFINED`: `{ \"type\": \"PREDEFINED\", \"name\": \"ADMIN\" | \"MEMBER\" | \"ANNOTATOR\" }` - `CUSTOM`: `{ \"type\": \"CUSTOM\", \"id\": \"<encoded-role-id>\" }` 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**UserRoleAssignmentType**](UserRoleAssignmentType.md) | Discriminator identifying this as a custom role assignment. Must be &#x60;CUSTOM&#x60;. | 
**name** | [**UserRole**](UserRole.md) |  | 
**id** | **str** | The unique identifier of the custom RBAC role. | 

## Example

```python
from arize._generated.api_client.models.user_role_assignment_request import UserRoleAssignmentRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UserRoleAssignmentRequest from a JSON string
user_role_assignment_request_instance = UserRoleAssignmentRequest.from_json(json)
# print the JSON string representation of the object
print(UserRoleAssignmentRequest.to_json())

# convert the object into a dict
user_role_assignment_request_dict = user_role_assignment_request_instance.to_dict()
# create an instance of UserRoleAssignmentRequest from a dict
user_role_assignment_request_from_dict = UserRoleAssignmentRequest.from_dict(user_role_assignment_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


