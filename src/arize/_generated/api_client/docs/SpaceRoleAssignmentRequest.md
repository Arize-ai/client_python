# SpaceRoleAssignmentRequest

Strict request form of SpaceRoleAssignment. Used in write request bodies. - `PREDEFINED`: `{ \"type\": \"PREDEFINED\", \"name\": \"ADMIN\" | \"MEMBER\" | \"READ_ONLY\" | \"ANNOTATOR\" }` - `CUSTOM`: `{ \"type\": \"CUSTOM\", \"id\": \"<encoded-role-id>\" }` 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**SpaceRoleAssignmentType**](SpaceRoleAssignmentType.md) | Discriminator identifying this as a custom RBAC role assignment. Must be &#x60;CUSTOM&#x60;. | 
**name** | [**UserSpaceRole**](UserSpaceRole.md) |  | 
**id** | **str** | The unique identifier of the custom RBAC role. | 

## Example

```python
from arize._generated.api_client.models.space_role_assignment_request import SpaceRoleAssignmentRequest

# TODO update the JSON string below
json = "{}"
# create an instance of SpaceRoleAssignmentRequest from a JSON string
space_role_assignment_request_instance = SpaceRoleAssignmentRequest.from_json(json)
# print the JSON string representation of the object
print(SpaceRoleAssignmentRequest.to_json())

# convert the object into a dict
space_role_assignment_request_dict = space_role_assignment_request_instance.to_dict()
# create an instance of SpaceRoleAssignmentRequest from a dict
space_role_assignment_request_from_dict = SpaceRoleAssignmentRequest.from_dict(space_role_assignment_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


