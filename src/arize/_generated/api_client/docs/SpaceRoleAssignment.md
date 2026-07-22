# SpaceRoleAssignment

Specifies which role to assign within a space. Discriminated by `type`: - `PREDEFINED`: a built-in platform role — `{ \"type\": \"PREDEFINED\", \"name\": \"ADMIN\" | \"MEMBER\" | \"READ_ONLY\" | \"ANNOTATOR\" }` - `CUSTOM`: a custom RBAC role identified by its ID — `{ \"type\": \"CUSTOM\", \"id\": \"<encoded-role-id>\" }`  Used wherever a space-level role assignment is required (memberships, service key bindings, etc.). 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**SpaceRoleAssignmentType**](SpaceRoleAssignmentType.md) | Discriminator identifying this as a custom RBAC role assignment. Must be &#x60;CUSTOM&#x60;. | 
**name** | **str** | Human-readable name of the custom role. Returned in responses only; ignored on input.  | [readonly] 
**id** | **str** | The unique identifier of the custom RBAC role. | 

## Example

```python
from arize._generated.api_client.models.space_role_assignment import SpaceRoleAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of SpaceRoleAssignment from a JSON string
space_role_assignment_instance = SpaceRoleAssignment.from_json(json)
# print the JSON string representation of the object
print(SpaceRoleAssignment.to_json())

# convert the object into a dict
space_role_assignment_dict = space_role_assignment_instance.to_dict()
# create an instance of SpaceRoleAssignment from a dict
space_role_assignment_from_dict = SpaceRoleAssignment.from_dict(space_role_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


