# SpaceRoleAssignment

A role assignment for a space membership. Discriminated by `type`: - `PREDEFINED`: one of the predefined roles (`ADMIN`, `MEMBER`, `READ_ONLY`, `ANNOTATOR`) - `CUSTOM`: a custom RBAC role identified by its ID 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**SpaceRoleAssignmentType**](SpaceRoleAssignmentType.md) | Discriminator identifying this as a custom RBAC role assignment. Always &#x60;CUSTOM&#x60; for this variant. | 
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


