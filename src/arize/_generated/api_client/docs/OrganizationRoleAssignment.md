# OrganizationRoleAssignment

A role assignment for an organization membership. Discriminated by `type`: - `PREDEFINED`: one of the predefined roles (`ADMIN`, `MEMBER`, `READ_ONLY`, `ANNOTATOR`) - `CUSTOM`: a custom RBAC role identified by its ID 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**OrganizationRoleAssignmentType**](OrganizationRoleAssignmentType.md) | Discriminator identifying this as a custom RBAC role assignment. Always &#x60;CUSTOM&#x60; for this variant. | 
**name** | **str** | Human-readable name of the custom role. Returned in responses only; ignored on input.  | [readonly] 
**id** | **str** | The unique identifier of the custom RBAC role. | 

## Example

```python
from arize._generated.api_client.models.organization_role_assignment import OrganizationRoleAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationRoleAssignment from a JSON string
organization_role_assignment_instance = OrganizationRoleAssignment.from_json(json)
# print the JSON string representation of the object
print(OrganizationRoleAssignment.to_json())

# convert the object into a dict
organization_role_assignment_dict = organization_role_assignment_instance.to_dict()
# create an instance of OrganizationRoleAssignment from a dict
organization_role_assignment_from_dict = OrganizationRoleAssignment.from_dict(organization_role_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


