# OrganizationCustomRoleAssignment

A custom RBAC role assignment.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**OrganizationRoleAssignmentType**](OrganizationRoleAssignmentType.md) |  | 
**id** | **str** | The unique identifier of the custom RBAC role. | 
**name** | **str** | Human-readable name of the custom role. Returned in responses only; ignored on input.  | [optional] [readonly] 

## Example

```python
from arize._generated.api_client.models.organization_custom_role_assignment import OrganizationCustomRoleAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationCustomRoleAssignment from a JSON string
organization_custom_role_assignment_instance = OrganizationCustomRoleAssignment.from_json(json)
# print the JSON string representation of the object
print(OrganizationCustomRoleAssignment.to_json())

# convert the object into a dict
organization_custom_role_assignment_dict = organization_custom_role_assignment_instance.to_dict()
# create an instance of OrganizationCustomRoleAssignment from a dict
organization_custom_role_assignment_from_dict = OrganizationCustomRoleAssignment.from_dict(organization_custom_role_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


