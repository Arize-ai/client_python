# OrganizationBuiltinRoleAssignment

A builtin (predefined) organization role assignment.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**OrganizationRoleAssignmentType**](OrganizationRoleAssignmentType.md) |  | 
**name** | [**OrganizationRole**](OrganizationRole.md) |  | 

## Example

```python
from arize._generated.api_client.models.organization_builtin_role_assignment import OrganizationBuiltinRoleAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationBuiltinRoleAssignment from a JSON string
organization_builtin_role_assignment_instance = OrganizationBuiltinRoleAssignment.from_json(json)
# print the JSON string representation of the object
print(OrganizationBuiltinRoleAssignment.to_json())

# convert the object into a dict
organization_builtin_role_assignment_dict = organization_builtin_role_assignment_instance.to_dict()
# create an instance of OrganizationBuiltinRoleAssignment from a dict
organization_builtin_role_assignment_from_dict = OrganizationBuiltinRoleAssignment.from_dict(organization_builtin_role_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


