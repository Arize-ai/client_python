# OrganizationPredefinedRoleAssignment

A predefined organization role assignment.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**OrganizationRoleAssignmentType**](OrganizationRoleAssignmentType.md) |  | 
**name** | [**OrganizationRole**](OrganizationRole.md) |  | 

## Example

```python
from arize._generated.api_client.models.organization_predefined_role_assignment import OrganizationPredefinedRoleAssignment

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationPredefinedRoleAssignment from a JSON string
organization_predefined_role_assignment_instance = OrganizationPredefinedRoleAssignment.from_json(json)
# print the JSON string representation of the object
print(OrganizationPredefinedRoleAssignment.to_json())

# convert the object into a dict
organization_predefined_role_assignment_dict = organization_predefined_role_assignment_instance.to_dict()
# create an instance of OrganizationPredefinedRoleAssignment from a dict
organization_predefined_role_assignment_from_dict = OrganizationPredefinedRoleAssignment.from_dict(organization_predefined_role_assignment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


