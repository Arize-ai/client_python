# OrganizationPredefinedRoleAssignmentRequest

A predefined organization role assignment in a write request (strict form of OrganizationPredefinedRoleAssignment).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**OrganizationRoleAssignmentType**](OrganizationRoleAssignmentType.md) | Discriminator identifying this as a predefined role assignment. Always &#x60;PREDEFINED&#x60; for this variant. | 
**name** | [**OrganizationRole**](OrganizationRole.md) |  | 

## Example

```python
from arize._generated.api_client.models.organization_predefined_role_assignment_request import OrganizationPredefinedRoleAssignmentRequest

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationPredefinedRoleAssignmentRequest from a JSON string
organization_predefined_role_assignment_request_instance = OrganizationPredefinedRoleAssignmentRequest.from_json(json)
# print the JSON string representation of the object
print(OrganizationPredefinedRoleAssignmentRequest.to_json())

# convert the object into a dict
organization_predefined_role_assignment_request_dict = organization_predefined_role_assignment_request_instance.to_dict()
# create an instance of OrganizationPredefinedRoleAssignmentRequest from a dict
organization_predefined_role_assignment_request_from_dict = OrganizationPredefinedRoleAssignmentRequest.from_dict(organization_predefined_role_assignment_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


