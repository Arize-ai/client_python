# OrganizationCustomRoleAssignmentRequest

A custom RBAC role assignment in a write request (strict form of OrganizationCustomRoleAssignment).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**OrganizationRoleAssignmentType**](OrganizationRoleAssignmentType.md) | Discriminator identifying this as a custom RBAC role assignment. Always &#x60;CUSTOM&#x60; for this variant. | 
**id** | **str** | The unique identifier of the custom RBAC role. | 

## Example

```python
from arize._generated.api_client.models.organization_custom_role_assignment_request import OrganizationCustomRoleAssignmentRequest

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationCustomRoleAssignmentRequest from a JSON string
organization_custom_role_assignment_request_instance = OrganizationCustomRoleAssignmentRequest.from_json(json)
# print the JSON string representation of the object
print(OrganizationCustomRoleAssignmentRequest.to_json())

# convert the object into a dict
organization_custom_role_assignment_request_dict = organization_custom_role_assignment_request_instance.to_dict()
# create an instance of OrganizationCustomRoleAssignmentRequest from a dict
organization_custom_role_assignment_request_from_dict = OrganizationCustomRoleAssignmentRequest.from_dict(organization_custom_role_assignment_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


