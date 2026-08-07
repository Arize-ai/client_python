# OrganizationRoleAssignmentRequest

Strict request form of OrganizationRoleAssignment. Used in write request bodies. - `PREDEFINED`: `{ \"type\": \"PREDEFINED\", \"name\": \"ADMIN\" | \"MEMBER\" | \"READ_ONLY\" | \"ANNOTATOR\" }` - `CUSTOM`: `{ \"type\": \"CUSTOM\", \"id\": \"<encoded-role-id>\" }` 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**OrganizationRoleAssignmentType**](OrganizationRoleAssignmentType.md) | Discriminator identifying this as a custom RBAC role assignment. Always &#x60;CUSTOM&#x60; for this variant. | 
**name** | [**OrganizationRole**](OrganizationRole.md) |  | 
**id** | **str** | The unique identifier of the custom RBAC role. | 

## Example

```python
from arize._generated.api_client.models.organization_role_assignment_request import OrganizationRoleAssignmentRequest

# TODO update the JSON string below
json = "{}"
# create an instance of OrganizationRoleAssignmentRequest from a JSON string
organization_role_assignment_request_instance = OrganizationRoleAssignmentRequest.from_json(json)
# print the JSON string representation of the object
print(OrganizationRoleAssignmentRequest.to_json())

# convert the object into a dict
organization_role_assignment_request_dict = organization_role_assignment_request_instance.to_dict()
# create an instance of OrganizationRoleAssignmentRequest from a dict
organization_role_assignment_request_from_dict = OrganizationRoleAssignmentRequest.from_dict(organization_role_assignment_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


