# CreateSamlRoleMappingRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**attributes** | **Dict[str, str]** | SAML attribute names and exact values that select this mapping. | 
**is_account_admin** | **bool** | Whether matching users receive account administrator access. | 
**organization** | [**SamlOrganizationAssignment**](SamlOrganizationAssignment.md) | Organization role assignment, or null when no organization role is assigned. | [optional] 
**space_roles** | [**List[SamlSpaceAssignment]**](SamlSpaceAssignment.md) | Space role assignments for matching users. An empty array grants no space roles. Non-empty assignments require organization. | [optional] 
**project_roles** | [**List[SamlProjectAssignment]**](SamlProjectAssignment.md) | Project role assignments. Omit to use the default unmanaged state on create. Null leaves this mapping unmanaged. An empty array manages project roles with no assignments. Non-empty assignments require organization. If any mapping in this identity provider supplies an array (including an empty array), SAML replaces all project role bindings for users at login when sync_user_roles is enabled, using their matched mappings. This also replaces manually assigned project roles. Null leaves this mapping unmanaged; manual project roles are preserved only when every mapping is unmanaged. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_saml_role_mapping_request import CreateSamlRoleMappingRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateSamlRoleMappingRequest from a JSON string
create_saml_role_mapping_request_instance = CreateSamlRoleMappingRequest.from_json(json)
# print the JSON string representation of the object
print(CreateSamlRoleMappingRequest.to_json())

# convert the object into a dict
create_saml_role_mapping_request_dict = create_saml_role_mapping_request_instance.to_dict()
# create an instance of CreateSamlRoleMappingRequest from a dict
create_saml_role_mapping_request_from_dict = CreateSamlRoleMappingRequest.from_dict(create_saml_role_mapping_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


