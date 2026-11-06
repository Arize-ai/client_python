# UpdateSamlRoleMappingRequest

Omitted fields remain unchanged. Attributes and assignment lists replace their complete values. Empty lists clear assignments. organization:null clears the organization only if the resulting mapping remains valid. project_roles:null leaves this mapping unmanaged; manual project roles are preserved only when every mapping is unmanaged.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**attributes** | **Dict[str, str]** | SAML attribute names and exact values that select this mapping. | [optional] 
**is_account_admin** | **bool** | Whether matching users receive account administrator access. | [optional] 
**organization** | [**SamlOrganizationAssignment**](SamlOrganizationAssignment.md) | Organization role assignment, or null when no organization role is assigned. | [optional] 
**space_roles** | [**List[SamlSpaceAssignment]**](SamlSpaceAssignment.md) | Space role assignments for matching users. An empty array grants no space roles. | [optional] 
**project_roles** | [**List[SamlProjectAssignment]**](SamlProjectAssignment.md) | Omit to leave this mapping unchanged. An array manages project roles; an empty array clears this mapping&#39;s project assignments. Null leaves this mapping unmanaged. With role sync enabled, any managed mapping enables project-role synchronization for the identity provider. Login replaces project roles using matched mappings, including removal of manual grants. Manual project roles are preserved only when every mapping is unmanaged. | [optional] 

## Example

```python
from arize._generated.api_client.models.update_saml_role_mapping_request import UpdateSamlRoleMappingRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateSamlRoleMappingRequest from a JSON string
update_saml_role_mapping_request_instance = UpdateSamlRoleMappingRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateSamlRoleMappingRequest.to_json())

# convert the object into a dict
update_saml_role_mapping_request_dict = update_saml_role_mapping_request_instance.to_dict()
# create an instance of UpdateSamlRoleMappingRequest from a dict
update_saml_role_mapping_request_from_dict = UpdateSamlRoleMappingRequest.from_dict(update_saml_role_mapping_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


