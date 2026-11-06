# SamlRoleMapping


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**attributes** | **Dict[str, str]** | SAML attribute names and exact values that select this mapping. | 
**is_account_admin** | **bool** | Whether matching users receive account administrator access. | 
**organization** | [**SamlOrganizationAssignment**](SamlOrganizationAssignment.md) | Organization role assignment, or null when no organization role is assigned. | [optional] 
**space_roles** | [**List[SamlSpaceAssignment]**](SamlSpaceAssignment.md) | Space role assignments for matching users. An empty array grants no space roles. | 
**project_roles** | [**List[SamlProjectAssignment]**](SamlProjectAssignment.md) | Configured project role assignments, or null when this mapping is unmanaged. If any mapping in this identity provider supplies an array (including an empty array), SAML replaces all project role bindings for users at login when sync_user_roles is enabled, using their matched mappings. This also replaces manually assigned project roles. Null leaves this mapping unmanaged; manual project roles are preserved only when every mapping is unmanaged. | 
**id** | **str** | Persistent server-generated mapping ID. File-managed mapping IDs identify the current file configuration. If a successful write returns null, read the configuration again before using the mapping ID. | 

## Example

```python
from arize._generated.api_client.models.saml_role_mapping import SamlRoleMapping

# TODO update the JSON string below
json = "{}"
# create an instance of SamlRoleMapping from a JSON string
saml_role_mapping_instance = SamlRoleMapping.from_json(json)
# print the JSON string representation of the object
print(SamlRoleMapping.to_json())

# convert the object into a dict
saml_role_mapping_dict = saml_role_mapping_instance.to_dict()
# create an instance of SamlRoleMapping from a dict
saml_role_mapping_from_dict = SamlRoleMapping.from_dict(saml_role_mapping_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


