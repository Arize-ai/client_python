# ApiKeyRoles

Role assignments for the bot user created with a service key.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**space_role** | [**ApiKeySpaceRole**](ApiKeySpaceRole.md) | Role to assign the bot user within the space. Defaults to &#x60;MEMBER&#x60; when omitted. Must be at or below the caller&#39;s own effective space role.  | [optional] 
**org_role** | [**ApiKeyOrganizationRole**](ApiKeyOrganizationRole.md) | Role to assign the bot user within the organization. Defaults to &#x60;READ_ONLY&#x60; when omitted. Must be at or below the caller&#39;s own organization role.  | [optional] 
**account_role** | [**ApiKeyAccountRole**](ApiKeyAccountRole.md) | Account-level role to assign the bot user. Defaults to &#x60;MEMBER&#x60; when omitted. Must be at or below the caller&#39;s own account role.  | [optional] 

## Example

```python
from arize._generated.api_client.models.api_key_roles import ApiKeyRoles

# TODO update the JSON string below
json = "{}"
# create an instance of ApiKeyRoles from a JSON string
api_key_roles_instance = ApiKeyRoles.from_json(json)
# print the JSON string representation of the object
print(ApiKeyRoles.to_json())

# convert the object into a dict
api_key_roles_dict = api_key_roles_instance.to_dict()
# create an instance of ApiKeyRoles from a dict
api_key_roles_from_dict = ApiKeyRoles.from_dict(api_key_roles_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


